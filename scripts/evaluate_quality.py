"""Reproducible development-set retrieval and answer evaluation.

Offline by default; --chat explicitly enables the configured chat generator.
Use --dense with COURSE_RAG_OLLAMA_MODEL=bge-m3:latest.
Query embeddings are cached with model and question fingerprints for offline replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from course_rag_assistant.answer import _parse_citations, answer, generator_from_env
from course_rag_assistant.content import navigation_reason
from course_rag_assistant.dense import NumpyVectorIndex, embedder_from_env
from course_rag_assistant.evaluate import evaluate_retriever, load_labeled
from course_rag_assistant.ingest import ingest_dir
from course_rag_assistant.rerank import reranker_from_env
from course_rag_assistant.retrieval import Hit, HybridRetriever


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CachedEmbedder:
    def __init__(self, queries, vectors):
        self.vectors = dict(zip(queries, vectors))

    def embed(self, texts):
        return np.asarray([self.vectors[text] for text in texts])


class DenseRetriever:
    def __init__(self, index, backend):
        self.index, self.backend = index, backend

    def search_many(self, queries, top_k=8):
        return [[Hit(c, s, "dense") for c, s in hits]
                for hits in self.index.search_many(queries, self.backend, top_k)]


class GenerationTrace:
    """Observe the real generator without changing the application's fallback."""

    def __init__(self, generator):
        self.generator = generator
        self.called = False
        self.raw_output = None
        self.error_type = None
        self.seconds = 0.0

    def generate(self, prompt):
        self.called = True
        start = time.perf_counter()
        try:
            self.raw_output = self.generator.generate(prompt)
            return self.raw_output
        except Exception as exc:
            # Error messages can contain credentials or endpoint URLs.
            self.error_type = type(exc).__name__
            raise
        finally:
            self.seconds = time.perf_counter() - start


class RankedResults:
    """Reuse exactly the same retrieved passages for retrieval and answer scores."""

    def __init__(self, queries, rankings):
        if len(queries) != len(rankings):
            raise ValueError("Retrieval returned a different number of query results")
        self.rankings = dict(zip(queries, rankings))

    def search_many(self, queries, top_k=8):
        return [self.rankings[query][:top_k] for query in queries]


def answer_metrics(retriever, questions, k, generator=None, *, progress=False):
    rankings = retriever.search_many([q["question"] for q in questions], top_k=k)
    if len(rankings) != len(questions):
        raise ValueError("Retrieval returned a different number of query results")
    rows = []
    for label, hits in zip(questions, rankings):
        trace = GenerationTrace(generator) if generator is not None else None
        start = time.perf_counter()
        result = answer(label["question"], hits, trace)
        elapsed = time.perf_counter() - start
        status = "offline_extractive"
        if trace is not None:
            status = ("skipped_no_evidence" if not trace.called else
                      "fallback_extractive" if trace.error_type else "model_output_checked")
        citations = _parse_citations(result.text)
        keys = [f"{doc}#{location.removeprefix('section=')}" for doc, location in citations]
        gold = set(label.get("gold_pages", []))
        compact = lambda text: re.sub(r"\s+", "", text).casefold()
        points = label.get("answer_points", [])
        rows.append({
            "id": label["id"], "question": label["question"], "answer": result.text,
            "abstained": "没有足够证据" in result.text,
            "abstention_reason": result.abstention_reason,
            "selected_sentence_ids": result.selected_sentence_ids,
            "matched_points": [p for p in points if compact(p) in compact(result.text)],
            "expected_answer_points": points, "gold_pages": sorted(gold),
            "point_count": len(points), "citation_count": len(keys),
            "gold_page_citations": sum(key in gold for key in keys),
            "unsupported_claims": result.unsupported_claims,
            "generation_status": status,
            "generation_error_type": trace.error_type if trace else None,
            "raw_model_output": trace.raw_output if trace else None,
            "generation_seconds": trace.seconds if trace else 0.0,
            "answer_seconds": elapsed,
            "evidence": [{"doc": h.chunk.doc, "page": h.chunk.page,
                          "section": h.chunk.section, "text": h.chunk.text} for h in result.hits],
        })
        if progress:
            print(label["id"], status, f"{elapsed:.2f}s", flush=True)
    citation_count = sum(r["citation_count"] for r in rows)
    return {
        "questions": len(rows),
        "abstention_rate": sum(r["abstained"] for r in rows) / max(len(rows), 1),
        "abstention_reasons": {reason: sum(r["abstention_reason"] == reason for r in rows)
                               for reason in sorted({r["abstention_reason"] for r in rows if r["abstention_reason"]})},
        "literal_answer_point_coverage": sum(len(r["matched_points"]) / max(r["point_count"], 1)
                                              for r in rows) / max(len(rows), 1),
        "gold_page_citation_precision": (sum(r["gold_page_citations"] for r in rows) / citation_count
                                         if citation_count else None),
        "details": rows,
        "generation_status_counts": {status: sum(r["generation_status"] == status for r in rows)
                                     for status in sorted({r["generation_status"] for r in rows})},
        "model_calls": sum(r["generation_status"] in {"fallback_extractive", "model_output_checked"}
                           for r in rows),
        "generation_failures": sum(r["generation_status"] == "fallback_extractive" for r in rows),
        "model_outputs_fully_rejected": sum(r["generation_status"] == "model_output_checked"
                                            and r["abstained"] and r["abstention_reason"] != "model_abstained"
                                            for r in rows),
        "model_abstentions": sum(r["abstention_reason"] == "model_abstained" for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/datamining-notes"))
    parser.add_argument("--labels", type=Path, default=Path("eval/real-questions-v2.jsonl"))
    parser.add_argument("--unanswerable", type=Path, default=Path("eval/real-unanswerable-v2.jsonl"))
    parser.add_argument("--index", type=Path, default=Path("data/index/embeddings.npy"))
    parser.add_argument("--cache", type=Path, default=Path("data/index/quality-query-v2.npy"))
    parser.add_argument("--output", type=Path, help="Defaults to quality-v8[-chat].json")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--chat", action="store_true", help="Evaluate the configured chat model, including failures")
    parser.add_argument("--modes", nargs="+", choices=["bm25", "dense", "hybrid", "hybrid+rerank"],
                        help="Evaluate only these modes; unavailable requested modes are errors")
    parser.add_argument("--limit", type=int, help="First N questions per split for smoke testing, not acceptance")
    parser.add_argument("--model", default="bge-m3:latest", help="Expected embedding model, including on cache replay")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    generator = generator_from_env() if args.chat else None
    if args.chat and generator is None:
        parser.error("--chat requires a configured chat generator; offline results cannot substitute for it")
    if args.modes and any(mode != "bm25" for mode in args.modes) and not args.dense:
        parser.error("Dense, hybrid and rerank modes require --dense")
    if args.modes and "hybrid+rerank" in args.modes and reranker_from_env() is None:
        parser.error("hybrid+rerank requires a configured reranker endpoint")
    args.output = args.output or Path("eval/reports/quality-v8-chat.json" if args.chat
                                      else "eval/reports/quality-v8.json")
    labels, negatives = load_labeled(args.labels), load_labeled(args.unanswerable)
    chunks = ingest_dir(args.data)
    # Validate reference snippets before measuring retrieval. Whitespace is the
    # only normalization used, so references cannot silently point to wrong pages.
    import pymupdf
    for label in labels:
        doc_name = label["gold_pages"][0].split("#")[0]
        with pymupdf.open(args.data / doc_name) as doc:
            for evidence in label["evidence"]:
                page = re.sub(r"\s+", "", doc[evidence["page"] - 1].get_text())
                if evidence["text"] not in page:
                    raise ValueError(f"Evidence mismatch: {label['id']}")
    retrievers = {"bm25": HybridRetriever(chunks)}
    skipped = {}
    if args.dense:
        index = NumpyVectorIndex.load(args.index)
        if chunks != index.chunks:
            raise ValueError("Dense index differs from current corpus; rebuild it")
        queries = [q["question"] for q in labels + negatives]
        fingerprint = {"model": args.model, "queries": queries}
        metadata = args.cache.with_suffix(".json")
        if args.cache.exists() and metadata.exists() and json.loads(metadata.read_text(encoding="utf-8")) == fingerprint:
            vectors = np.load(args.cache, allow_pickle=False)
        else:
            backend = embedder_from_env()
            if backend is None or backend.model != args.model:
                raise ValueError("Configure an embedding backend matching --model")
            vectors = backend.embed(queries)
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.cache, vectors)
            metadata.write_text(json.dumps(fingerprint, ensure_ascii=False), encoding="utf-8")
        if vectors.shape != (len(queries), index.vectors.shape[1]) or not np.isfinite(vectors).all():
            raise ValueError("Invalid query embedding cache")
        backend = CachedEmbedder(queries, vectors)
        retrievers["dense"] = DenseRetriever(index, backend)
        retrievers["hybrid"] = HybridRetriever(chunks, index, backend)
        reranker = reranker_from_env()
        if reranker is not None:
            retrievers["hybrid+rerank"] = HybridRetriever(chunks, index, backend, reranker)
        else:
            skipped["hybrid+rerank"] = "No configured BGE reranker endpoint; lexical reranking is not a substitute."
    else:
        skipped["dense/hybrid/rerank"] = "Run with --dense and matching embedding backend or cached query vectors."
    if args.modes:
        retrievers = {name: retrievers[name] for name in dict.fromkeys(args.modes)}
    full_counts = {"answerable": len(labels), "unanswerable": len(negatives)}
    if args.limit is not None:
        labels, negatives = labels[:args.limit], negatives[:args.limit]
    report = {
        "schema_version": 3, "benchmark": "d2l-v2-development-not-held-out",
        "run_scope": "smoke_test" if args.limit is not None else "full_development_set",
        "full_dataset_counts": full_counts,
        "selected_ids": {"answerable": [q["id"] for q in labels],
                         "unanswerable": [q["id"] for q in negatives]},
        "requested_modes": args.modes, "completed": False,
        "content_filter": "explicit_navigation_before_top_k_and_sentence_selection_v1",
        "excluded_navigation_chunks": dict(Counter(reason for chunk in chunks
                                                    if (reason := navigation_reason(chunk.text)))),
        "answerability_policy": "offline_coverage_0.5_scoped_record_values_conflicts_and_exact_api_symbols",
        "code_sha256": {str(p): digest(p) for p in [
            Path("scripts/evaluate_quality.py"), *sorted(Path("src/course_rag_assistant").glob("*.py"))]},
        "labels_sha256": digest(args.labels), "unanswerable_sha256": digest(args.unanswerable),
        "corpus_sha256": {str(p.relative_to(args.data)): digest(p) for p in sorted(args.data.rglob("*.pdf"))},
        "chunks": len(chunks), "generation": ("chat_sentence_id_selection_with_exact_verification_and_transport_fallback"
                                                if args.chat else "offline_extractive_complete_source_sentences"),
        "chat_config": ({"backend": type(generator).__name__, "model": generator.model,
                         "timeout": generator.timeout,
                         "num_predict": getattr(generator, "num_predict", None),
                         "think": getattr(generator, "think", None)} if args.chat else None),
        "notes": [
            "Agent-checked evidence; independent human review and held-out evaluation are still required.",
            "Gold pages are selected reference evidence, not exhaustive relevant-page judgments.",
            "Page citation precision is not semantic entailment; keyword coverage is not answer correctness.",
            "Non-abstention on negatives means unsupported answering, not necessarily fabricated text.",
            "Timing excludes parsing and query embedding generation; cached vectors are reused across modes.",
            "Model outputs are subject to exact-source-sentence verification, not free-paraphrase semantic verification.",
            "Fallback answers measure application behavior and must not be reported as successful model generation.",
            "Selected sentence IDs are 1-based over answer._sentence_options(evidence), excluding explicit navigation.",
        ], "skipped": skipped, "results": {},
    }
    if args.dense:
        report.update(model=args.model, index_sha256=digest(args.index),
                      index_metadata_sha256=digest(args.index.with_suffix('.json')),
                      query_vectors_sha256=digest(args.cache))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, retriever in retrievers.items():
        start = time.perf_counter()
        queries = [q["question"] for q in labels + negatives]
        ranked = RankedResults(queries, retriever.search_many(queries, top_k=8))
        retrieval = evaluate_retriever(ranked, labels, 8, include_details=True)
        positive = answer_metrics(ranked, labels, 8, generator, progress=args.chat)
        negative = answer_metrics(ranked, negatives, 8, generator, progress=args.chat)
        report["results"][name] = {"retrieval": retrieval, "answerable": positive,
                                   "unanswerable": negative, "evaluation_seconds": time.perf_counter() - start}
        print(name, "Recall@8", round(retrieval['recall_at_k'], 4), "MRR@8", round(retrieval['mrr'], 4),
              "negative abstention", round(negative['abstention_rate'], 4), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["completed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved", args.output)


if __name__ == "__main__":
    main()

