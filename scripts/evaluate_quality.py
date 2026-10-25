"""Reproducible development-set retrieval and extractive-answer evaluation.

No chat calls are made. Use --dense with COURSE_RAG_OLLAMA_MODEL=bge-m3:latest.
Query embeddings are cached with model and question fingerprints for offline replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

from course_rag_assistant.answer import _parse_citations, answer
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


def answer_metrics(retriever, questions, k):
    rankings = retriever.search_many([q["question"] for q in questions], top_k=k)
    rows = []
    for label, hits in zip(questions, rankings):
        result = answer(label["question"], hits)
        citations = _parse_citations(result.text)
        keys = [f"{doc}#{location.removeprefix('section=')}" for doc, location in citations]
        gold = set(label.get("gold_pages", []))
        compact = lambda text: re.sub(r"\s+", "", text).casefold()
        points = label.get("answer_points", [])
        rows.append({
            "id": label["id"], "question": label["question"], "answer": result.text,
            "abstained": "没有足够证据" in result.text,
            "abstention_reason": result.abstention_reason,
            "matched_points": [p for p in points if compact(p) in compact(result.text)],
            "point_count": len(points), "citation_count": len(keys),
            "gold_page_citations": sum(key in gold for key in keys),
            "unsupported_claims": result.unsupported_claims,
        })
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
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/datamining-notes"))
    parser.add_argument("--labels", type=Path, default=Path("eval/real-questions-v2.jsonl"))
    parser.add_argument("--unanswerable", type=Path, default=Path("eval/real-unanswerable-v2.jsonl"))
    parser.add_argument("--index", type=Path, default=Path("data/index/embeddings.npy"))
    parser.add_argument("--cache", type=Path, default=Path("data/index/quality-query-v2.npy"))
    parser.add_argument("--output", type=Path, default=Path("eval/reports/quality-v5.json"))
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--model", default="bge-m3:latest", help="Expected embedding model, including on cache replay")
    args = parser.parse_args()
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
    report = {
        "schema_version": 2, "benchmark": "d2l-v2-development-not-held-out",
        "answerability_policy": "offline_coverage_0.5_scoped_record_values_conflicts_and_exact_api_symbols",
        "code_sha256": {str(p): digest(p) for p in [
            Path("scripts/evaluate_quality.py"), *sorted(Path("src/course_rag_assistant").glob("*.py"))]},
        "labels_sha256": digest(args.labels), "unanswerable_sha256": digest(args.unanswerable),
        "corpus_sha256": {str(p.relative_to(args.data)): digest(p) for p in sorted(args.data.rglob("*.pdf"))},
        "chunks": len(chunks), "generation": "offline_extractive_complete_source_sentences",
        "notes": [
            "Agent-checked evidence; independent human review and held-out evaluation are still required.",
            "Gold pages are selected reference evidence, not exhaustive relevant-page judgments.",
            "Page citation precision is not semantic entailment; keyword coverage is not answer correctness.",
            "Non-abstention on negatives means unsupported answering, not necessarily fabricated text.",
            "Timing excludes parsing and query embedding generation; cached vectors are reused across modes.",
        ], "skipped": skipped, "results": {},
    }
    if args.dense:
        report.update(model=args.model, index_sha256=digest(args.index),
                      index_metadata_sha256=digest(args.index.with_suffix('.json')),
                      query_vectors_sha256=digest(args.cache))
    for name, retriever in retrievers.items():
        start = time.perf_counter()
        retrieval = evaluate_retriever(retriever, labels, 8, include_details=True)
        positive = answer_metrics(retriever, labels, 8)
        negative = answer_metrics(retriever, negatives, 8)
        report["results"][name] = {"retrieval": retrieval, "answerable": positive,
                                   "unanswerable": negative, "evaluation_seconds": time.perf_counter() - start}
        print(name, "Recall@8", round(retrieval['recall_at_k'], 4), "MRR@8", round(retrieval['mrr'], 4),
              "negative abstention", round(negative['abstention_rate'], 4), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved", args.output)


if __name__ == "__main__":
    main()

