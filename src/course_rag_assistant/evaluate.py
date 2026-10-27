"""Evaluation: prove the system retrieves and cites correctly.

Labeled set format (eval/labeled-questions.jsonl, one JSON object per line):
    {"question": "...", "gold_pages": ["notes.md#sec"], "answer_points": ["..."]}

Metrics:
- retrieval: recall@k, MRR over gold chunks
- citation: precision of [doc, p.N] citations against gold passages
- abstention: on unanswerable questions, did it refuse instead of hallucinating?
- ablation: dense-only vs hybrid vs +rerank on the same set (the portfolio's
  most persuasive table)
"""
from __future__ import annotations

import json
from pathlib import Path

from .answer import TextGenerator, answer
from .config import get_settings
from .ingest import Chunk, ingest_dir
from .retrieval import HybridRetriever, LexicalReranker


def load_labeled(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / max(len(gold), 1)


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    for rank, key in enumerate(ranked, start=1):
        if key in gold:
            return 1.0 / rank
    return 0.0


def chunk_keys(chunk: Chunk) -> set[str]:
    """Return label keys accepted for a chunk's page/section provenance."""
    keys = {f"{chunk.doc}#{chunk.section}"} if chunk.section else set()
    if chunk.section and " / " in chunk.section:
        keys.add(f"{chunk.doc}#{chunk.section.rsplit(' / ', 1)[-1]}")
    if chunk.page:
        keys.add(f"{chunk.doc}#p.{chunk.page}")
        keys.add(f"{chunk.doc}#{chunk.page}")
    return keys


def evaluate_retriever(
    retriever: HybridRetriever, questions: list[dict], k: int = 8, *, include_details: bool = False
) -> dict:
    recalls: list[float] = []
    mrrs: list[float] = []
    query_list = [item["question"] for item in questions]
    hit_lists = retriever.search_many(query_list, top_k=k) if hasattr(retriever, "search_many") else [retriever.search(q, top_k=k) for q in query_list]
    if len(hit_lists) != len(questions):
        raise ValueError("Retrieval returned a different number of query results")
    details = []
    for item, hits in zip(questions, hit_lists):
        gold = set(item.get("gold_pages", []))
        ranked = [
            next((key for key in chunk_keys(hit.chunk) if key in gold), next(iter(chunk_keys(hit.chunk)), ""))
            for hit in hits
        ]
        recalls.append(recall_at_k(ranked, gold, k))
        mrrs.append(reciprocal_rank(ranked, gold))
        if include_details:
            details.append({
                "question": item["question"],
                "gold_pages": sorted(gold),
                "recall_at_k": recalls[-1],
                "reciprocal_rank": mrrs[-1],
                "hits": [
                    {"doc": h.chunk.doc, "page": h.chunk.page, "section": h.chunk.section,
                     "score": h.score, "stage": h.stage}
                    for h in hits
                ],
            })
    count = len(questions)
    result = {
        "questions": count,
        "recall_at_k": sum(recalls) / max(count, 1),
        "mrr": sum(mrrs) / max(count, 1),
        "k": k,
    }
    if include_details:
        result["details"] = details
    return result


def evaluate_ablation(
    retrievers: dict[str, HybridRetriever], questions: list[dict], k: int = 8
) -> dict[str, dict]:
    """Run identical labels through multiple retrieval configurations."""
    return {name: evaluate_retriever(retriever, questions, k) for name, retriever in retrievers.items()}


def evaluate_abstention(
    retriever: HybridRetriever, questions: list[dict], generator: TextGenerator | None = None, k: int = 8
) -> dict:
    """Measure refusal rate for labels representing questions outside the corpus."""
    refused = 0
    for item in questions:
        result = answer(item["question"], retriever.search(item["question"], top_k=k), generator)
        if "没有足够证据" in result.text:
            refused += 1
    total = len(questions)
    return {"questions": total, "refused": refused, "non_abstained": total - refused,
            "abstention_rate": refused / max(total, 1)}


def run_eval(labeled_path: Path, data_dir: Path = Path("data/samples"), k: int = 8) -> dict:
    """Evaluate the local retriever against a JSONL labeled set."""
    questions = load_labeled(labeled_path)
    retriever = HybridRetriever(ingest_dir(data_dir))
    return evaluate_retriever(retriever, questions, k)


def run_ablation(labeled_path: Path, data_dir: Path = Path("data/samples"), k: int = 8) -> dict[str, dict]:
    questions = load_labeled(labeled_path)
    chunks = ingest_dir(data_dir)
    retrievers = {
        "bm25": HybridRetriever(chunks),
        "bm25+lexical_rerank": HybridRetriever(chunks, reranker=LexicalReranker()),
    }
    # Add remote stages when both a persisted dense index and its configured
    # embedding backend are available; offline evaluation remains deterministic.
    from .dense import NumpyVectorIndex, embedder_from_env
    from .rerank import reranker_from_env
    dense_path = get_settings().index_dir / "embeddings.npy"
    backend = embedder_from_env()
    if dense_path.exists() and backend is not None:
        dense = NumpyVectorIndex.load(dense_path)
        retrievers["bm25+dense"] = HybridRetriever(chunks, dense, backend)
        reranker = reranker_from_env()
        if reranker is not None:
            retrievers["bm25+dense+rerank"] = HybridRetriever(chunks, dense, backend, reranker)
    return evaluate_ablation(retrievers, questions, k)


def run_abstention(unanswerable_path: Path, data_dir: Path = Path("data/samples"), k: int = 8) -> dict:
    questions = load_labeled(unanswerable_path)
    return evaluate_abstention(HybridRetriever(ingest_dir(data_dir)), questions, k=k)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate retrieval against a JSONL label set")
    parser.add_argument("labels", nargs="?", type=Path, default=Path("eval/labeled-questions.jsonl"))
    parser.add_argument("--data", type=Path, default=Path("data/samples"))
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--ablation", action="store_true", help="Compare BM25 with lexical reranking")
    parser.add_argument("--unanswerable", action="store_true", help="Evaluate refusal on unanswerable labels")
    args = parser.parse_args()
    if args.unanswerable:
        result = run_abstention(args.labels, args.data, args.k)
    elif args.ablation:
        result = run_ablation(args.labels, args.data, args.k)
    else:
        result = run_eval(args.labels, args.data, args.k)
    print(result)
