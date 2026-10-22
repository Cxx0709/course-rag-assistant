"""Persist BM25 vs dense hybrid metrics and per-question rankings."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from course_rag_assistant.dense import NumpyVectorIndex, embedder_from_env
from course_rag_assistant.evaluate import evaluate_retriever, load_labeled
from course_rag_assistant.ingest import ingest_dir
from course_rag_assistant.retrieval import HybridRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("eval/real-labeled-questions.jsonl"))
    parser.add_argument("--data", type=Path, default=Path("data/raw/datamining-notes"))
    parser.add_argument("--index", type=Path, default=Path("data/index/embeddings.npy"))
    parser.add_argument("--output", type=Path, default=Path("eval/reports/dense-ablation.json"))
    args = parser.parse_args()
    backend = embedder_from_env()
    if backend is None:
        parser.error("Configure COURSE_RAG_OLLAMA_MODEL or COURSE_RAG_EMBEDDING_*")
    chunks = ingest_dir(args.data)
    dense = NumpyVectorIndex.load(args.index)
    if chunks != dense.chunks:
        parser.error("Dense index does not match the current corpus; rebuild it first")
    questions = load_labeled(args.labels)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "labels": str(args.labels),
        "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256(args.index.read_bytes()).hexdigest(),
        "chunks": len(chunks),
        "dimensions": dense.vectors.shape[1],
        "model": getattr(backend, "model", type(backend).__name__),
        "notes": [
            "Exploratory 10-question labels; not an independently reviewed benchmark.",
            "Gold keys use PDF physical page numbers. MRR is truncated at k=8.",
            "Retrieval metrics do not measure answer correctness or citation entailment.",
        ],
        "results": {},
    }
    for name, retriever in {
        "bm25": HybridRetriever(chunks),
        "bm25+dense": HybridRetriever(chunks, dense, backend),
    }.items():
        print(f"Evaluating {name} on {len(questions)} questions...", flush=True)
        result = evaluate_retriever(retriever, questions, 8, include_details=True)
        report["results"][name] = result
        print(f"recall@8={result['recall_at_k']:.4f}, MRR@8={result['mrr']:.4f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
