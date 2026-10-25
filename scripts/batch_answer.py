"""Generate cited answers for a JSONL question set.

Usage:
    python scripts/batch_answer.py eval/labeled-questions.jsonl --data data/samples
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from course_rag_assistant.answer import answer, generator_from_env
from course_rag_assistant.ingest import ingest_dir
from course_rag_assistant.retrieval import HybridRetriever


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def build_retriever(data_dir: Path, config: str) -> HybridRetriever:
    chunks = ingest_dir(data_dir)
    retriever = HybridRetriever(chunks)
    if config in {"hybrid", "hybrid_rerank"}:
        from course_rag_assistant.dense import NumpyVectorIndex, embedder_from_env

        backend = embedder_from_env()
        dense_path = Path("data/index/embeddings.npy")
        if backend is None or not dense_path.exists():
            raise RuntimeError("hybrid mode requires embedding environment variables and data/index/embeddings.npy")
        retriever.dense_index = NumpyVectorIndex.load(dense_path)
        retriever.embedding_backend = backend
    if config == "hybrid_rerank":
        from course_rag_assistant.rerank import reranker_from_env

        retriever.reranker = reranker_from_env()
        if retriever.reranker is None:
            raise RuntimeError("hybrid_rerank mode requires COURSE_RAG_RERANK_BASE_URL and COURSE_RAG_RERANK_API_KEY")
    return retriever


def run(input_path: Path, data_dir: Path, output_path: Path, config: str, top_k: int) -> int:
    retriever = build_retriever(data_dir, config)
    generator = generator_from_env()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for item in load_questions(input_path):
            question = item["question"]
            result = answer(question, retriever.search(question, top_k=top_k), generator)
            stream.write(json.dumps({
                "question": question,
                "answer": result.text,
                "unsupported_claims": result.unsupported_claims,
                "abstained": "没有足够证据" in result.text,
                "abstention_reason": result.abstention_reason,
                "config": config,
            }, ensure_ascii=False) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate cited answers for a JSONL question set")
    parser.add_argument("questions", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/samples"))
    parser.add_argument("--output", type=Path, default=Path("eval/answers.jsonl"))
    parser.add_argument("--config", choices=("bm25", "hybrid", "hybrid_rerank"), default="bm25")
    parser.add_argument("-k", "--top-k", type=int, default=8)
    args = parser.parse_args()
    return run(args.questions, args.data, args.output, args.config, args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())
