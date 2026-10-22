"""Unified command-line interface for the course RAG assistant."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .answer import answer, format_citation, generator_from_env
from .evaluate import run_ablation, run_abstention, run_eval
from .ingest import ingest_dir
from .retrieval import HybridRetriever, build_bm25_index


def _retriever(root: Path, index_path: Path | None = None) -> HybridRetriever:
    retriever = build_bm25_index(ingest_dir(root), index_path) if index_path else HybridRetriever(ingest_dir(root))
    from .dense import NumpyVectorIndex, embedder_from_env
    from .rerank import reranker_from_env

    dense_path = (index_path.parent if index_path else Path("data/index")) / "embeddings.npy"
    backend = embedder_from_env()
    if dense_path.exists() and backend is not None:
        retriever.dense_index = NumpyVectorIndex.load(dense_path)
        retriever.embedding_backend = backend
    retriever.reranker = reranker_from_env()
    return retriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="course-rag", description="Course-focused retrieval and cited answering")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Parse course documents into chunks")
    ingest.add_argument("root", type=Path)
    ingest.add_argument("--json", action="store_true", help="Print machine-readable chunks")

    search = sub.add_parser("search", help="Search course materials")
    search.add_argument("query")
    search.add_argument("--data", type=Path, default=Path("data/samples"))
    search.add_argument("--index", type=Path, default=Path("data/index/bm25.json"))
    search.add_argument("-k", "--top-k", type=int, default=8)

    ask = sub.add_parser("answer", help="Answer with provenance citations")
    ask.add_argument("question")
    ask.add_argument("--data", type=Path, default=Path("data/samples"))
    ask.add_argument("-k", "--top-k", type=int, default=8)

    evaluate = sub.add_parser("evaluate", help="Run retrieval or abstention evaluation")
    evaluate.add_argument("labels", type=Path)
    evaluate.add_argument("--data", type=Path, default=Path("data/samples"))
    evaluate.add_argument("-k", type=int, default=8)
    evaluate.add_argument("--ablation", action="store_true")
    evaluate.add_argument("--unanswerable", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ingest":
        chunks = ingest_dir(args.root)
        if args.json:
            print(json.dumps([chunk.__dict__ for chunk in chunks], ensure_ascii=False, indent=2))
        else:
            print(f"ingested {len(chunks)} chunks")
            for chunk in chunks:
                print(f"- {chunk.doc} section={chunk.section or '-'}")
        return 0
    if args.command == "search":
        retriever = _retriever(args.data, args.index)
        for rank, hit in enumerate(retriever.search(args.query, args.top_k), 1):
            print(f"[{rank}] {hit.score:.4f} {format_citation(hit)}")
            print(hit.chunk.text)
        return 0
    if args.command == "answer":
        retriever = _retriever(args.data)
        result = answer(args.question, retriever.search(args.question, args.top_k), generator_from_env())
        print(result.text)
        return 0
    if args.unanswerable:
        result = run_abstention(args.labels, args.data, args.k)
    elif args.ablation:
        result = run_ablation(args.labels, args.data, args.k)
    else:
        result = run_eval(args.labels, args.data, args.k)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
