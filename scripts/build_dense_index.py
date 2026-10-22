"""Build a persisted dense index using Ollama or an OpenAI-compatible API."""
from __future__ import annotations

import argparse
from pathlib import Path

from course_rag_assistant.dense import build_numpy_index, embedder_from_env
from course_rag_assistant.ingest import ingest_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the course dense vector index")
    parser.add_argument("--data", type=Path, default=Path("data/raw/datamining-notes"))
    parser.add_argument("--output", type=Path, default=Path("data/index/embeddings.npy"))
    args = parser.parse_args()
    backend = embedder_from_env()
    if backend is None:
        raise SystemExit("Set COURSE_RAG_OLLAMA_MODEL or the COURSE_RAG_EMBEDDING_* variables first")
    chunks = ingest_dir(args.data)
    index = build_numpy_index(chunks, backend, args.output)
    print(f"built dense index: {len(index.chunks)} chunks at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
