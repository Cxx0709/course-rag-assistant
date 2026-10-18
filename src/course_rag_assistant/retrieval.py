"""Hybrid retrieval: BM25 + dense (BGE-M3), fused, then reranked (BGE reranker).

Design notes:
- BM25 catches exact terminology ("条件独立", "基尼指数"); dense catches
  paraphrase ("为什么朴素贝叶斯假设特征独立"). Both are needed for course Q&A.
- Rerank the top ~50 fused candidates down to the top 8 for generation.
- Each hit keeps its chunk provenance (doc, page) for citation rendering.
"""
from __future__ import annotations

from dataclasses import dataclass

from .ingest import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float
    stage: str  # "bm25" | "dense" | "fused" | "reranked"


class HybridRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        # TODO (week 2): build BM25 index + BGE-M3 embedding index (pgvector/Qdrant)

    def search(self, query: str, top_k: int = 8) -> list[Hit]:
        """BM25 + dense retrieval, RRF fusion, rerank top candidates."""
        # TODO (week 2-3): implement; return reranked hits with provenance
        raise NotImplementedError
