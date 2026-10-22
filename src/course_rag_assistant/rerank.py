"""Optional BGE reranker using an OpenAI-compatible HTTP endpoint."""
from __future__ import annotations

import os

import httpx

from .retrieval import Hit


class BGEReranker:
    def __init__(self, base_url: str, api_key: str, model: str = "BAAI/bge-reranker-v2-m3", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits or top_k <= 0:
            return []
        response = httpx.post(
            f"{self.base_url}/rerank",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "query": query, "documents": [h.chunk.text for h in hits]},
            timeout=self.timeout,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        scored = []
        for item in results:
            index = item.get("index")
            if isinstance(index, int) and 0 <= index < len(hits):
                scored.append(Hit(hits[index].chunk, float(item.get("relevance_score", 0.0)), "reranked"))
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]


def reranker_from_env() -> BGEReranker | None:
    base_url = os.getenv("COURSE_RAG_RERANK_BASE_URL")
    api_key = os.getenv("COURSE_RAG_RERANK_API_KEY")
    if not base_url or not api_key:
        return None
    return BGEReranker(base_url, api_key, os.getenv("COURSE_RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"))
