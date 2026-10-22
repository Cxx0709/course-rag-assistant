"""Optional dense retrieval backed by an OpenAI-compatible embeddings API.

The API client is deliberately small so a local BGE-M3 server and hosted
embedding providers can share the same interface.  Only vectors and chunk
metadata are persisted locally; no database service is required.
"""
from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .ingest import Chunk


class EmbeddingBackend(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return one float vector per input text."""


class OpenAICompatibleEmbedder:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        import httpx

        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": list(texts)},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        vectors = [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]
        if len(vectors) != len(texts):
            raise ValueError(f"Embedding API returned {len(vectors)} vectors for {len(texts)} texts")
        return _normalize(np.asarray(vectors, dtype=np.float32))


class OllamaEmbedder:
    """Embedding client for a model served by a local Ollama instance."""

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", timeout: float = 120.0, batch_size: int = 32):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = max(batch_size, 1)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        import httpx

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": batch},
                timeout=self.timeout,
            )
            response.raise_for_status()
            embeddings = response.json().get("embeddings", [])
            if len(embeddings) != len(batch):
                raise ValueError(f"Ollama returned {len(embeddings)} vectors for {len(batch)} texts")
            vectors.extend(embeddings)
        return _normalize(np.asarray(vectors, dtype=np.float32))


def embedder_from_env() -> EmbeddingBackend | None:
    """Create an API embedder from ``COURSE_RAG_*`` variables if configured."""
    ollama_model = os.getenv("COURSE_RAG_OLLAMA_MODEL")
    if ollama_model:
        batch_size = int(os.getenv("COURSE_RAG_OLLAMA_BATCH_SIZE", "128"))
        return OllamaEmbedder(
            ollama_model,
            os.getenv("COURSE_RAG_OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            batch_size=batch_size,
        )
    base_url = os.getenv("COURSE_RAG_EMBEDDING_BASE_URL")
    api_key = os.getenv("COURSE_RAG_EMBEDDING_API_KEY")
    model = os.getenv("COURSE_RAG_EMBEDDING_MODEL")
    if not any((base_url, api_key, model)):
        return None
    if not all((base_url, api_key, model)):
        raise ValueError(
            "COURSE_RAG_EMBEDDING_BASE_URL, COURSE_RAG_EMBEDDING_API_KEY, "
            "and COURSE_RAG_EMBEDDING_MODEL must be set together"
        )
    return OpenAICompatibleEmbedder(base_url, api_key, model)


@dataclass
class NumpyVectorIndex:
    chunks: list[Chunk]
    vectors: np.ndarray

    @classmethod
    def build(cls, chunks: list[Chunk], backend: EmbeddingBackend) -> NumpyVectorIndex:
        vectors = _normalize(backend.embed([chunk.text for chunk in chunks]))
        if len(vectors) != len(chunks):
            raise ValueError("Embedding backend returned a different number of vectors")
        return cls(chunks, vectors)

    def save(self, path: Path) -> None:
        """Save vectors to ``path`` and provenance metadata beside it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.vectors)
        metadata = [
            {"text": c.text, "doc": c.doc, "page": c.page, "section": c.section, "concepts": c.concepts}
            for c in self.chunks
        ]
        path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> NumpyVectorIndex:
        vectors = np.load(path, allow_pickle=False)
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in metadata]
        if len(chunks) != len(vectors):
            raise ValueError("Vector and metadata lengths do not match")
        return cls(chunks, _normalize(vectors))

    def search(self, query: str, backend: EmbeddingBackend, top_k: int = 8) -> list[tuple[Chunk, float]]:
        if top_k <= 0 or not self.chunks:
            return []
        query_vector = _normalize(backend.embed([query]))[0]
        scores = self.vectors @ query_vector
        indices = np.argsort(-scores)[:top_k]
        return [(self.chunks[int(i)], float(scores[int(i)])) for i in indices]

    def search_many(
        self, queries: Sequence[str], backend: EmbeddingBackend, top_k: int = 8
    ) -> list[list[tuple[Chunk, float]]]:
        """Search several queries with one embedding backend call."""
        if not queries:
            return []
        if top_k <= 0 or not self.chunks:
            return [[] for _ in queries]
        query_vectors = _normalize(backend.embed(list(queries)))
        if len(query_vectors) != len(queries):
            raise ValueError("Embedding backend returned a different number of query vectors")
        output: list[list[tuple[Chunk, float]]] = []
        for query_vector in query_vectors:
            scores = self.vectors @ query_vector
            indices = np.argsort(-scores)[: max(top_k, 0)]
            output.append([(self.chunks[int(i)], float(scores[int(i)])) for i in indices])
        return output


def build_numpy_index(
    chunks: list[Chunk], backend: EmbeddingBackend, path: Path = Path("data/index/embeddings.npy")
) -> NumpyVectorIndex:
    """Build and persist a dense index for later hybrid searches."""
    index = NumpyVectorIndex.build(chunks, backend)
    index.save(path)
    return index


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)
