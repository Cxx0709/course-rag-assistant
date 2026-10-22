"""Hybrid retrieval: local BM25 baseline, with dense/rerank extension points.

Design notes:
- BM25 catches exact terminology ("条件独立", "基尼指数"); dense catches
  paraphrase ("为什么朴素贝叶斯假设特征独立"). Both are needed for course Q&A.
- Later stages will rerank the top ~50 fused candidates down to the top 8.
- Each hit keeps its chunk provenance (doc, page) for citation rendering.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .ingest import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float
    stage: str  # "bm25" | "dense" | "fused" | "reranked"


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        """Return hits ordered by query-passage relevance."""


class LexicalReranker:
    """Deterministic fallback reranker used before a BGE model is configured."""

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        query_terms = set(_tokenize(query))
        scored = []
        for hit in hits:
            passage_terms = set(_tokenize(f"{hit.chunk.section} {hit.chunk.text}"))
            overlap = len(query_terms & passage_terms) / max(len(query_terms), 1)
            scored.append(Hit(hit.chunk, overlap + hit.score * 1e-3, "reranked"))
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[: max(top_k, 0)]


def reciprocal_rank_fusion(
    *rankings: list[Hit], rrf_k: int = 60, top_k: int | None = None
) -> list[Hit]:
    """Fuse ranked hit lists while keeping the highest provenance score.

    A chunk can appear in several retrievers.  Its fused score is the sum of
    ``1 / (rrf_k + rank)`` for each list, with ranks starting at one.
    """
    merged: dict[tuple[str, int, str, str], Hit] = {}
    scores: dict[tuple[str, int, str, str], float] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = (hit.chunk.doc, hit.chunk.page, hit.chunk.section, hit.chunk.text)
            merged.setdefault(key, hit)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
    fused = [Hit(merged[key].chunk, score, "fused") for key, score in scores.items()]
    fused.sort(key=lambda hit: hit.score, reverse=True)
    return fused if top_k is None else fused[: max(top_k, 0)]


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], dense_index=None, embedding_backend=None, reranker=None):
        self.chunks = chunks
        # Index section titles too: Markdown headings often carry the exact
        # concept a question names while the body starts with a formula.
        self._tokens = [_tokenize(f"{chunk.section} {chunk.text}") for chunk in chunks]
        self._section_tokens = [_tokenize(chunk.section) for chunk in chunks]
        self._bm25 = _BM25(self._tokens)
        self.dense_index = dense_index
        self.embedding_backend = embedding_backend
        self.reranker = reranker

    def search(self, query: str, top_k: int = 8) -> list[Hit]:
        """Return the best matching chunks using the local BM25 baseline.

        The dense and reranking stages will plug into this method later. Keeping
        the result type and score direction stable lets evaluation start before
        a model download or remote embedding API is available.
        """
        if top_k <= 0 or not self.chunks:
            return []
        candidate_k = max(top_k, 50) if (self.dense_index and self.embedding_backend) or self.reranker else top_k
        scores = self._bm25.get_scores(_tokenize(query))
        query_tokens = set(_tokenize(query))
        # A phrase appearing in a section heading is a strong topical signal.
        # Give it a modest boost while retaining BM25's body-text evidence.
        scores = [
            score + 1.5 * len(query_tokens.intersection(section_tokens))
            for score, section_tokens in zip(scores, self._section_tokens)
        ]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        bm25_hits = [Hit(self.chunks[i], scores[i], "bm25") for i in ranked[:candidate_k]]
        if not self.dense_index or not self.embedding_backend:
            result = bm25_hits[:top_k]
        else:
            dense_hits = [Hit(chunk, score, "dense") for chunk, score in self.dense_index.search(query, self.embedding_backend, candidate_k)]
            result = reciprocal_rank_fusion(bm25_hits, dense_hits, top_k=candidate_k)
        if self.reranker:
            return self.reranker.rerank(query, result, top_k)
        return result[:top_k]

    def search_many(self, queries: list[str], top_k: int = 8) -> list[list[Hit]]:
        """Search multiple queries, batching dense embedding requests when available."""
        if not queries:
            return []
        if top_k <= 0 or not self.chunks:
            return [[] for _ in queries]
        if not self.dense_index or not self.embedding_backend or self.reranker:
            return [self.search(query, top_k) for query in queries]
        candidate_k = max(top_k, 50)
        dense_rankings = self.dense_index.search_many(queries, self.embedding_backend, candidate_k)
        results: list[list[Hit]] = []
        for query, dense_hits in zip(queries, dense_rankings):
            scores = self._bm25.get_scores(_tokenize(query))
            query_tokens = set(_tokenize(query))
            scores = [
                score + 1.5 * len(query_tokens.intersection(section_tokens))
                for score, section_tokens in zip(scores, self._section_tokens)
            ]
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            bm25_hits = [Hit(self.chunks[i], scores[i], "bm25") for i in ranked[:candidate_k]]
            results.append(reciprocal_rank_fusion(bm25_hits, [Hit(chunk, score, "dense") for chunk, score in dense_hits], top_k=top_k))
        return results

    def save(self, path: Path) -> None:
        """Persist chunks and tokenizer output as a portable JSON index."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "chunks": [
                {
                    "text": chunk.text,
                    "doc": chunk.doc,
                    "page": chunk.page,
                    "section": chunk.section,
                    "concepts": chunk.concepts,
                }
                for chunk in self.chunks
            ],
            "tokens": self._tokens,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> HybridRetriever:
        """Load an index created by :meth:`save`."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError(f"Unsupported retrieval index version: {payload.get('version')!r}")
        chunks = [Chunk(**item) for item in payload["chunks"]]
        retriever = cls(chunks)
        tokens = payload.get("tokens")
        if tokens is not None and len(tokens) == len(chunks):
            retriever._tokens = tokens
            retriever._bm25 = _BM25(tokens)
        return retriever


def build_bm25_index(chunks: list[Chunk], path: Path = Path("data/index/bm25.json")) -> HybridRetriever:
    """Build and persist the local baseline index, returning its retriever."""
    retriever = HybridRetriever(chunks)
    retriever.save(path)
    return retriever


class _BM25:
    """Small dependency-free BM25 implementation for the local MVP index."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, corpus)) / max(len(corpus), 1)
        self.doc_freq: dict[str, int] = {}
        for tokens in corpus:
            for token in set(tokens):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

    def get_scores(self, query: list[str]) -> list[float]:
        scores: list[float] = []
        n_docs = len(self.corpus)
        for document in self.corpus:
            counts = {token: document.count(token) for token in set(document)}
            score = 0.0
            for token in query:
                if token not in counts:
                    continue
                df = self.doc_freq.get(token, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                tf = counts[token]
                norm = tf + self.k1 * (1 - self.b + self.b * len(document) / max(self.avgdl, 1e-9))
                score += idf * tf * (self.k1 + 1) / norm
            scores.append(score)
        return scores


def _tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text without a heavyweight segmenter."""
    tokens: list[str] = []
    stopwords = {"的", "了", "和", "与", "及", "是", "什么", "怎么", "哪些", "为什么", "如何", "是否", "我", "你", "有", "在", "中", "对", "从", "这", "那", "用"}
    for part in re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            if part in stopwords:
                continue
            # Character-level tokens make unrelated Chinese sentences match on
            # common characters. Bigrams retain short-concept recall without
            # turning words such as "机器学习" into noisy one-character hits.
            if len(part) == 1 and part not in stopwords:
                tokens.append(part)
            tokens.extend(
                part[i : i + 2]
                for i in range(len(part) - 1)
                if part[i : i + 2] not in stopwords
            )
        else:
            if part not in stopwords:
                tokens.append(part)
    return tokens


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build and query the local BM25 index")
    parser.add_argument("root", nargs="?", type=Path, default=Path("data/samples"))
    parser.add_argument("--output", type=Path, default=Path("data/index/bm25.json"))
    parser.add_argument("--query", default="", help="Run a query after building the index")
    args = parser.parse_args()

    from .dense import NumpyVectorIndex, embedder_from_env
    from .ingest import ingest_dir
    from .rerank import reranker_from_env

    retriever = build_bm25_index(ingest_dir(args.root), args.output)
    dense_path = args.output.parent / "embeddings.npy"
    embedding_backend = embedder_from_env()
    if dense_path.exists() and embedding_backend is not None:
        retriever.dense_index = NumpyVectorIndex.load(dense_path)
        retriever.embedding_backend = embedding_backend
        print(f"loaded dense index from {dense_path}")
    retriever.reranker = reranker_from_env()
    if retriever.reranker is not None:
        print("loaded configured reranker")
    print(f"indexed {len(retriever.chunks)} chunks at {args.output}")
    if args.query:
        for rank, hit in enumerate(retriever.search(args.query), start=1):
            print(f"[{rank}] {hit.score:.4f} {hit.chunk.doc} section={hit.chunk.section or '-'}")
            print(hit.chunk.text)
