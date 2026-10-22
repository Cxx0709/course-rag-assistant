"""Offline tests for the optional dense index."""

import numpy as np

from course_rag_assistant.dense import NumpyVectorIndex
from course_rag_assistant.ingest import Chunk


class FakeEmbedder:
    def embed(self, texts):
        return np.asarray([[1.0, 0.0] if "贝叶斯" in text else [0.0, 1.0] for text in texts])


def test_numpy_index_round_trip_and_search(tmp_path):
    chunks = [Chunk("贝叶斯定理", "notes.md", 0), Chunk("决策树", "notes.md", 0)]
    backend = FakeEmbedder()
    index = NumpyVectorIndex.build(chunks, backend)
    path = tmp_path / "vectors.npy"
    index.save(path)

    restored = NumpyVectorIndex.load(path)
    hits = restored.search("贝叶斯问题", backend, top_k=1)
    assert hits[0][0] == chunks[0]
    assert hits[0][1] > 0.99


def test_batched_hybrid_matches_single_queries_with_one_backend_call():
    from course_rag_assistant.retrieval import HybridRetriever

    class CountingEmbedder(FakeEmbedder):
        def __init__(self):
            self.calls = []

        def embed(self, texts):
            self.calls.append(list(texts))
            return super().embed(texts)

    chunks = [Chunk("贝叶斯定理", "notes.md", 0), Chunk("决策树", "notes.md", 0)]
    backend = CountingEmbedder()
    index = NumpyVectorIndex.build(chunks, backend)
    retriever = HybridRetriever(chunks, index, backend)
    queries = ["贝叶斯问题", "决策树", "贝叶斯问题"]
    expected = [retriever.search(q, 2) for q in queries]
    backend.calls.clear()
    assert retriever.search_many(queries, 2) == expected
    assert backend.calls == [queries]
    backend.calls.clear()
    assert retriever.search_many(queries, 0) == [[], [], []]
    assert index.search_many(queries, backend, 0) == [[], [], []]
    assert backend.calls == []


def test_batch_rejects_missing_query_vectors():
    import pytest

    class BrokenEmbedder:
        def embed(self, texts):
            return np.array([[1.0, 0.0]])

    index = NumpyVectorIndex([Chunk("text", "notes.md", 0)], np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError, match="query vectors"):
        index.search_many(["one", "two"], BrokenEmbedder())
