"""Tests for the local retrieval baseline."""

from course_rag_assistant.ingest import Chunk
from course_rag_assistant.retrieval import Hit, HybridRetriever, LexicalReranker, reciprocal_rank_fusion


def test_bm25_prioritizes_exact_course_term():
    chunks = [
        Chunk("朴素贝叶斯在类条件独立假设下工作。", "notes.md", 0, "Bayes"),
        Chunk("决策树使用信息增益选择划分特征。", "notes.md", 0, "Trees"),
    ]

    hits = HybridRetriever(chunks).search("什么是类条件独立假设", top_k=1)

    assert len(hits) == 1
    assert hits[0].chunk is chunks[0]
    assert hits[0].stage == "bm25"


def test_empty_query_and_zero_top_k_are_safe():
    chunks = [Chunk("贝叶斯定理", "notes.md", 0)]
    retriever = HybridRetriever(chunks)

    assert retriever.search("", top_k=0) == []
    assert retriever.search("", top_k=1)[0].score == 0


def test_index_round_trip(tmp_path):
    chunks = [Chunk("条件独立是朴素贝叶斯的假设。", "notes.md", 0, "Bayes", ["条件独立"])]
    path = tmp_path / "bm25.json"
    original = HybridRetriever(chunks)
    original.save(path)

    restored = HybridRetriever.load(path)
    hit = restored.search("条件独立", top_k=1)[0]
    assert hit.chunk == chunks[0]
    assert hit.score > 0


def test_rrf_promotes_results_seen_by_both_retrievers():
    first = Chunk("条件独立", "notes.md", 0)
    second = Chunk("贝叶斯定理", "notes.md", 0)
    fused = reciprocal_rank_fusion(
        [Hit(first, 1.0, "bm25"), Hit(second, 0.5, "bm25")],
        [Hit(first, 1.0, "dense"), Hit(second, 0.5, "dense")],
        top_k=2,
    )

    assert fused[0].chunk is first
    assert fused[0].score > fused[1].score
    assert all(hit.stage == "fused" for hit in fused)


def test_optional_reranker_marks_results():
    chunks = [
        Chunk("朴素贝叶斯使用类条件独立假设。", "notes.md", 0, "Bayes"),
        Chunk("决策树使用信息增益。", "notes.md", 0, "Trees"),
    ]
    hits = HybridRetriever(chunks, reranker=LexicalReranker()).search("类条件独立", top_k=1)

    assert hits[0].chunk is chunks[0]
    assert hits[0].stage == "reranked"
