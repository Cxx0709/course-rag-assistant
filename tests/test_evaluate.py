"""Tests for retrieval evaluation metrics."""

from course_rag_assistant.evaluate import evaluate_ablation, evaluate_retriever, reciprocal_rank
from course_rag_assistant.ingest import Chunk
from course_rag_assistant.retrieval import HybridRetriever, LexicalReranker


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b"], {"b"}) == 0.5
    assert reciprocal_rank(["a"], {"b"}) == 0.0


def test_evaluate_retriever_matches_markdown_section():
    chunks = [
        Chunk("朴素贝叶斯使用类条件独立假设。", "notes.md", 0, "Bayes"),
        Chunk("决策树使用信息增益。", "notes.md", 0, "Trees"),
    ]
    questions = [{"question": "什么是类条件独立假设？", "gold_pages": ["notes.md#Bayes"]}]

    metrics = evaluate_retriever(HybridRetriever(chunks), questions, k=1)

    assert metrics["questions"] == 1
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] == 1.0


def test_ablation_uses_same_questions_for_each_retriever():
    chunks = [Chunk("贝叶斯定理", "notes.md", 0, "Bayes")]
    questions = [{"question": "贝叶斯定理是什么？", "gold_pages": ["notes.md#Bayes"]}]
    results = evaluate_ablation(
        {"bm25": HybridRetriever(chunks), "rerank": HybridRetriever(chunks, reranker=LexicalReranker())},
        questions,
        k=1,
    )

    assert set(results) == {"bm25", "rerank"}
    assert results["bm25"]["mrr"] == results["rerank"]["mrr"] == 1.0
