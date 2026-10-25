from course_rag_assistant.ingest import Chunk
from course_rag_assistant.retrieval import HybridRetriever
from scripts.evaluate_quality import answer_metrics


def test_citation_presence_does_not_count_as_correct_reference():
    retriever = HybridRetriever([Chunk("模型需要训练数据。", "notes.pdf", 2)])
    labels = [{"id": "q1", "question": "训练数据", "gold_pages": ["notes.pdf#p.1"],
               "answer_points": ["训练数据"]}]
    metrics = answer_metrics(retriever, labels, 8)
    assert metrics["details"][0]["citation_count"] == 1
    assert metrics["gold_page_citation_precision"] == 0
    assert metrics["literal_answer_point_coverage"] == 1


def test_refusal_counts_as_missing_answer_points():
    labels = [{"id": "q1", "question": "训练数据", "gold_pages": ["notes.pdf#p.1"],
               "answer_points": ["训练数据"]}]
    metrics = answer_metrics(HybridRetriever([]), labels, 8)
    assert metrics["gold_page_citation_precision"] is None
    assert metrics["literal_answer_point_coverage"] == 0
    assert metrics["abstention_rate"] == 1
