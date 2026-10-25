import pytest

from course_rag_assistant.ingest import Chunk
from course_rag_assistant.retrieval import HybridRetriever
from scripts.evaluate_quality import RankedResults, answer_metrics, main


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


def _quality_case():
    return (HybridRetriever([Chunk("模型需要训练数据。", "notes.pdf", 2)]),
            [{"id": "q1", "question": "训练数据", "gold_pages": ["notes.pdf#p.2"],
              "answer_points": ["训练数据"]}])


def test_chat_failure_is_reported_even_when_fallback_answers_correctly():
    class BrokenGenerator:
        def generate(self, prompt):
            raise RuntimeError("secret credential must not appear in report")

    retriever, labels = _quality_case()
    result = answer_metrics(retriever, labels, 8, BrokenGenerator())
    assert result["literal_answer_point_coverage"] == 1
    assert result["model_calls"] == result["generation_failures"] == 1
    row = result["details"][0]
    assert row["generation_status"] == "fallback_extractive"
    assert row["generation_error_type"] == "RuntimeError"
    assert "secret credential" not in str(result)


def test_chat_output_is_preserved_before_rejection():
    class ContradictingGenerator:
        def generate(self, prompt):
            return "模型不需要训练数据。 [notes.pdf, p.2]"

    retriever, labels = _quality_case()
    result = answer_metrics(retriever, labels, 8, ContradictingGenerator())
    assert result["model_calls"] == result["model_outputs_fully_rejected"] == 1
    assert result["generation_failures"] == 0
    assert result["abstention_rate"] == 1
    row = result["details"][0]
    assert "不需要" in row["raw_model_output"]
    assert row["abstention_reason"] == "invalid_sentence_selection"
    assert row["selected_sentence_ids"] == []
    assert row["evidence"][0]["text"] == "模型需要训练数据。"


def test_evidence_refusal_does_not_count_as_model_call():
    class UnusedGenerator:
        def generate(self, prompt):
            pytest.fail("No model call expected without evidence")

    _, labels = _quality_case()
    result = answer_metrics(HybridRetriever([]), labels, 8, UnusedGenerator())
    assert result["model_calls"] == result["generation_failures"] == 0
    assert result["details"][0]["generation_status"] == "skipped_no_evidence"


def test_successful_chat_is_measured_after_verification():
    class Generator:
        def generate(self, prompt):
            return '{"sentence_ids":[1]}'

    retriever, labels = _quality_case()
    result = answer_metrics(retriever, labels, 8, Generator())
    assert result["model_calls"] == 1
    assert result["generation_failures"] == result["model_outputs_fully_rejected"] == 0
    assert result["gold_page_citation_precision"] == result["literal_answer_point_coverage"] == 1
    assert result["details"][0]["selected_sentence_ids"] == [1]


def test_model_abstention_is_not_counted_as_invalid_output():
    class Generator:
        def generate(self, prompt):
            return '{"sentence_ids":[]}'

    retriever, labels = _quality_case()
    result = answer_metrics(retriever, labels, 8, Generator())
    assert result["model_abstentions"] == 1
    assert result["model_outputs_fully_rejected"] == result["generation_failures"] == 0


def test_short_rankings_cannot_silently_omit_questions():
    class BrokenRetriever:
        def search_many(self, queries, top_k):
            return []

    _, labels = _quality_case()
    with pytest.raises(ValueError, match="different number"):
        answer_metrics(BrokenRetriever(), labels, 8)
    with pytest.raises(ValueError, match="different number"):
        RankedResults(["q"], [])


def test_chat_configuration_is_required(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["evaluate_quality.py", "--chat"])
    monkeypatch.setattr("scripts.evaluate_quality.generator_from_env", lambda: None)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "requires a configured chat generator" in capsys.readouterr().err


def test_requested_reranking_is_not_silently_skipped(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["evaluate_quality.py", "--dense", "--modes", "hybrid+rerank"])
    monkeypatch.setattr("scripts.evaluate_quality.reranker_from_env", lambda: None)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "requires a configured reranker" in capsys.readouterr().err
