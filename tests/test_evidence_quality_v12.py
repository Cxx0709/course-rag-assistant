import pytest

from course_rag_assistant.answer import (
    CitedAnswer,
    _sentence_options,
    answer,
    render_selection,
    verify_citations,
)
from course_rag_assistant.ingest import Chunk
from course_rag_assistant.retrieval import Hit


def hit(text, page=1, stage="bm25"):
    return Hit(Chunk(text, "book.pdf", page), 1, stage)


def test_resolved_but_unrelated_extra_citation_is_rejected():
    hits = [hit("模型需要训练数据。"), hit("模型需要验证数据。", 2)]
    result = verify_citations(CitedAnswer(
        "模型需要训练数据。 [book.pdf, p.1] [book.pdf, p.2]", hits, []))
    assert result.abstention_reason == "unsupported_citations"


@pytest.mark.parametrize("stage", ["bm25", "fused", "chat_reranked"])
def test_identical_sentences_deduplicate_without_merging_negation(stage):
    hits = [hit("模型需要训练数据。", stage=stage),
            hit("模型需要训练数据。模型不需要训练数据。", 2, stage)]
    result = render_selection('{"sentence_ids":[1,2,3]}', hits)
    assert result.selected_sentence_ids == [1, 3]
    assert result.text.count("模型需要训练数据。") == 1
    assert "模型不需要训练数据。" in result.text


def test_offline_deduplicates_repeated_source_units():
    result = answer("训练数据", [hit("模型需要训练数据。模型需要训练数据。")])
    assert result.text.count("模型需要训练数据。") == 1


def test_closing_quote_stays_with_its_sentence():
    text = "教材称之为“表示学习。”模型需要训练数据。"
    units = _sentence_options([hit(text)])
    assert units[0][0] == "教材称之为“表示学习。”"
    result = render_selection('{"sentence_ids":[1,2]}', [hit(text)])
    assert not result.unsupported_claims
    assert result.text.count("[book.pdf, p.1]") == 2


def test_english_sentences_separate_without_splitting_decimal_or_api():
    text = "The rate is 0.03. Training uses torch.optim.SGD."
    assert [s for s, _ in _sentence_options([hit(text)])] == [
        "The rate is 0.03.", "Training uses torch.optim.SGD."]


def test_multi_sentence_quotation_keeps_its_context():
    text = "教材指出：“模型需要训练数据。数据也需要清洗。”"
    assert _sentence_options([hit(text)])[0][0] == text


@pytest.mark.parametrize("text", ["print('第一句。第二句。')", "Dr. Smith trains a model."])
def test_code_literals_and_abbreviations_are_not_split(text):
    assert [s for s, _ in _sentence_options([hit(text)])] == [text]
