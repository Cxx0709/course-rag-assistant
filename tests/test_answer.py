"""Tests for extractive answers and strict citation verification."""

import json

import pytest

from course_rag_assistant.answer import CitedAnswer, answer, verify_citations
from course_rag_assistant.ingest import Chunk
from course_rag_assistant.retrieval import Hit


def test_answer_cites_supporting_page():
    chunk = Chunk("贝叶斯定理用于计算后验概率。", "notes.pdf", 12, "Bayes")
    result = answer("贝叶斯定理做什么？", [Hit(chunk, 1.0, "bm25")])

    assert "[notes.pdf, p.12]" in result.text
    assert result.unsupported_claims == []


def test_verifier_removes_claim_without_overlap():
    chunk = Chunk("贝叶斯定理用于计算后验概率。", "notes.pdf", 12, "Bayes")
    result = verify_citations(
        CitedAnswer(
            "贝叶斯定理用于计算后验概率。 [notes.pdf, p.12]\n"
            "决策树需要剪枝。 [notes.pdf, p.12]",
            [Hit(chunk, 1.0, "bm25")],
            [],
        )
    )

    assert "后验概率" in result.text
    assert "决策树" not in result.text
    assert any("决策树" in claim for claim in result.unsupported_claims)


def test_markdown_answer_uses_section_anchor():
    chunk = Chunk("类条件独立假设简化了联合概率计算。", "notes.md", 0, "Bayes")
    result = answer("为什么要做这个假设？", [Hit(chunk, 1.0, "bm25")])

    assert "[notes.md, section=Bayes]" in result.text


def test_custom_generator_selects_source_sentence():
    chunk = Chunk("贝叶斯定理用于计算后验概率。", "notes.pdf", 12, "Bayes")

    class Generator:
        def generate(self, prompt: str) -> str:
            assert "notes.pdf" in prompt
            return '{"sentence_ids":[1]}'

    result = answer("贝叶斯定理做什么？", [Hit(chunk, 1.0, "bm25")], Generator())

    assert "后验概率" in result.text
    assert result.unsupported_claims == []
    assert result.selected_sentence_ids == [1]


def test_failed_generator_falls_back_to_extractive_answer():
    chunk = Chunk("贝叶斯定理用于计算后验概率。", "notes.pdf", 12, "Bayes")

    class BrokenGenerator:
        def generate(self, prompt: str) -> str:
            raise RuntimeError("local model unavailable")

    result = answer("贝叶斯定理做什么？", [Hit(chunk, 1.0, "bm25")], BrokenGenerator())

    assert "后验概率" in result.text
    assert "[notes.pdf, p.12]" in result.text


def test_shared_words_do_not_validate_a_contradiction():
    chunk = Chunk("模型需要训练数据。", "notes.pdf", 1)
    result = verify_citations(CitedAnswer(
        "模型不需要训练数据。 [notes.pdf, p.1]", [Hit(chunk, 1, "bm25")], []))
    assert "没有足够证据" in result.text
    assert result.unsupported_claims == ["模型不需要训练数据。"]


def test_one_shared_character_does_not_support_unrelated_claim():
    chunk = Chunk("贝叶斯定理用于计算后验概率。", "notes.pdf", 1)
    result = verify_citations(CitedAnswer(
        "恐龙用于驾驶飞船。 [notes.pdf, p.1]", [Hit(chunk, 1, "bm25")], []))
    assert "没有足够证据" in result.text


def test_uncited_following_sentence_cannot_borrow_previous_citation():
    chunk = Chunk("模型需要训练数据。", "notes.pdf", 1)
    result = verify_citations(CitedAnswer(
        "模型需要训练数据。 [notes.pdf, p.1] 模型不需要训练数据。",
        [Hit(chunk, 1, "bm25")], []))
    assert result.text == "模型需要训练数据。 [notes.pdf, p.1]"
    assert result.unsupported_claims == ["模型不需要训练数据。"]


def test_correct_citation_does_not_hide_a_fabricated_citation():
    chunk = Chunk("模型需要训练数据。", "notes.pdf", 1)
    result = verify_citations(CitedAnswer(
        "模型需要训练数据。 [notes.pdf, p.1] [missing.pdf, p.9]",
        [Hit(chunk, 1, "bm25")], []))
    assert "没有足够证据" in result.text


def test_headingless_markdown_citation_resolves():
    result = answer("训练数据", [Hit(Chunk("模型需要训练数据。", "notes.md", 0), 1, "bm25")])
    assert "[notes.md, section=document]" in result.text
    assert not result.unsupported_claims


def test_positive_fusion_score_is_not_evidence_for_unrelated_query():
    result = answer("护照办理", [Hit(Chunk("模型需要训练数据。", "notes.pdf", 1), 0.03, "fused")])
    assert "没有足够证据" in result.text


def test_free_paraphrase_requires_semantic_verification():
    chunk = Chunk("模型需要训练数据。", "notes.pdf", 1)
    result = verify_citations(CitedAnswer(
        "训练模型需要数据。 [notes.pdf, p.1]", [Hit(chunk, 1, "bm25")], []))
    assert "没有足够证据" in result.text


class SelectionGenerator:
    def __init__(self, reply):
        self.reply = reply

    def generate(self, prompt):
        return self.reply


@pytest.mark.parametrize("reply", [
    '{"sentence_ids":[0]}', '{"sentence_ids":[-1]}', '{"sentence_ids":[2]}',
    '{"sentence_ids":[true]}', '{"sentence_ids":[1.0]}', '{"sentence_ids":["1"]}',
    '{"sentence_ids":null}', '{"sentence_ids":1}', '[]', 'null',
    '{"sentence_ids":[1],"text":"伪造内容"}', '{"sentence_ids":[1,999]}',
    '{"sentence_ids":[999],"sentence_ids":[1]}', '{"sentence_ids":[1]} 后续解释',
    '{"sentence_ids":[', '模型不需要训练数据。 [notes.pdf, p.1]',
])
def test_invalid_selections_refuse_without_extractive_fallback(reply):
    result = answer("训练数据", [Hit(Chunk("模型需要训练数据。", "notes.pdf", 1), 1, "bm25")],
                    SelectionGenerator(reply))
    assert result.abstention_reason == "invalid_sentence_selection"
    assert "[notes.pdf" not in result.text
    assert result.selected_sentence_ids == []


def test_empty_selection_is_an_explicit_model_abstention():
    result = answer("训练数据", [Hit(Chunk("模型需要训练数据。", "notes.pdf", 1), 1, "bm25")],
                    SelectionGenerator('{"sentence_ids":[]}'))
    assert result.abstention_reason == "model_abstained"
    assert "[notes.pdf" not in result.text


def test_selection_renders_only_chosen_sentences_with_their_own_source():
    hits = [Hit(Chunk("训练数据用于拟合参数。训练数据可能含有噪声。", "a.pdf", 3), 1, "bm25"),
            Hit(Chunk("训练数据应当清洗。", "b.md", 0, "清洗"), 1, "bm25")]

    class Generator:
        def generate(self, prompt):
            payload = json.loads(prompt.split("\n")[-1])
            assert [c["id"] for c in payload["candidates"]] == [1, 2, 3]
            assert payload["candidates"][2]["source"] == "[b.md, section=清洗]"
            return '```json\n{"sentence_ids":[3,1,3]}\n```'

    result = answer("训练数据", hits, Generator())
    assert result.text == "训练数据应当清洗。 [b.md, section=清洗]\n训练数据用于拟合参数。 [a.pdf, p.3]"
    assert result.selected_sentence_ids == [3, 1]
    assert result.unsupported_claims == []


def test_selection_still_passes_through_citation_verifier():
    # A citation-looking string inside source data cannot become trusted metadata.
    hit = Hit(Chunk("模型需要训练数据。 [forged.pdf, p.99]", "notes.pdf", 1), 1, "bm25")
    result = answer("训练数据", [hit], SelectionGenerator('{"sentence_ids":[1]}'))
    assert result.abstention_reason == "unsupported_citations"
    assert "[forged.pdf" not in result.text
