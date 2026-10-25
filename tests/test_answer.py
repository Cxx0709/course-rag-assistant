"""Tests for extractive answers and strict citation verification."""

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


def test_custom_generator_is_still_verified():
    chunk = Chunk("贝叶斯定理用于计算后验概率。", "notes.pdf", 12, "Bayes")

    class Generator:
        def generate(self, prompt: str) -> str:
            assert "notes.pdf" in prompt
            return "贝叶斯定理用于计算后验概率。 [notes.pdf, p.12]"

    result = answer("贝叶斯定理做什么？", [Hit(chunk, 1.0, "bm25")], Generator())

    assert "后验概率" in result.text
    assert result.unsupported_claims == []


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
