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
