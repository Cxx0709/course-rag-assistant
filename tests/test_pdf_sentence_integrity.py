import json

import pytest

from course_rag_assistant.answer import (
    CitedAnswer,
    _sentence_options,
    answer,
    verify_citations,
)
from course_rag_assistant.ingest import Chunk
from course_rag_assistant.retrieval import Hit


def hit(text, doc="book.pdf", page=1):
    return Hit(Chunk(text, doc, page), 1, "chat_reranked")


@pytest.mark.parametrize("fragment", [
    "深度学习消除了计算机视觉、语音识别、",
    "深度学习的流程如下：", "深度学习的区别主要在于",
    "深度学习（deep learning。", "learning）用于深度学习。",
    "l(i)(w, b) = 1", "2", "\x10",
])
def test_pdf_fragments_are_not_candidates_or_verified_claims(fragment):
    candidate = hit(fragment)
    assert not _sentence_options([candidate])
    result = verify_citations(CitedAnswer(fragment + " [book.pdf, p.1]", [candidate], []))
    assert result.abstention_reason == "unsupported_citations"


def test_only_partial_evidence_refuses_before_model_call():
    class NeverCalled:
        def generate(self, prompt):
            pytest.fail("Incomplete source must not reach the model")

    for generator in [None, NeverCalled()]:
        result = answer("深度学习", [hit("深度学习的流程如下：")], generator)
        assert result.abstention_reason == "incomplete_source_sentences"


def test_fragment_only_passages_do_not_crowd_out_complete_evidence():
    candidates = [hit("深度学习的流程如下：", page=p) for p in [1, 2, 3]]
    candidates.append(hit("深度学习可以学习多层表示。", page=4))
    result = answer("深度学习", candidates)
    assert result.text == "深度学习可以学习多层表示。 [book.pdf, p.4]"


def test_mixed_pdf_keeps_complete_definition_in_both_generation_paths():
    candidate = hit("平方误差是预测值与实际值之差的平方。\n平方误差的公式如下：\nl(i)(w, b) = 1\n2\n\x10")

    class SelectDefinition:
        def generate(self, prompt):
            options = json.loads(prompt.split("\n")[-1])["candidates"]
            assert len(options) == 1 and options[0]["text"] == "平方误差是预测值与实际值之差的平方。"
            return '{"sentence_ids":[1]}'

    class BrokenModel:
        def generate(self, prompt):
            raise RuntimeError("unavailable")

    for generator in [None, SelectDefinition(), BrokenModel()]:
        result = answer("平方误差", [candidate], generator)
        assert result.text == "平方误差是预测值与实际值之差的平方。 [book.pdf, p.1]"


@pytest.mark.parametrize("text", [
    "1. 用来学习的数据；", "• 通过反向传播计算梯度；",
    "模型在区间(0, 1]内输出。", "只有输入为正时，模型才输出正值。",
    "模型不使用测试集拟合参数。", "trainer = torch.optim.SGD(net.parameters(), lr=0.03)",
    "y = w * x + b", "print(')')", "print('中文')", "A model learns from data.",
])
def test_complete_items_conditions_intervals_and_code_remain_selectable(text):
    assert _sentence_options([hit(text)])[0][0] == text


def test_separate_pdf_pages_are_never_joined_to_invent_a_sentence():
    candidates = [hit("模型只在输入满足", page=1), hit("时更新参数。", page=2)]
    assert "模型只在输入满足时更新参数。" not in answer("模型", candidates).text


@pytest.mark.parametrize("doc", ["notes.md", "slides.pptx"])
def test_unpunctuated_notes_are_not_subject_to_pdf_prose_rules(doc):
    assert _sentence_options([hit("学习率为0.03", doc)])[0][0] == "学习率为0.03"


@pytest.mark.parametrize("prefix", ["tensor([[6., 6.],\n[22., 22.]])", "print('中文')"])
def test_code_output_does_not_swallow_following_explanation(prefix):
    statement = "矩阵乘法不应与Hadamard积混淆。"
    candidate = hit(prefix + "\n" + statement)
    options = _sentence_options([candidate])
    assert statement in [text for text, _ in options]
    result = answer("Hadamard积和矩阵乘法是否相同？", [candidate])
    assert statement + " [book.pdf, p.1]" in result.text
    assert not result.unsupported_claims
