import json
from dataclasses import replace
from pathlib import Path

import pytest

from course_rag_assistant.ingest import Chunk
from course_rag_assistant.quiz import make_questions
from course_rag_assistant.quiz_candidates import extract_candidates, prompt_rejection
from course_rag_assistant.quiz_content import cloze_prompt


def test_complete_inline_name_owns_alias_and_evidence_under_short_heading():
    sentence = "我们将只利用前一个时刻观测值进行估计的方法称为单步预测（one‐step‐ahead prediction）。"
    chunks = [Chunk(sentence, "book.pdf", 9, "预测"),
              Chunk("区间预测是指给出未来观测值可能范围的预测方法。", "book.pdf", 9, "预测")]
    candidate = extract_candidates(chunks)[0]
    assert candidate.concept == "单步预测"
    assert candidate.aliases == ("one‐step‐ahead prediction",)
    assert candidate.sentence == sentence and candidate.page == 9
    assert make_questions(chunks, target_concept="预测") == []
    prompt = cloze_prompt(candidate.concept, candidate.sentence)
    assert "one" not in prompt and prompt.count("____") == 1
    assert prompt_rejection(candidate, prompt) is None


def test_qualified_name_never_becomes_a_heading_substring():
    text = "由于ht从未被观测到，这类模型也被称为隐变量自回归模型（latent autoregressive models）。"
    chunks = [Chunk(text, "book.pdf", 309, "自回归模型"),
              Chunk("线性模型是一种用输入特征的线性组合表示输出的模型。", "book.pdf", 309, "线性模型")]
    candidate = extract_candidates(chunks)[0]
    assert candidate.concept == "隐变量自回归模型"
    assert candidate.aliases == ("latent autoregressive models",)
    assert prompt_rejection(candidate, cloze_prompt(candidate.concept, candidate.sentence)) is not None
    assert make_questions(chunks) == []  # Rejected evidence cannot supply options.


@pytest.mark.parametrize("text,heading", [
    ("处理序列数据需要统计工具和新的深度神经网络架构。", "统计工具"),
    ("为了更好地理解门控循环单元模型，我们从零开始实现它。", "从零开始实现"),
    ("首先是检查模型预测下一个时间步的能力，也就是单步预测（one‐step‐ahead prediction）。", "预测"),
])
def test_heading_mentions_and_procedures_do_not_establish_concepts(text, heading):
    assert extract_candidates([Chunk(text, "book.pdf", 9, heading)]) == []


@pytest.mark.parametrize("text,reason", [
    ("候选隐状态用于计算图9.1.1中所示的输出。", "external_figure_or_formula"),
    ("候选隐状态用于公式(8.4.5)中的更新机制。", "external_figure_or_formula"),
    ("候选隐状态用于计算更新门Zt接近0时的输出。", "notation_or_latin_context"),
    ("候选隐状态用于上述模型中的状态更新过程。", "unresolved_reference"),
    ("这类模型用于预测未来时间步的观测结果。", "unresolved_reference"),
])
def test_unsupported_context_is_rejected_without_rewriting_source(text, reason):
    heading = "这类模型" if text.startswith("这类模型") else "候选隐状态"
    candidate, = extract_candidates([Chunk(text, "book.pdf", 9, heading)])
    assert prompt_rejection(candidate, cloze_prompt(candidate.concept, candidate.sentence)) == reason
    assert candidate.sentence == text


def test_unrelated_concept_cannot_fill_a_quiz_or_a_target_quiz():
    chunks = [Chunk("过拟合是指模型在训练集上表现良好，但在新数据上表现较差。", "book.pdf", 8, "过拟合"),
              Chunk("学习率控制每次参数更新的步长。", "book.pdf", 8, "学习率")]
    assert len(extract_candidates(chunks)) == 2
    assert make_questions(chunks) == []
    assert make_questions(chunks, target_concept="过拟合") == []
    chunks.append(Chunk("欠拟合是指模型表达能力不足，训练误差与验证误差都较高。", "book.pdf", 9, "欠拟合"))
    questions = make_questions(chunks)
    assert len(questions) == 2
    assert all(set(q["options"]) == {"过拟合", "欠拟合"} for q in questions)
    assert make_questions([chunks[0], replace(chunks[-1], doc="other.pdf")]) == []


@pytest.mark.parametrize("text", [
    "我们希望有一些机制来跳过隐状态表示中的此类词元。",
    "输出的预测值由输入特征通过线性模型的仿射变换决定。",
    "我们试图找到一个能够尽可能拟合训练数据的函数。",
])
def test_predicate_alone_cannot_turn_a_clause_into_a_concept(text):
    assert extract_candidates([Chunk(text, "book.pdf", 9, "模型")]) == []


def test_explicit_same_alias_cannot_be_an_alternative_answer():
    chunks = [Chunk("开发误差（development error）是指模型在开发集上的误差。", "book.pdf", 9, "误差"),
              Chunk("验证误差（development error）是指模型在验证集上的误差。", "book.pdf", 9, "误差")]
    assert len(extract_candidates(chunks)) == 2
    assert make_questions(chunks) == []


def test_preserved_failed_holdout_sentences_cannot_return_as_questions_or_options():
    report = json.loads((Path(__file__).parents[1] / "eval/reports/quiz-holdout-v1.json").read_text(encoding="utf-8"))
    rows = [q for chapter in report["chapters"] for q in chapter["questions"]]
    assert len(rows) == 7
    chunks = [Chunk(q["sentence"], q["doc"], q["page"], q["section"], [q["concept"]]) for q in rows]
    assert make_questions(chunks) == []
