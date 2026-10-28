import pytest

from course_rag_assistant.answer import answer
from course_rag_assistant.answer_selection import definition_sentence_ids
from course_rag_assistant.ingest import Chunk
from course_rag_assistant.retrieval import Hit


def hit(text, page):
    return Hit(Chunk(text, "course.pdf", page), 1, "bm25")


@pytest.mark.parametrize("broken_model", [False, True])
def test_offline_definition_beats_top_ranked_example_and_keeps_source(broken_model):
    class Broken:
        def generate(self, prompt):
            raise RuntimeError("offline")

    definition = "将模型在训练数据上拟合的比在潜在分布中更接近的现象称为过拟合（overfitting）。"
    hits = [hit("复杂模型对数据造成了过拟合。", 166),
            hit("过拟合可能发生在训练过程中。", 174),
            hit("过拟合会受到训练样本数量影响。", 162), hit(definition, 158)]
    result = answer("什么是过拟合？", hits, Broken() if broken_model else None)
    assert result.text == definition + " [course.pdf, p.158]"
    assert not result.unsupported_claims


def test_comparison_selects_both_definitions_in_question_order_with_conditions():
    training = "训练误差（training error）是指模型在训练数据集上计算得到的误差。"
    generalization = "泛化误差（generalization error）是指模型在同分布新样本上的期望误差。"
    hits = [hit("多。当数据量增加，正面比例偏离1\n2的可能性降低，训练误差和泛化误差逐渐接近。", 3),
            hit(generalization, 2), hit(training, 1)]
    result = answer("训练误差和泛化误差有什么区别？", hits)
    assert result.text == training + " [course.pdf, p.1]\n" + generalization + " [course.pdf, p.2]"
    assert not result.unsupported_claims


def test_definition_matching_keeps_qualified_concepts_and_negation_separate():
    assert definition_sentence_ids("什么是独立？", ["条件独立是指给定条件后两事件独立。"] ) == []
    assert definition_sentence_ids("什么是过拟合？", ["这种现象不能称为过拟合。"] ) == []
    assert definition_sentence_ids("甲概念和乙概念有什么区别？", ["甲概念是指第一类现象。"] ) == []
    assert definition_sentence_ids("如何避免过拟合？", ["过拟合是指训练误差远小于验证误差。"] ) == []


def test_definition_preference_does_not_bypass_personal_record_evidence():
    result = answer("我的模型准确率是什么？", [hit("我的模型准确率是指测试样本上的正确比例。", 1)])
    assert result.abstention_reason == "missing_context_record"
