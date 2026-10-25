"""Development regressions for field-specific values and conflicting records."""
import pytest

from course_rag_assistant.answer import answer
from course_rag_assistant.ingest import Chunk
from course_rag_assistant.retrieval import Hit


def records(*texts, section="实验记录"):
    return [Hit(Chunk(text, f"record-{i}.md", 0, section), 1, "bm25") for i, text in enumerate(texts)]


@pytest.mark.parametrize("text", [
    "模型精度尚未测量。", "模型精度没有记录。", "耗时87秒，模型精度尚未测量。",
    "模型精度用于衡量效果，实验进行了87轮。", "目标模型精度为95%。",
    "模型精度不是95%。", "模型精度为90%到95%。",
    "模型精度为0.95秒。",
])
def test_missing_field_cannot_borrow_other_numbers(text):
    result = answer("我的模型精度是多少？", records(text))
    assert result.abstention_reason == "missing_requested_fact"
    assert "精度" in result.text


def test_actual_value_and_equivalent_format_do_not_conflict():
    result = answer("我的模型精度是多少？", records("模型精度为95%。", "模型精度为0.95。"))
    assert result.abstention_reason is None
    assert "95%" in result.text


@pytest.mark.parametrize("texts", [
    ("模型精度为95%。", "模型精度为96%。"),
    ("模型精度为95%。", "模型精度尚未测量。"),
    ("模型精度为95%；模型精度为96%。",),
])
def test_conflicts_are_not_resolved_by_retrieval_rank(texts):
    result = answer("我的模型精度是多少？", records(*texts))
    assert result.abstention_reason == "conflicting_records"
    assert "冲突" in result.text


def test_training_metric_does_not_fill_test_metric():
    result = answer("我的模型测试准确率是多少？", records("模型训练准确率为95%，测试准确率尚未测量。"))
    assert result.abstention_reason == "missing_requested_fact"


def test_conflicting_training_metric_does_not_poison_test_metric():
    result = answer("我的模型测试准确率是多少？", records(
        "模型训练准确率为95%，测试准确率为90%。", "模型训练准确率为96%，测试准确率为90%。"))
    assert result.abstention_reason is None


def test_image_count_must_be_explicit():
    result = answer("我上传的数据集有多少张图片？", records("我上传的数据集有87MB，图片数量未统计。", section="数据集统计"))
    assert result.abstention_reason == "missing_requested_fact"


@pytest.mark.parametrize("value", ["100MB", "100.5", "100万"])
def test_image_count_does_not_parse_numeric_prefix_of_other_units(value):
    result = answer("我上传的数据集有多少张图片？", records(
        f"我上传的数据集的图片数量为{value}。", section="数据集统计"))
    assert result.abstention_reason == "missing_requested_fact"


def test_all_requested_metrics_need_values():
    result = answer("我的模型精度和损失是多少？", records("模型精度为95%，损失尚未测量。"))
    assert result.abstention_reason == "missing_requested_fact"
    assert "损失" in result.text


def test_explicit_record_question_checks_values_without_first_person():
    result = answer("实验记录中模型精度是多少？", records("模型精度尚未测量。"))
    assert result.abstention_reason == "missing_requested_fact"
    supplied = answer("实验记录中模型精度是多少？", records("模型精度为95%。"))
    assert supplied.abstention_reason is None


def test_plain_learning_comparison_does_not_get_refused():
    result = answer("训练集和测试集的作用有什么不同？", records(
        "训练集用于拟合参数，测试集用于评估模型。", section="课程笔记"))
    assert result.abstention_reason is None
    assert "拟合参数" in result.text and "评估模型" in result.text


def test_conflict_is_checked_before_generator():
    class Generator:
        def generate(self, prompt):
            raise AssertionError("Do not generate through conflicting evidence")
    result = answer("我的模型精度是多少？", records("模型精度为95%。", "模型精度为96%。"), Generator())
    assert result.abstention_reason == "conflicting_records"
