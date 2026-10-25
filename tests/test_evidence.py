"""Paired behavioral controls: refuse missing facts, accept supplied records."""
import pytest

from course_rag_assistant.answer import answer
from course_rag_assistant.ingest import Chunk
from course_rag_assistant.retrieval import Hit


def hit(text, section="教材", doc="notes.md"):
    return Hit(Chunk(text, doc, 0, section), 1, "fused")


@pytest.mark.parametrize("question,text", [
    ("我的模型测试准确率是多少？", "模型测试准确率可以用于评估泛化能力。"),
    ("我们班考试几点开始？", "考试可以评估学生的学习成果。"),
    ("本周作业截止日期是什么？", "作业截止日期由教师决定。"),
    ("我的学号对应哪个小组？", "学号可以用于小组分组。"),
    ("我上传的数据集有多少张图片？", "数据集含有一千张图片。"),
])
def test_topic_match_cannot_supply_personal_facts(question, text):
    result = answer(question, [hit(text)])
    assert result.abstention_reason == "missing_context_record"
    assert not result.hits
    assert not result.unsupported_claims  # No claim was generated.


@pytest.mark.parametrize("question,text,section", [
    ("我的模型测试准确率是多少？", "模型测试准确率为92%。", "实验记录"),
    ("我们班考试几点开始？", "我们班考试9点开始。", "课程通知"),
    ("本周作业截止日期是什么？", "本周作业截止日期为10月30日。", "作业通知"),
    ("我的学号对应哪个小组？", "我的学号对应第三小组。", "分组名单"),
    ("我上传的数据集有多少张图片？", "我上传的数据集有300张图片。", "数据集清单"),
])
def test_explicit_record_can_support_contextual_question(question, text, section):
    result = answer(question, [hit(text, section)])
    assert result.abstention_reason is None
    assert text in result.text


def test_record_title_alone_does_not_support_requested_fact():
    result = answer("我的模型测试准确率是多少？", [hit("今天安装了显卡驱动。", "实验记录")])
    assert result.abstention_reason == "insufficient_query_coverage"


def test_first_person_advice_does_not_require_personal_record():
    result = answer("我们如何提高模型准确率？", [hit("提高模型准确率可以使用更多训练数据。")])
    assert result.abstention_reason is None


def test_polite_first_person_definition_is_answerable():
    result = answer("我想了解自动微分是什么？", [hit("自动微分通过计算图计算导数。")])
    assert result.abstention_reason is None


def test_generic_question_words_do_not_establish_relevance():
    result = answer("如何办理签证申请？", [hit("申请计算资源需要填写申请表。")])
    assert result.abstention_reason == "insufficient_query_coverage"


def test_choose_supported_candidate_not_top_ranked_noise():
    result = answer("自动微分是什么？", [hit("微分的历史。"), hit("自动微分计算梯度。")])
    assert "自动微分计算梯度" in result.text
    assert "微分的历史" not in result.text


def test_rejected_evidence_does_not_call_generator():
    class UnexpectedGenerator:
        def generate(self, prompt):
            raise AssertionError("Rejected questions must not reach generation")

    result = answer("我的模型测试准确率是多少？", [hit("模型测试准确率是一个指标。")], UnexpectedGenerator())
    assert result.abstention_reason == "missing_context_record"
