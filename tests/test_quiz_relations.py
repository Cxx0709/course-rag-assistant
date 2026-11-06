from dataclasses import replace

import pytest

from course_rag_assistant.ingest import Chunk
from course_rag_assistant.quiz import make_questions
from course_rag_assistant.quiz_candidates import (
    candidate_prompt,
    extract_candidates,
    prompt_rejection,
)

PAIR = ("在统计学中，前者（对超出已知观测范围进行预测）称为外推法（extrapolation），"
        "而后者（在现有观测值之间进行估计）称为内插法（interpolation）。")
INTRO = "长短期记忆网络引入了记忆元（memory cell），或简称为单元（cell）。"
CONTEXT = INTRO + "为了控制记忆元，我们需要许多门。"
OUTPUT = "其中一个门用来从单元中输出条目，我们将其称为输出门（output gate）。"
INPUT = "另外一个门用来决定何时将数据读入单元，我们将其称为输入门（input gate）。"


def test_parallel_definitions_keep_both_full_source_and_scope_but_hide_the_other_description():
    q1, q2 = make_questions([Chunk(PAIR, "course.pdf", 308, "8.1 序列模型")])
    assert [q["concept"] for q in (q1, q2)] == ["外推法", "内插法"]
    for q in (q1, q2):
        assert q["sentence"] == PAIR and q["page"] == 308
        assert set(q["options"]) == {"外推法", "内插法"}
        assert q["options"][q["correct_index"]] == q["concept"]
        assert q["prompt"].startswith("按资料原文补全空缺：在统计学中，")
        assert q["prompt"].count("____") == 1
        assert not any(s in q["prompt"] for s in ("前者", "后者", "外推法", "内插法", "extrapolation", "interpolation"))
    assert "超出已知观测范围" in q1["prompt"] and "现有观测值之间" not in q1["prompt"]
    assert "现有观测值之间" in q2["prompt"] and "超出已知观测范围" not in q2["prompt"]


def test_related_options_cannot_borrow_from_other_pairs_or_documents():
    a = Chunk(PAIR, "course.pdf", 308, "8.1 序列模型")
    b = Chunk(PAIR.replace("外推法", "第一方法").replace("内插法", "第二方法"), "course.pdf", 309, "8.1 序列模型")
    unrelated = Chunk("随机采样是一种按随机机制抽取样本的方法。", "course.pdf", 309, "采样")
    rows = make_questions([a, b, unrelated], count=10)
    assert len(rows) == 4
    assert all(set(q["options"]) == ({"外推法", "内插法"} if q["page"] == 308 else {"第一方法", "第二方法"}) for q in rows)
    assert make_questions([a, replace(a, doc="another.pdf")]) == make_questions([a])


@pytest.mark.parametrize("source", [
    PAIR.replace("），而后者", "）。另一个标题。而后者"),
    PAIR.replace("（interpolation）。", "（interpolation），仅在局部连续时成立。"),
    PAIR.replace("extrapolation", "when samples match"),
    PAIR.replace("interpolation", "extrapolation"),
    PAIR.replace("在现有观测值之间进行估计", "对超出已知观测范围进行预测"),
    PAIR.replace("对超出已知观测范围进行预测", "上述方法在区间外进行预测"),
])
def test_ambiguous_or_incomplete_pairs_do_not_become_questions(source):
    assert make_questions([Chunk(source, "course.pdf", 3, "方法")]) == []


def test_adjacent_functions_keep_shared_object_context_and_complete_feedback():
    source = CONTEXT + OUTPUT + INPUT
    rows = make_questions([Chunk(source, "course.pdf", 360, "9.2 记忆网络")])
    assert len(rows) == 2
    for q in rows:
        assert q["sentence"] == source
        assert q["page"] == 360 and set(q["options"]) == {"输出门", "输入门"}
        assert "长短期记忆网络引入了记忆元，或简称为单元。" in q["prompt"]
        assert "为了控制记忆元，我们需要许多门。" in q["prompt"]
        assert not any(x in q["prompt"] for x in ("其中一个", "另外一个", "输出门", "输入门", "cell", "gate"))
    assert "从单元中输出条目" in rows[0]["prompt"] and "将数据读入" not in rows[0]["prompt"]
    assert "何时将数据读入单元" in rows[1]["prompt"] and "输出条目" not in rows[1]["prompt"]
    assert make_questions([Chunk(source, "course.pdf", 360, "9.2 记忆网络")], target_concept="输入门") == [
        {**rows[1], "id": "1"}]


@pytest.mark.parametrize("source", [
    OUTPUT + INPUT,
    CONTEXT + OUTPUT + "\n\n" + INPUT,
    CONTEXT + OUTPUT + "\n9.3 另一节\n" + INPUT,
    CONTEXT + OUTPUT + "另外讨论一种算法。" + INPUT,
    CONTEXT + OUTPUT + INPUT.replace("gate）。", "gate），仅在特定条件下成立。"),
    CONTEXT + OUTPUT + INPUT.replace("另外一个", "其中一个"),
    CONTEXT + OUTPUT + INPUT.replace("单元", "缓冲区"),
    CONTEXT + OUTPUT + INPUT.replace("input gate", "when active"),
])
def test_function_groups_require_contiguous_context_and_both_valid_members(source):
    assert make_questions([Chunk(source, "course.pdf", 360, "9.2 记忆网络")]) == []


def test_functions_never_join_across_page_or_chunk_boundaries():
    for page in (360, 361):
        assert make_questions([Chunk(CONTEXT + OUTPUT, "course.pdf", 360, "9.2 记忆网络"),
                               Chunk(INPUT, "course.pdf", page, "9.2 记忆网络")]) == []


def test_unresolved_other_member_is_rejected_even_with_a_separately_defined_peer():
    candidate, = extract_candidates([Chunk(INPUT, "course.pdf", 360, "输入门")])
    assert prompt_rejection(candidate, candidate_prompt(candidate)) == "unresolved_reference"
    assert make_questions([Chunk(INPUT, "course.pdf", 360, "输入门"),
                           Chunk("输出门用于控制记忆元中的信息输出。", "course.pdf", 360, "输出门")]) == []


def test_function_template_supports_a_different_system_and_preserves_negation():
    source = (CONTEXT + OUTPUT + INPUT).replace("长短期记忆网络", "缓存系统").replace("记忆元", "数据槽")
    source = source.replace("单元", "槽位").replace("门", "队列").replace("从槽位中输出条目", "保留不能立即发出的槽位条目")
    rows = make_questions([Chunk(source, "notes.md", 0, "缓存")])
    assert len(rows) == 2
    assert "不能立即发出" in rows[0]["prompt"]
    assert all(set(q["options"]) == {"输出队列", "输入队列"} for q in rows)
