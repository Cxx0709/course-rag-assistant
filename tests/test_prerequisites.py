from course_rag_assistant.ingest import Chunk
from course_rag_assistant.prerequisites import PrerequisiteGraph


def lesson(name, body, doc="notes.md", page=0):
    return Chunk(body, doc, page, name)


def test_transitive_review_order_has_separate_relation_and_lesson_evidence():
    chunks = [lesson("朴素贝叶斯", "先修概念：条件独立、贝叶斯定理。"),
              lesson("条件独立", "先修概念：条件概率。"),
              lesson("贝叶斯定理", "先修概念：条件概率。"),
              lesson("条件概率", "条件概率是在给定事件发生时另一个事件发生的概率。", "probability.pdf", 3)]
    plan = PrerequisiteGraph(chunks).plan("朴素贝叶斯")
    assert plan["status"] == "ready"
    assert [s["concept"] for s in plan["steps"]] == ["条件概率", "条件独立", "贝叶斯定理"]
    first = plan["steps"][0]
    assert first["required_by"] == ["条件独立", "贝叶斯定理"]
    assert first["sources"][0]["doc"] == "probability.pdf"
    assert first["sources"][0]["page"] == 3
    assert plan["steps"][1]["sources"] == []  # A declaration is not a lesson.
    assert plan["relations"][0]["sources"][0]["text"] == chunks[0].text


def test_no_relations_inferred_from_mentions_or_fenced_examples():
    graph = PrerequisiteGraph([lesson("朴素贝叶斯", "朴素贝叶斯使用条件独立假设。\n```\n先修概念：条件概率。\n```")])
    assert graph.plan("朴素贝叶斯")["status"] == "not_declared"
    assert graph.matching_concepts("朴素贝叶斯是什么？") == []


def test_explicit_sentence_and_numbered_heading_are_recognized():
    graph = PrerequisiteGraph([lesson("1.2 过拟合", "学习过拟合前，建议先复习训练集与测试集。"),
                               lesson("训练集与测试集", "训练集用于拟合模型参数，测试集用于评估泛化能力。")])
    assert graph.matching_concepts("怎么理解过拟合？") == ["过拟合"]
    assert graph.plan("1.2 过拟合")["steps"][0]["concept"] == "训练集与测试集"
    assert graph.matching_concepts("什么是梯度下降？") == []


def test_cycle_and_self_loop_never_produce_a_false_learning_order():
    for chunks in [[lesson("甲概念", "先修概念：乙概念。"), lesson("乙概念", "先修概念：甲概念。")],
                   [lesson("甲概念", "先修概念：甲概念。")]]:
        plan = PrerequisiteGraph(chunks).plan("甲概念")
        assert plan["status"] == "cycle"
        assert not plan["steps"]
        assert plan["relations"]


def test_graph_is_scoped_deduplicated_and_bounded():
    declaration = lesson("目标概念", "先修概念：前置概念、前置概念。")
    scoped = PrerequisiteGraph([declaration, declaration])
    plan = scoped.plan("目标概念")
    assert len(plan["relations"]) == len(plan["relations"][0]["sources"]) == 1
    assert plan["steps"][0]["sources"] == []
    chain = [lesson(f"概念{i}", f"先修概念：概念{i + 1}。") for i in range(40)]
    limited = PrerequisiteGraph(chain).plan("概念0")
    assert limited["status"] == "truncated"
    assert not limited["steps"]
