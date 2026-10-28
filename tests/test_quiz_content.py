import pytest

from course_rag_assistant.ingest import Chunk
from course_rag_assistant.quiz import make_questions


@pytest.mark.parametrize("preview", [
    "为了进一步讨论这一现象，我们需要了解训练误差和泛化误差。",
    "本节将介绍模型复杂性和过拟合的相关内容。",
    "我们接下来会进一步讨论模型复杂性的有关问题。",
    "模型复杂性由什么构成是一个复杂的问题。",
    "如何计算模型复杂性并据此选择一个合适的模型？",
    "请思考模型复杂性和训练误差之间的联系。",
    "为了说明一些关于过拟合和模型复杂性的经典直觉，我们给出一个多项式的例子。",
])
def test_previews_and_questions_never_become_questions_or_options(preview):
    chunks = [Chunk(preview, "a.pdf", 1, "模型复杂性", concepts=["模型复杂性", "训练误差和泛化误差"]),
              Chunk("过拟合是指训练误差远小于验证误差。", "a.pdf", 2, "过拟合"),
              Chunk("欠拟合是指模型无法继续减少训练误差。", "a.pdf", 2, "欠拟合")]
    questions = make_questions(chunks)
    assert {q["concept"] for q in questions} == {"过拟合", "欠拟合"}
    assert all(q["sentence"] != preview for q in questions)
    assert all(set(q["options"]) == {"过拟合", "欠拟合"} for q in questions)


def test_bilingual_definitions_are_extracted_verbatim_before_weak_heading_mentions():
    chunks = [Chunk("数据集大小和模型复杂性之间通常存在关系。", "a.pdf", 1, "数据集大小"),
              Chunk("训练误差（training error）是指，模型在训练数据集上计算得到的误差。", "a.pdf", 2, "误差"),
              Chunk("泛化误差（generalization error）是指模型在新数据上的期望误差。", "a.pdf", 2, "误差")]
    questions = make_questions(chunks, count=2)
    assert [q["concept"] for q in questions] == ["训练误差", "泛化误差"]
    assert [q["sentence"] for q in questions] == [c.text for c in chunks[1:]]
    assert "training error" in questions[0]["sentence"]
    assert "training error" not in questions[0]["prompt"]
    assert "generalization error" not in questions[1]["prompt"]


@pytest.mark.parametrize("parenthesis", ["（training error）", "(training error)"])
def test_alias_is_hidden_at_repeated_term_but_preserved_in_evidence(parenthesis):
    sentence = f"训练误差{parenthesis}是指模型在训练集上的误差，训练误差{parenthesis}不能代表泛化误差。"
    chunks = [Chunk(sentence, "a.md", 0, "误差"),
              Chunk("学习率控制每次参数更新的步长。", "a.md", 0, "学习率")]
    q = make_questions(chunks, target_concept="训练误差")[0]
    assert q["sentence"] == sentence
    assert "training error" not in q["prompt"]
    assert "不能代表泛化误差" in q["prompt"]
    assert q["prompt"].count("____") == 2


@pytest.mark.parametrize("condition", ["（仅在同分布时）", "（when data is independent）", "（x > 0）", "（2026）"])
def test_cloze_keeps_parenthesized_conditions_and_formulas(condition):
    from course_rag_assistant.quiz_content import cloze_prompt

    sentence = f"训练误差{condition}是指模型在训练数据上计算得到的误差。"
    assert condition in cloze_prompt("训练误差", sentence)


def test_later_definition_replaces_first_mention_without_changing_negation_or_conditions():
    strong = "过拟合是指模型在训练集表现良好但在新数据表现较差，不能只凭训练误差判断泛化性能。"
    other = "当数据来自相同分布时，测试集用于评价模型的泛化性能。"
    chunks = [Chunk("过拟合在许多机器学习任务中都会出现。", "a.md", 0, "过拟合"),
              Chunk(strong, "a.md", 0, "过拟合"), Chunk(other, "a.md", 0, "测试集")]
    questions = make_questions(chunks)
    assert [q["sentence"] for q in questions] == [strong, other]
    assert "不能只凭" in questions[0]["prompt"]
    assert "当数据来自相同分布时" in questions[1]["prompt"]


def test_procedure_headings_do_not_displace_concepts_from_same_numbered_section():
    chunks = [Chunk("激活函数是一种用于引入非线性的函数。", "a.pdf", 20, "4.3 激活函数"),
              Chunk("泛化误差是指模型在新样本上的期望误差。", "a.pdf", 21, "4.4.1 误差"),
              Chunk("数据集大小通常会影响模型的泛化表现。", "a.pdf", 22, "数据集大小"),
              Chunk("欠拟合是指模型无法继续减少训练误差。", "a.pdf", 23, "小结"),
              Chunk("过拟合是指训练误差远小于验证误差。", "a.pdf", 24, "小结"),
              Chunk("初始化模型参数是训练之前需要完成的操作。", "a.pdf", 24, "初始化模型参数")]
    q = make_questions(chunks, target_concept="过拟合")[0]
    assert set(q["options"]) == {"过拟合", "欠拟合", "泛化误差", "数据集大小"}
    assert "_chapter" not in q
    assert make_questions(chunks * 2, target_concept="过拟合") == [q]


def test_explicit_definition_can_override_a_procedural_heading_name():
    chunks = [Chunk("训练模型是指利用训练数据调整模型参数的过程。", "a.md", 0, "训练模型"),
              Chunk("学习率控制每次参数更新的步长。", "a.md", 0, "学习率")]
    assert make_questions(chunks, target_concept="训练模型")[0]["concept"] == "训练模型"


def test_qualified_definition_does_not_become_shorter_target_under_broad_heading():
    chunks = [Chunk("条件独立是指给定条件后两个事件相互独立。", "a.md", 0, "独立"),
              Chunk("学习率控制每次参数更新的步长。", "a.md", 0, "学习率")]
    assert make_questions(chunks, target_concept="独立") == []


def test_only_previews_yield_no_quiz_even_with_many_headings():
    chunks = [Chunk(f"本节将介绍{topic}的相关内容和常见问题。", "a.md", 0, topic)
              for topic in ["过拟合", "欠拟合", "学习率", "梯度下降"]]
    assert make_questions(chunks) == []
