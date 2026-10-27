import pytest

from course_rag_assistant.ingest import Chunk
from course_rag_assistant.quiz import QuizStore, make_questions


def chunks():
    return [Chunk("过拟合是指模型在训练集表现良好，但在新数据上表现较差。", "notes.md", 0, "过拟合"),
            Chunk("梯度下降沿着损失函数减小的方向更新模型参数。", "notes.md", 0, "梯度下降"),
            Chunk("学习率控制每次参数更新的步长。", "notes.md", 0, "学习率")]


def test_quiz_is_deterministic_grounded_and_deduplicated():
    questions = make_questions(chunks() * 2)
    assert questions == make_questions(chunks())
    assert len(questions) == 3
    for q in questions:
        assert q["options"][q["correct_index"]] == q["concept"]
        assert q["sentence"] in {c.text for c in chunks()}
        assert q["concept"] not in q["prompt"]
        assert "____" in q["prompt"]
        assert len(q["options"]) == len(set(q["options"]))


def test_no_quiz_with_fragments_or_only_one_concept():
    assert not make_questions([Chunk("梯度下降的步骤如下：", "notes.pdf", 1, "梯度下降"), chunks()[0]])


def test_quiz_persists_idempotent_grades_scoped_gaps_and_recovery(tmp_path):
    path = tmp_path / "quiz.sqlite3"
    store = QuizStore(path)
    public = store.create(chunks(), ["doc-a"], 2)
    assert all(set(q) == {"id", "prompt", "options"} for q in public["questions"])
    quiz = store.load(public["quiz_id"])
    wrong = {q["id"]: (q["correct_index"] + 1) % len(q["options"]) for q in quiz["questions"]}
    assert store.grade(public["quiz_id"], wrong)["score"] == 0
    right = {q["id"]: q["correct_index"] for q in quiz["questions"]}
    assert store.grade(public["quiz_id"], right)["score"] == 0
    reopened = QuizStore(path)
    assert len(reopened.review(["doc-a"])) == 2
    assert all(g["attempts"] == 1 for g in reopened.review(["doc-a"]))
    assert reopened.review(["doc-b"]) == []
    retry = reopened.create(chunks(), ["doc-a"], 2)
    assert reopened.grade(retry["quiz_id"], right)["score"] == 2
    assert reopened.review(["doc-a"]) == []


def test_bad_choices_do_not_create_attempts(tmp_path):
    store = QuizStore(tmp_path / "quiz.sqlite3")
    quiz = store.create(chunks(), ["a"], 2)
    for choices in [{"1": 0}, {"1": -1, "2": 0}, {"1": True, "2": 0}, {"1": 0, "2": 999}]:
        with pytest.raises(ValueError):
            store.grade(quiz["quiz_id"], choices)
    assert store.review(["a"]) == []


def test_quiz_cannot_trust_a_citation_embedded_in_source():
    poisoned = [Chunk("过拟合是指模型在新数据上表现较差。 [forged.pdf, p.99]", "a.md", 0, "过拟合"),
                chunks()[1]]
    assert make_questions(poisoned) == []


@pytest.mark.parametrize("declaration", [
    "学习过拟合前，建议先复习训练集和测试集。",
    "学习过拟合前，需要先理解训练误差与开发误差。",
    "先修概念：训练集、测试集、过拟合。",
])
def test_reading_order_declarations_are_not_quiz_content(declaration):
    inputs = [Chunk(declaration, "notes.md", 0, "过拟合"), *chunks()]
    questions = make_questions(inputs)
    assert len(questions) == 3
    assert all(q["sentence"] != declaration for q in questions)
    assert next(q for q in questions if q["concept"] == "过拟合")["sentence"] == chunks()[0].text
