import pytest
from fastapi.testclient import TestClient

from course_rag_assistant.ingest import Chunk
from course_rag_assistant.quiz import QuizStore, make_questions
from course_rag_assistant.quiz_scope import practice_scope, review_matches, section_ranges
from course_rag_assistant.web import create_app
from course_rag_assistant.web.library import Document, Library


def document(source="book.pdf"):
    titles = ["前言", "1 第一章", "1.1 拟合", "1.1.1 示例", "小结", "1.10 训练", "2 第二章"]
    return Document("a", "book.pdf", source, 1, [
        Chunk(f"这是章节中的完整讲解内容：{title}。", source, i + 1, title)
        for i, title in enumerate(titles)])


def test_pdf_chapter_hierarchy_excludes_preface_and_next_peer():
    doc = document()
    sections = section_ranges(doc)
    assert [s["title"] for s in sections] == ["1 第一章", "1.1 拟合", "1.1.1 示例", "1.10 训练", "2 第二章"]
    assert [(s["page_from"], s["page_to"]) for s in sections] == [(2, 6), (3, 5), (4, 5), (6, 6), (7, 7)]
    chunks, scope = practice_scope([doc], section_id=sections[1]["id"])
    assert chunks == doc.chunks[2:5]
    assert "start" not in scope["section"]
    assert section_ranges(document()) == sections


def test_headingless_pdf_has_no_invented_chapters():
    doc = document()
    doc.chunks = doc.chunks[:1]
    assert section_ranges(doc) == []


def test_markdown_parent_children_and_same_name_documents():
    doc = document("one.md")
    doc.chunks = [Chunk("完整的课程讲解内容保存在这里。", doc.source, 0, s)
                  for s in ["学习", "学习 / 拟合", "学习 / 训练", "学习率"]]
    sections = section_ranges(doc)
    assert [(s["start"], s["end"]) for s in sections] == [(0, 3), (1, 2), (2, 3), (3, 4)]
    other = Document("b", doc.name, "two.md", 1, doc.chunks)
    assert sections[0]["id"] != section_ranges(other)[0]["id"]
    with pytest.raises(ValueError, match="不属于"):
        practice_scope([other], section_id=sections[0]["id"])


@pytest.mark.parametrize("target", ["", "  ", "单", "长" * 41])
def test_invalid_concept_never_falls_back_to_all(target):
    with pytest.raises(ValueError, match="目标概念"):
        practice_scope([document()], target_concept=target)


def test_target_requires_exact_concept_and_prefers_definition():
    chunks = [Chunk("条件独立是指给定条件后两个事件相互独立。", "a.pdf", 1, "条件独立"),
              Chunk("过拟合在许多机器学习任务中都会出现。", "a.pdf", 2, "过拟合"),
              Chunk("• 过拟合是指训练误差远小于验证误差。", "a.pdf", 3, "小结"),
              Chunk("欠拟合是指模型无法继续减少训练误差。", "a.pdf", 3, "小结"),
              Chunk("本节小结概括了本章的主要学习内容。", "a.pdf", 3, "小结")]
    assert make_questions(chunks, target_concept="独立") == []
    questions = make_questions(chunks, target_concept="过拟合")
    assert len(questions) == 1
    assert questions[0]["page"] == 3
    assert questions[0]["concept"] == "过拟合"
    assert "小结" not in questions[0]["options"]
    assert "欠拟合" in questions[0]["options"]
    assert make_questions(chunks, target_concept="量子力学") == []


def test_review_chapter_checks_exact_passage_not_just_page():
    chunk = Chunk("过拟合是指训练误差远小于验证误差。", "a.pdf", 3, "小结")
    row = dict(doc=chunk.doc, page=chunk.page, section=chunk.section, sentence=chunk.text, concept="过拟合")
    scope = {"section": {"id": "x"}, "target_concept": "过拟合"}
    assert review_matches(row, [chunk], scope)
    assert not review_matches({**row, "concept": "欠拟合"}, [chunk], scope)
    assert not review_matches({**row, "sentence": "同页其他章节的内容。"}, [chunk], scope)


def test_max_length_target_supported_and_single_concept_cannot_borrow_options():
    target = "课" * 40
    chunk = Chunk(f"{target}是指本课程中的特定训练过程。", "a.md", 0, target)
    assert make_questions([chunk], target_concept=target) == []
    other = Chunk("学习率控制每次参数更新的步长。", "a.md", 0, "学习率")
    assert make_questions([chunk, other], target_concept=target)[0]["concept"] == target


NOTE = """# 前言
课程介绍用于说明整本笔记的编排方式和学习目标。
# 拟合
## 过拟合
过拟合是指模型在训练集表现良好，但在新数据上表现较差。
## 欠拟合
欠拟合是指模型无法继续减少训练误差。
## 学习率
学习率控制每次参数更新的步长。
# 其他
梯度下降沿着损失函数减小的方向更新模型参数。
"""


def test_api_scope_intersection_review_and_restart(tmp_path):
    doc = Library(tmp_path).add("课程.md", NOTE.encode())
    with TestClient(create_app(tmp_path), base_url="http://127.0.0.1") as client:
        client.headers["X-Workspace-Token"] = client.get("/api/state").json()["token"]
        base = {"document_ids": [doc.id]}
        def post(url, body):
            response = client.post(url, json=body)
            assert response.is_success, response.text
            return response.json()
        sections = post("/api/quiz/sections", base)["sections"]
        chapter = next(s for s in sections if s["title"] == "拟合")
        other = next(s for s in sections if s["title"] == "其他")
        scope = {**base, "target_concept": " 过拟合 ", "section_id": chapter["id"]}
        quiz = post("/api/quiz", scope)
        assert len(quiz["questions"]) == 1
        assert "1 / 5" in quiz["note"]
        assert quiz["scope"]["target_concept"] == "过拟合"
        assert set(quiz["questions"][0]) == {"id", "prompt", "options"}
        saved = QuizStore(tmp_path / "quiz.sqlite3").load(quiz["quiz_id"])
        assert saved["scope"] == quiz["scope"]
        q = saved["questions"][0]
        assert set(q["options"]) <= {"过拟合", "欠拟合", "学习率"}
        result = post(f"/api/quiz/{quiz['quiz_id']}/grade", {"choices": {q["id"]: (q["correct_index"] + 1) % len(q["options"])}})
        assert result["scope"] == quiz["scope"]
        assert result["score"] == 0
        review = post("/api/review", scope)
        assert [g["concept"] for g in review["gaps"]] == ["过拟合"]
        assert post("/api/review", {**base, "target_concept": "学习率"})["gaps"] == []
        assert post("/api/review", {**base, "section_id": other["id"]})["gaps"] == []
        for extra in [{"target_concept": "不存在"}, {"target_concept": "  "}, {"section_id": "foreign"},
                      {"target_concept": "过拟合", "section_id": other["id"]}]:
            assert client.post("/api/quiz", json={**base, **extra}).status_code == 422
        assert client.post("/api/review", json={**base, "section_id": "foreign"}).status_code == 422
        client.headers.pop("X-Workspace-Token")
        assert client.post("/api/quiz/sections", json=base).status_code == 403
    with TestClient(create_app(tmp_path), base_url="http://127.0.0.1") as client:
        client.headers["X-Workspace-Token"] = client.get("/api/state").json()["token"]
        assert client.post("/api/review", json=scope).json() == review
        assert client.post("/api/quiz/sections", json=base).json()["sections"] == sections
        retry = client.post("/api/quiz", json={**base, "section_id": chapter["id"]}).json()
        keys = QuizStore(tmp_path / "quiz.sqlite3").load(retry["quiz_id"])["questions"]
        assert {q["concept"] for q in keys} == {"过拟合", "欠拟合", "学习率"}
        response = client.post(f"/api/quiz/{retry['quiz_id']}/grade", json={"choices": {q["id"]: q["correct_index"] for q in keys}})
        assert response.status_code == 200
        assert client.post("/api/review", json=scope).json()["gaps"] == []
