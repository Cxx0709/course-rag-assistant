"""Check bounded practice against the two pinned real PDFs in an isolated library."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from course_rag_assistant.compare import normalized
from course_rag_assistant.quiz import QuizStore
from course_rag_assistant.quiz_scope import practice_scope
from course_rag_assistant.web import create_app
from course_rag_assistant.web.library import Library


def grounded(question, chunks):
    # _source_units normalizes PDF line-wrap whitespace. Keep source identity,
    # page, section and all non-whitespace characters exact.
    return any((c.doc, c.page, c.section) == (question["doc"], question["page"], question["section"])
               and normalized(question["sentence"]) in normalized(c.text) for c in chunks)


def evaluate(root, folder):
    sources = json.loads((root / "eval/learning-flow-v1.json").read_text(encoding="utf-8"))["sources"]
    quality_raw = (root / "eval/quiz-quality-v1.json").read_bytes()
    quality = json.loads(quality_raw)
    library = Library(folder)
    documents = []
    for source in sources:
        raw = (root / source["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == source["sha256"], source["path"]
        documents.append(library.add(Path(source["path"]).name, raw))
    base = {"document_ids": [d.id for d in documents], "count": 5}
    checks = {}
    with TestClient(create_app(folder, generator_factory=lambda: None), base_url="http://127.0.0.1") as client:
        client.headers["X-Workspace-Token"] = client.get("/api/state").json()["token"]
        def post(path, body):
            response = client.post(path, json=body)
            response.raise_for_status()
            return response.json()
        store = QuizStore(folder / "quiz.sqlite3")
        sections = post("/api/quiz/sections", base)["sections"]
        chapter = next(s for s in sections if s["document_id"] == documents[0].id and s["title"] == "4.4 模型选择、欠拟合和过拟合")
        checks["chapter_pages"] = (chapter["page_from"], chapter["page_to"]) == (157, 167)
        target = {**base, "target_concept": "过拟合"}
        quiz = post("/api/quiz", target)
        questions = store.load(quiz["quiz_id"])["questions"]
        checks["only_target_no_padding"] = len(questions) == 1 and questions[0]["concept"] == "过拟合"
        checks["direct_definition"] = questions[0]["sentence"] == "过拟合是指训练误差远小于验证误差。" and questions[0]["page"] == 167
        checks["no_navigation_options"] = not {"小结", "论坛", "前言", "目录"} & set(questions[0]["options"])
        all_chunks = [c for d in documents for c in d.chunks]
        checks["concept_source_support"] = all(grounded(q, all_chunks) for q in questions)
        checks["public_key_hidden"] = all(set(q) == {"id", "prompt", "options"} for q in quiz["questions"])
        wrong = {q["id"]: (q["correct_index"] + 1) % len(q["options"]) for q in questions}
        result = post(f"/api/quiz/{quiz['quiz_id']}/grade", {"choices": wrong})
        checks["scope_saved_with_grade"] = result["scope"] == quiz["scope"] and result["score"] == 0
        checks["page_links"] = all(r["source_url"].endswith("#page=167") and client.get(r["source_url"], headers={"Range": "bytes=0-4"}).status_code in (200, 206) for r in result["results"])
        review = post("/api/review", target)
        checks["target_review"] = [r["concept"] for r in review["gaps"]] == ["过拟合"]
        checks["other_target_review_empty"] = not post("/api/review", {**base, "target_concept": "正则化"})["gaps"]
        chapter_scope = {**base, "section_id": chapter["id"]}
        checks["chapter_review_includes_target"] = len(post("/api/review", chapter_scope)["gaps"]) == 1
        chapter_quiz = post("/api/quiz", chapter_scope)
        chapter_questions = store.load(chapter_quiz["quiz_id"])["questions"]
        chapter_chunks, _ = practice_scope(documents, section_id=chapter["id"])
        checks["chapter_five_questions"] = len(chapter_questions) == 5
        checks["chapter_source_support"] = all(157 <= q["page"] <= 167 and grounded(q, chapter_chunks) for q in chapter_questions)
        checks["chapter_options_in_scope"] = all(any(option in c.text for c in chapter_chunks) for q in chapter_questions for option in q["options"])
        checks["chapter_knowledge_statements"] = len(chapter_questions) == len(quality["chapter_questions"]) and all(
            q["concept"] == expected["concept"] and q["page"] == expected["page"]
            and expected["contains"] in q["sentence"]
            for q, expected in zip(chapter_questions, quality["chapter_questions"]))
        checks["no_observed_previews"] = all(q["sentence"] not in quality["excluded_previews"] for q in chapter_questions)
        checks["target_related_options"] = set(questions[0]["options"]) == set(quality["target_options"])
        checks["no_procedural_options"] = all(not set(q["options"]) & set(quality["excluded_options"]) for q in [*questions, *chapter_questions])
        checks["no_bilingual_answer_hint"] = all(
            alias not in q["prompt"] for q in chapter_questions
            for alias in ("training error", "generalization error"))
        checks["bilingual_source_preserved"] = all(
            alias in chapter_questions[i]["sentence"]
            for i, alias in enumerate(("training error", "generalization error")))
        checks["unknown_target_no_fallback"] = client.post("/api/quiz", json={**base, "target_concept": "不存在的课程概念"}).status_code == 422
        checks["foreign_chapter_rejected"] = client.post("/api/quiz", json={"document_ids": [documents[1].id], "section_id": chapter["id"]}).status_code == 422
    with TestClient(create_app(folder), base_url="http://127.0.0.1") as client:
        client.headers["X-Workspace-Token"] = client.get("/api/state").json()["token"]
        checks["review_survives_restart"] = client.post("/api/review", json=target).json() == review
        retry = client.post("/api/quiz", json=target).json()
        keys = store.load(retry["quiz_id"])["questions"]
        result = client.post(f"/api/quiz/{retry['quiz_id']}/grade", json={"choices": {q["id"]: q["correct_index"] for q in keys}}).json()
        checks["correct_retry_clears_review"] = result["score"] == 1 and not client.post("/api/review", json=chapter_scope).json()["gaps"]
    return {"sources": sources, "quality_manifest_sha256": hashlib.sha256(quality_raw).hexdigest(),
            "checks": checks, "passed": all(checks.values()),
            "chapter": chapter, "concept_questions": questions, "chapter_questions": chapter_questions,
            "limitations": ["Literal source recognition, not mastery assessment; target name can reveal the answer.",
                            "Chapter boundaries depend on extracted numbered headings; source quality still requires human review."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="eval/reports/targeted-practice-v2.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    (root / ".tmp").mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="targeted-eval-", dir=root / ".tmp") as folder:
        report = evaluate(root, Path(folder))
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": report["checks"], "output": str(output)}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
