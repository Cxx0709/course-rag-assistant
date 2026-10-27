"""Exercise a real-material learning flow in a disposable offline library.

Uses the answer key only in this test harness to simulate one wrong response
and a correct retry. Passing is not evidence of human learning or semantic QA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from course_rag_assistant.compare import normalized
from course_rag_assistant.quiz import QuizStore
from course_rag_assistant.web import create_app
from course_rag_assistant.web.library import Library
from scripts.evaluate_course_features import digest, grounded


def run_flow(spec, root, folder):
    library = Library(folder)
    pdfs = [library.add(Path(s["path"]).name, (root / s["path"]).read_bytes()) for s in spec["sources"]]
    note = library.add(Path(spec["note"]["path"]).name, (root / spec["note"]["path"]).read_bytes())
    ids = [d.id for d in pdfs]
    scope = {"document_ids": [note.id], "count": 5}
    checks, comparisons = {}, []
    with TestClient(create_app(folder, generator_factory=lambda: None), base_url="http://127.0.0.1") as client:
        client.headers["X-Workspace-Token"] = client.get("/api/state").json()["token"]

        def post(url, body):
            response = client.post(url, json=body)
            response.raise_for_status()
            return response.json()

        for case in spec["comparisons"]:
            result = post("/api/compare", {"document_ids": ids, "concept": case["concept"], "sense": case.get("sense")})
            quotes = [e for row in result["rows"] for cell in row["cells"] for e in cell]
            supported = all(grounded(e, next(d.chunks for d in pdfs if d.id == e["document_id"])) for e in quotes)
            found = [d["status"] == "found" for d in result["documents"]]
            page_hit = all(bool(set(refs) & {e["page"] for e in quotes if e["document_id"] == ids[i]})
                           if expected else not any(e["document_id"] == ids[i] for e in quotes)
                           for i, (expected, refs) in enumerate(zip(case["expected_found"], case["reference_pages"])))
            passed = supported and page_hit and found == case["expected_found"] and result["alignment"]["status"] == case["expected_alignment"]
            comparisons.append({"concept": case["concept"], "sense": case.get("sense"),
                                "status": result["status"], "alignment": result["alignment"]["status"],
                                "found": found, "source_support": supported, "reference_pages": page_hit, "passed": passed})
        checks["comparisons"] = all(c["passed"] for c in comparisons)
        answer = post("/api/ask", {"question": "什么是过拟合？", "document_ids": [*ids, note.id]})
        checks["answer_available"] = not answer["abstained"] and bool(answer["claims"])
        answer_text = "".join(c["text"] for c in answer["claims"])
        checks["answer_defines_target"] = "过拟合" in answer_text and "训练误差" in answer_text and "验证误差" in answer_text
        checks["answer_not_reading_order"] = "建议先复习" not in answer_text and "先修概念" not in answer_text
        sources = {s["id"]: s for s in answer["sources"]}
        checks["answer_source_support"] = all(c["source_ids"] and any(
            normalized(c["text"]) in normalized(sources[s]["text"]) for s in c["source_ids"])
                                             for c in answer["claims"])
        checks["answer_learning_path"] = any(p["concept"] == "过拟合" and p["status"] == "ready"
                                             and [s["concept"] for s in p["steps"]] == spec["review_order"]
                                             for p in answer["learning_plans"])
        checks["source_links"] = all(client.get(s["url"], headers={"Range": "bytes=0-4"}).status_code in (200, 206)
                                    for s in answer["sources"])
        # Observe whole-PDF practice separately; it must not be confused with
        # the focused note workflow, which the user explicitly selects below.
        pdf_quiz = post("/api/quiz", {"document_ids": ids, "count": 5})
        pdf_questions = QuizStore(folder / "quiz.sqlite3").load(pdf_quiz["quiz_id"])["questions"]
        quiz = post("/api/quiz", scope)
        checks["public_quiz_hides_answer_key"] = all(set(q) == {"id", "prompt", "options"} for q in quiz["questions"])
        questions = QuizStore(folder / "quiz.sqlite3").load(quiz["quiz_id"])["questions"]
        checks["practice_has_target"] = any(q["concept"] == "过拟合" for q in questions)
        checks["no_prerequisite_quiz"] = all("建议先复习" not in q["sentence"] and "先修概念" not in q["sentence"] for q in questions)
        checks["practice_grounded_in_draft"] = all(grounded({**q, "text": q["sentence"]}, note.chunks) for q in questions)
        choices = {q["id"]: (q["correct_index"] + int(q["concept"] == "过拟合")) % len(q["options"]) for q in questions}
        graded = post(f"/api/quiz/{quiz['quiz_id']}/grade", {"choices": choices})
        checks["one_wrong_response"] = graded["score"] == graded["total"] - 1
        right = {q["id"]: q["correct_index"] for q in questions}
        checks["grade_is_immutable"] = post(f"/api/quiz/{quiz['quiz_id']}/grade", {"choices": right}) == graded
        review = post("/api/review", scope)
        checks["review_targets_wrong_concept"] = [g["concept"] for g in review["gaps"]] == ["过拟合"]
        checks["review_order"] = bool(review["learning_plans"]) and [s["concept"] for s in review["learning_plans"][0]["steps"]] == spec["review_order"]
        checks["review_lessons_present"] = bool(review["learning_plans"]) and all(s["sources"] for s in review["learning_plans"][0]["steps"])
        checks["review_scope_isolation"] = post("/api/review", {"document_ids": ids})["gaps"] == []
    with TestClient(create_app(folder, generator_factory=lambda: None), base_url="http://127.0.0.1") as client:
        client.headers["X-Workspace-Token"] = client.get("/api/state").json()["token"]
        restored = client.post("/api/review", json=scope).json()
        checks["review_survives_restart"] = restored == review
        retry_response = client.post("/api/quiz", json=scope)
        retry_response.raise_for_status()
        retry = retry_response.json()
        retry_questions = QuizStore(folder / "quiz.sqlite3").load(retry["quiz_id"])["questions"]
        right = {q["id"]: q["correct_index"] for q in retry_questions}
        corrected = client.post(f"/api/quiz/{retry['quiz_id']}/grade", json={"choices": right}).json()
        checks["new_quiz_correct_retry"] = corrected["score"] == corrected["total"] == len(questions)
        final_review = client.post("/api/review", json=scope).json()
        checks["correct_retry_clears_review"] = final_review["gaps"] == final_review["learning_plans"] == []
    return {"checks": checks, "passed": all(checks.values()), "comparisons": comparisons,
            "answer": {"claims": len(answer["claims"]), "sources": [{"name": s["name"], "page": s["page"]} for s in answer["sources"]]},
            "practice_concepts": [q["concept"] for q in questions],
            "practice_evidence": [{"concept": q["concept"], "sentence_sha256": hashlib.sha256(q["sentence"].encode()).hexdigest()} for q in questions],
            "first_score": [graded["score"], graded["total"]], "retry_score": [corrected["score"], corrected["total"]],
            "pdf_only_practice_concepts": [q["concept"] for q in pdf_questions], "review_order": spec["review_order"]}


def evaluate(manifest, root=Path(".")):
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    for source in [*spec["sources"], spec["note"]]:
        if digest(root / source["path"]) != source["sha256"]:
            raise ValueError(f"Input fingerprint changed: {source['path']}")
    temp_root = root / ".tmp"
    temp_root.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="learning-flow-", dir=temp_root) as folder:
        run = run_flow(spec, root, Path(folder))
    return {"purpose": "Offline software acceptance with real textbooks and an unreviewed assistant-authored note",
            "manifest_sha256": digest(manifest), "run": run, "regression_passed": run["passed"],
            "human_semantic_review": "pending", "teaching_order_review": "pending", "default_library_modified": False,
            "limitations": ["模拟正确重答不表示真人理解或掌握知识。", "复习路径来自待审核草稿，不是教材明确规定的先修关系。",
                            "整本 PDF 默认出题仍可能选择前言等非目标学习内容；本次聚焦练习使用单独勾选的课程笔记。"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("eval/learning-flow-v1.json"))
    parser.add_argument("--output", type=Path, default=Path("eval/reports/learning-flow-v1.json"))
    args = parser.parse_args()
    report = evaluate(args.manifest)
    report["code_sha256"] = {str(p): digest(p) for p in [Path(__file__), *sorted(Path("src/course_rag_assistant").rglob("*.py"))]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["regression_passed"], **report["run"]}, ensure_ascii=False, indent=2))
    if not report["regression_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
