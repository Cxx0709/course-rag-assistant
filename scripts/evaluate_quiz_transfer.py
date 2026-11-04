"""Observe two preselected chapters without tuning against their quiz output."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from course_rag_assistant.answer import CitedAnswer, _sentences, verify_citations
from course_rag_assistant.compare import normalized
from course_rag_assistant.quiz import QuizStore
from course_rag_assistant.quiz_scope import practice_scope, section_ranges
from course_rag_assistant.retrieval import Hit
from course_rag_assistant.web import create_app
from course_rag_assistant.web.library import Library


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="eval/reports/quiz-transfer-v2.json")
    args = parser.parse_args()
    manifest_path = root / "eval/quiz-transfer-v1.json"
    spec = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = root / spec["source"]["path"]
    assert digest(source) == spec["source"]["sha256"], "Source changed"
    (root / ".tmp").mkdir(exist_ok=True)
    report = {"purpose": spec["purpose"], "source": spec["source"],
              "evaluation_phase": "development_regression_after_first_observation",
              "manifest_sha256": digest(manifest_path), "default_library_modified": False,
              "code_sha256": {str(p.relative_to(root)): digest(p) for p in [
                  root / "src/course_rag_assistant/quiz.py", root / "src/course_rag_assistant/quiz_content.py"]},
              "chapters": [], "human_teaching_review": "pending"}
    with TemporaryDirectory(prefix="quiz-transfer-", dir=root / ".tmp") as folder:
        library = Library(Path(folder))
        document = library.add(source.name, source.read_bytes())
        sections = section_ranges(document)
        with TestClient(create_app(Path(folder), generator_factory=lambda: None), base_url="http://127.0.0.1") as client:
            client.headers["X-Workspace-Token"] = client.get("/api/state").json()["token"]
            store = QuizStore(Path(folder) / "quiz.sqlite3")
            for case in spec["chapters"]:
                section = next(s for s in sections if s["title"] == case["title"])
                chunks, _ = practice_scope([document], section_id=section["id"])
                response = client.post("/api/quiz", json={"document_ids": [document.id],
                                      "section_id": section["id"], "count": spec["requested_count"]})
                payload = response.json()
                questions = store.load(payload["quiz_id"])["questions"] if response.status_code == 201 else []
                checks = {"recognized_pages": [section["page_from"], section["page_to"]] == case["recognized_pages"],
                          "explicit_result": response.status_code == 201 or (
                              response.status_code == 422 and "不会从范围外凑题" in payload.get("detail", "")),
                          "bounded_count": len(questions) <= spec["requested_count"]}
                evidence = []
                for q in questions:
                    matching = [c for c in chunks if (c.doc, c.page, c.section) == (q["doc"], q["page"], q["section"])
                                and normalized(q["sentence"]) in normalized(c.text)]
                    verified = verify_citations(CitedAnswer(
                        "\n".join(f"{s} {q['citation']}" for s in _sentences(q["sentence"])),
                        [Hit(c, 1, "quiz") for c in matching], []))
                    evidence.append(bool(matching) and not verified.abstention_reason and not verified.unsupported_claims)
                checks["contiguous_verified_source"] = all(evidence)
                checks["options_in_scope"] = all(any(o in c.text for c in chunks) for q in questions for o in q["options"])
                checks["correct_option_hidden_in_prompt"] = all(q["concept"] not in q["prompt"] for q in questions)
                checks["public_key_hidden"] = response.status_code != 201 or all(
                    set(q) == {"id", "prompt", "options"} for q in payload["questions"])
                checks["shortfall_explained"] = len(questions) >= spec["requested_count"] or (
                    "不扩大范围" in payload.get("note", "") or response.status_code == 422)
                checks["observed_alias_hint_removed"] = all("loss function" not in q["prompt"] for q in questions)
                checks["observed_dangling_prompts_removed"] = all(
                    not any(fragment in q["prompt"] for fragment in (
                        "现在引用上述的第一个原则", "现在引用上述的第二个原则", "然而这种方法有一个问题"))
                    for q in questions)
                report["chapters"].append({"title": case["title"], "recognized_pages": case["recognized_pages"],
                    "status_code": response.status_code, "generated_count": len(questions),
                    "note": payload.get("note", payload.get("detail")), "questions": questions,
                    "checks": checks, "passed": all(checks.values())})
    report["passed"] = all(c["passed"] for c in report["chapters"])
    report["all_chapters_produce_practice"] = all(c["generated_count"] > 0 for c in report["chapters"])
    report["teaching_acceptance"] = "not_established"
    report["limitations"] = ["Structural checks do not establish question value or semantic uniqueness.",
                              "No minimum quiz yield is assumed; empty output is reported explicitly.",
                              "Chapter scope depends on extracted headings and requires source review."]
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
