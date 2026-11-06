"""Check known practice chapters and record output on a preselected new chapter."""
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
from course_rag_assistant.quiz_candidates import (
    candidate_prompt,
    extract_candidates,
    prompt_rejection,
)
from course_rag_assistant.quiz_scope import practice_scope, section_ranges
from course_rag_assistant.retrieval import Hit
from course_rag_assistant.web import create_app
from course_rag_assistant.web.library import Library


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="eval/quiz-transfer-v4.json")
    parser.add_argument("--output")
    args = parser.parse_args()
    manifest_path = root / args.manifest
    if not args.output and manifest_path != root / "eval/quiz-transfer-v4.json":
        parser.error("A custom --manifest requires a separate --output path.")
    spec = json.loads(manifest_path.read_text(encoding="utf-8"))
    excluded_sentences = set()
    if spec.get("excluded_question_report"):
        previous = json.loads((root / spec["excluded_question_report"]).read_text(encoding="utf-8"))
        excluded_sentences = {normalized(q["sentence"]) for c in previous["chapters"] for q in c["questions"]}
    for path, expected in spec.get("frozen_code_sha256", {}).items():
        if digest(root / path) != expected:
            raise ValueError(f"Frozen quiz implementation changed: {path}")
    source = root / spec["source"]["path"]
    assert digest(source) == spec["source"]["sha256"], "Source changed"
    (root / ".tmp").mkdir(exist_ok=True)
    report = {"purpose": spec["purpose"], "source": spec["source"],
              "evaluation_phase": "see_individual_chapter_roles",
              "manifest_sha256": digest(manifest_path), "default_library_modified": False,
              "code_sha256": {str(p.relative_to(root)): digest(p) for p in [
                  root / "src/course_rag_assistant/quiz.py", root / "src/course_rag_assistant/quiz_content.py",
                  root / "src/course_rag_assistant/quiz_candidates.py", root / "src/course_rag_assistant/quiz_relations.py",
                  root / "src/course_rag_assistant/quiz_terms.py"]},
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
                checks["single_blank"] = all(q["prompt"].count("____") == 1 for q in questions)
                if "expected_relation" in case:
                    expected = case["expected_relation"]
                    checks["complete_relation_members"] = {q["concept"] for q in questions} == set(expected["terms"])
                    checks["relation_evidence_and_stems"] = all(
                        q["page"] == expected["page"] and set(q["options"]) == set(expected["terms"])
                        and all(term in q["sentence"] and term not in q["prompt"] for term in expected["terms"])
                        and not any(word in q["prompt"] for word in expected["excluded_hints"])
                        for q in questions)
                if excluded_sentences:
                    checks["failed_holdout_sentences_excluded"] = all(
                        normalized(q["sentence"]) not in excluded_sentences for q in questions)
                checks["public_key_hidden"] = response.status_code != 201 or all(
                    set(q) == {"id", "prompt", "options"} for q in payload["questions"])
                checks["shortfall_explained"] = len(questions) >= spec["requested_count"] or (
                    "不扩大范围" in payload.get("note", "") or response.status_code == 422)
                checks["observed_alias_hint_removed"] = all("loss function" not in q["prompt"] for q in questions)
                checks["observed_dangling_prompts_removed"] = all(
                    not any(fragment in q["prompt"] for fragment in (
                        "现在引用上述的第一个原则", "现在引用上述的第二个原则", "然而这种方法有一个问题"))
                    for q in questions)
                if "principles_page" in case:
                    principles = {"平移不变性": ("哪个位置", "相似的反应"), "局部性": ("局部区域", "不过度")}
                    observed = {q["concept"]: q for q in questions}
                    checks["both_grounded_principles"] = all(
                        term in observed and observed[term]["page"] == case["principles_page"]
                        and set(observed[term]["options"]) == set(principles)
                        and all(word in observed[term]["prompt"] for word in words)
                        and not any(hint in observed[term]["prompt"] for hint in ("translation invariance", "locality"))
                        for term, words in principles.items())
                if "contrast_page" in case:
                    row = next((q for q in questions if q["concept"] == "互相关运算"), {})
                    checks["source_contrast"] = (
                        row.get("page") == case["contrast_page"]
                        and set(row.get("options", [])) == {"互相关运算", "卷积运算"}
                        and "其实是____" in row["prompt"] and "correlation" not in row["prompt"]
                        and "而不是卷积运算" in row["sentence"] and "cross‐correlation" in row["sentence"])
                    checks["no_chapter_recall"] = all("回想一下" not in q["prompt"]
                        and "互相关和卷积" not in q["options"] for q in questions)
                role = case["role"]
                if case.get("padding_problem_required", case["title"] == "6.3 填充和步幅"):
                    role = "regression_after_preserved_first_observation"
                    observed = {q["concept"]: q for q in questions}
                    checks["observed_inline_glosses_removed"] = all(
                        term in observed and alias in observed[term]["sentence"] and alias not in observed[term]["prompt"]
                        for term, alias in (("填充", "padding"), ("步幅", "stride")))
                    padding = observed.get("填充", {})
                    checks["padding_problem_and_single_blank"] = (
                        padding.get("page") == 246 and padding["prompt"].count("____") == 1
                        and "边缘像素" in padding["prompt"] and "累积丢失" in padding["prompt"]
                        and "通常填充元素是0" in padding["sentence"])
                audit = []
                for candidate in extract_candidates(chunks):
                    prompt = candidate_prompt(candidate)
                    audit.append({**candidate.evidence(), "aliases": candidate.aliases, "synonyms": candidate.synonyms,
                                  "origin": candidate.origin, "option_class": candidate.kind,
                                  "relation_group": candidate.relation_group, "prompt": prompt,
                                  "stem_rejection": prompt_rejection(candidate, prompt),
                                  "selected": any(q["concept"] == candidate.concept and q["sentence"] == candidate.sentence
                                                  for q in questions)})
                report["chapters"].append({"title": case["title"], "role": role, "recognized_pages": case["recognized_pages"],
                    "status_code": response.status_code, "generated_count": len(questions),
                    "note": payload.get("note", payload.get("detail")), "questions": questions, "candidate_audit": audit,
                    "checks": checks, "passed": all(checks.values())})
    report["passed"] = all(c["passed"] for c in report["chapters"])
    report["all_chapters_produce_practice"] = all(c["generated_count"] > 0 for c in report["chapters"])
    report["teaching_acceptance"] = "not_established"
    report["limitations"] = ["Structural checks do not establish question value or semantic uniqueness.",
                              "No minimum quiz yield is assumed; empty output is reported explicitly.",
                              "Chapter scope depends on extracted headings and requires source review."]
    output = root / (args.output or "eval/reports/quiz-transfer-v5.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
