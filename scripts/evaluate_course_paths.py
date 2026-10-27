"""Validate an authored reading-plan draft and its local PDF references.

This checks graph order and source locations, not pedagogical correctness or
whether the textbooks explicitly declare these prerequisite relationships.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from course_rag_assistant.ingest import extract_markdown, extract_pdf
from course_rag_assistant.prerequisites import PrerequisiteGraph
from scripts.evaluate_course_features import digest, grounded


def evaluate(manifest_path, *, root=Path(".")):
    spec = json.loads(manifest_path.read_text(encoding="utf-8"))
    note = root / spec["note"]["path"]
    if digest(note) != spec["note"]["sha256"]:
        raise ValueError("Draft fingerprint changed")
    pages = {}
    for source in spec["sources"]:
        path = root / source["path"]
        if digest(path) != source["sha256"]:
            raise ValueError("Source fingerprint changed")
        pages[path.resolve()] = {c.page for c in extract_pdf(path)}
    chunks = extract_markdown(note)
    graph = PrerequisiteGraph(chunks)
    refs = re.findall(r"\]\(([^)]+\.pdf)#page=(\d+)\)", note.read_text(encoding="utf-8"))
    references = [{"path": target, "page": int(page),
                   "valid": int(page) in pages.get((note.parent / target).resolve(), set())}
                  for target, page in refs]
    details = []
    if not spec["plans"]:
        raise ValueError("Require at least one expected learning plan")
    for expected in spec["plans"]:
        plan = graph.plan(expected["concept"])
        checks = {"ready": plan["status"] == "ready",
                  "order": [s["concept"] for s in plan["steps"]] == expected["order"],
                  "lessons_present": all(s["sources"] for s in plan["steps"]),
                  "draft_grounding": all(grounded(e, chunks) for row in [*plan["steps"], *plan["relations"]]
                                          for e in row["sources"])}
        details.append({"concept": expected["concept"], "checks": checks, "passed": all(checks.values()),
                        "order": [s["concept"] for s in plan["steps"]],
                        "relation_count": len(plan["relations"])})
    checks = {"references": bool(references) and all(r["valid"] for r in references),
              "plans": all(p["passed"] for p in details)}
    return {"purpose": "Structural validation of assistant-authored draft; not textbook-declared prerequisites",
            "manifest_sha256": digest(manifest_path), "note": spec["note"],
            "regression_passed": all(checks.values()), "checks": checks,
            "references": references, "plans": details,
            "review_status": "pending_human_review", "default_library_modified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("eval/course-paths-draft-v1.json"))
    parser.add_argument("--output", type=Path, default=Path("eval/reports/course-paths-draft-v1.json"))
    args = parser.parse_args()
    report = evaluate(args.manifest)
    report["code_sha256"] = {str(p): digest(p) for p in
                             [Path(__file__), Path("scripts/evaluate_course_features.py"),
                              Path("src/course_rag_assistant/ingest.py"),
                              Path("src/course_rag_assistant/prerequisites.py")]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("regression_passed", "checks", "review_status")}, indent=2))
    if not report["regression_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
