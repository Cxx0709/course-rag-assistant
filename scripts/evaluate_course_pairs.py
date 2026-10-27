"""Pinned, offline two-textbook extraction checks; not semantic acceptance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from course_rag_assistant.alignment import aligned_columns
from course_rag_assistant.compare import collect_excerpts, compare_excerpts
from course_rag_assistant.ingest import extract_pdf
from course_rag_assistant.prerequisites import PrerequisiteGraph
from course_rag_assistant.retrieval import HybridRetriever
from scripts.evaluate_course_features import digest, grounded


def evaluate(manifest_path, *, root=Path("."), repeats=2):
    spec = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources, cases = spec["sources"], spec["cases"]
    mode = spec.get("mode", "literal")
    if mode not in {"literal", "alignment"}:
        raise ValueError("Unsupported comparison evaluation mode")
    identities = [(c["concept"], c.get("sense")) for c in cases]
    if (repeats < 2 or len(sources) != 2 or sources[0]["path"] == sources[1]["path"]
            or not cases or len(set(identities)) != len(cases)):
        raise ValueError("Require two distinct sources, unique nonempty cases and at least two repeats")
    for case in cases:
        if len(case["expected_found"]) != 2 or len(case["reference_pages"]) != 2:
            raise ValueError("Each case requires expectations and reference pages for both sources")
    for source in sources:
        path = root / source["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Pinned source unavailable: {source['path']}")
        if digest(path) != source["sha256"]:
            raise ValueError(f"Source fingerprint changed: {source['path']}")
    runs = []
    for _ in range(repeats):
        corpora = [extract_pdf(root / source["path"]) for source in sources]
        retrievers = [HybridRetriever(chunks) for chunks in corpora]
        details = []
        for case in cases:
            alignment = None
            if mode == "alignment":
                result = aligned_columns(case["concept"], retrievers, sense=case.get("sense"))
                columns = result["columns"]
                alignment = result["alignment"]
                rows = [] if result.get("needs_disambiguation") else compare_excerpts(columns, comparable=result["comparable"])
            else:
                columns = [collect_excerpts(case["concept"], r) for r in retrievers]
                rows = compare_excerpts(columns)
            checks = {
                "found": [bool(c) for c in columns] == case["expected_found"],
                "reference_pages": all(
                    bool(set(refs) & {e["page"] for e in column}) if expected else not column
                    for column, expected, refs in zip(columns, case["expected_found"], case["reference_pages"])
                ),
                "exact_source_support": all(grounded(e, corpora[i]) for i, col in enumerate(columns) for e in col),
                "literal_highlights": all("".join(p["text"] for p in e["parts"]) == e["text"]
                                          for row in rows for cell in row["cells"] for e in cell if "parts" in e),
            }
            if alignment is not None:
                checks["alignment_decision"] = alignment["status"] == case["expected_alignment"]
                checks["sense"] = alignment.get("sense") == case.get("sense")
                if not result["comparable"]:
                    checks["no_unaligned_highlights"] = not any("parts" in e for row in rows for cell in row["cells"] for e in cell)
            details.append({"concept": case["concept"], "kind": case["kind"],
                            "passed": all(checks.values()), "checks": checks,
                            "found": [bool(c) for c in columns],
                            "facets": {r["key"]: r["status"] for r in rows},
                            "evidence": [[{"page": e["page"], "section": e["section"],
                                           "text_sha256": hashlib.sha256(e["text"].encode()).hexdigest()}
                                          for e in col] for col in columns],
                            "review_note": case["review_note"]})
            if alignment is not None:
                details[-1].update(sense=case.get("sense"), alignment=alignment)
        runs.append({"chunks": [len(c) for c in corpora],
                     "prerequisite_roots": [len(PrerequisiteGraph(c).edges) for c in corpora],
                     "cases": len(details), "passed": sum(d["passed"] for d in details), "details": details})
    checks = {"stable": all(r == runs[0] for r in runs[1:]),
              "extraction_contracts": runs[0]["cases"] == runs[0]["passed"]}
    return {"purpose": spec["purpose"], "manifest_sha256": digest(manifest_path),
            "sources": sources, "repeats": repeats, "checks": checks,
            "regression_passed": all(checks.values()), "runs": runs,
            "semantic_acceptance": "pending_independent_review",
            "limitations": spec["limitations"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("eval/course-pairs-v1.json"))
    parser.add_argument("--output", type=Path, default=Path("eval/reports/course-pairs-v1.json"))
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    report = evaluate(args.manifest, repeats=args.repeats)
    report["code_sha256"] = {str(p): digest(p) for p in
                             [Path(__file__), Path("scripts/evaluate_course_features.py"),
                              *sorted(Path("src/course_rag_assistant").rglob("*.py"))]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"regression_passed": report["regression_passed"], "checks": report["checks"],
                      "cases": report["runs"][0]["cases"], "passed": report["runs"][0]["passed"],
                      "semantic_acceptance": report["semantic_acceptance"]}, ensure_ascii=False, indent=2))
    if not report["regression_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
