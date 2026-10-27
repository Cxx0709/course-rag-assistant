"""Repeat source-grounding contracts and diagnose a pinned local course corpus.

The public regression gate is distinct from real-course readiness. A single
textbook and unannotated prerequisites cannot validate a two-textbook workflow.
Run from the repository root; no model or external service is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from course_rag_assistant.compare import collect_excerpts, compare_excerpts
from course_rag_assistant.ingest import (
    Chunk,
    extract_markdown,
    extract_pdf,
    extract_pptx,
)
from course_rag_assistant.prerequisites import PrerequisiteGraph
from course_rag_assistant.retrieval import HybridRetriever


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grounded(excerpt, chunks):
    return any((c.doc, c.page, c.section) == (excerpt["doc"], excerpt["page"], excerpt["section"])
               and excerpt["text"] in c.text for c in chunks)


def evaluate_cases(cases):
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Cases must be nonempty and have unique IDs")
    results = []
    for case in cases:
        if case["kind"] == "compare":
            inputs = [[Chunk(**c) for c in case[side]] for side in ("left", "right")]
            columns = [collect_excerpts(case["concept"], HybridRetriever(chunks)) for chunks in inputs]
            rows = compare_excerpts(columns)
            statuses = {r["key"]: r["status"] for r in rows}
            checks = {"found": [bool(c) for c in columns] == case["expected_found"],
                      "facets": all(statuses[k] == v for k, v in case["expected_facets"].items()),
                      "source_grounding": all(grounded(e, inputs[i]) for i, c in enumerate(columns) for e in c),
                      "literal_highlights": all("".join(p["text"] for p in e["parts"]) == e["text"]
                                                for r in rows for cell in r["cells"] for e in cell if "parts" in e)}
            observed = {"found": [bool(c) for c in columns], "facets": statuses}
        elif case["kind"] == "prerequisite":
            chunks = [Chunk(**c) for c in case["chunks"]]
            plan = PrerequisiteGraph(chunks).plan(case["concept"])
            observed = {"status": plan["status"], "order": [s["concept"] for s in plan["steps"]],
                        "missing": [s["concept"] for s in plan["steps"] if not s["sources"]]}
            checks = {key: observed[key] == case[f"expected_{key}"] for key in observed}
            checks["source_grounding"] = all(grounded(s, chunks) for row in [*plan["steps"], *plan["relations"]]
                                             for s in row["sources"])
        else:
            raise ValueError(f"Unknown case kind: {case['kind']}")
        results.append({"id": case["id"], "passed": all(checks.values()), "checks": checks, "observed": observed})
    return {"cases": len(results), "passed": sum(r["passed"] for r in results), "details": results}


def diagnose_corpus(spec, root):
    folder = root / spec["path"]
    if not folder.is_dir() or any(not (folder / name).is_file() for name in spec["inputs_sha256"]):
        return {"status": "unavailable", "note": "Pinned local corpus is not installed; no real-course claims can be made."}
    for name, expected in spec["inputs_sha256"].items():
        if digest(folder / name) != expected:
            raise ValueError(f"Local corpus fingerprint changed: {name}")
    # Restrict input to pinned files; new private files must not silently change
    # the evaluation population or be included in the saved report.
    parsers = {".pdf": extract_pdf, ".md": extract_markdown, ".pptx": extract_pptx}
    chunks = [c for name in sorted(spec["inputs_sha256"])
              for c in parsers[Path(name).suffix.lower()](folder / name)]
    retriever = HybridRetriever(chunks)
    graph = PrerequisiteGraph(chunks)
    details = []
    for probe in spec["probes"]:
        excerpts = collect_excerpts(probe["concept"], retriever)
        pages = sorted({e["page"] for e in excerpts})
        hits_reference = bool(set(pages) & set(probe["reference_pages"]))
        details.append({"concept": probe["concept"], "expected_found": probe["expected_found"],
                        "found": bool(excerpts), "reference_page_hit": hits_reference,
                        "passed": hits_reference if probe["expected_found"] else not excerpts,
                        "pages": pages, "excerpt_count": len(excerpts),
                        "grounded_count": sum(grounded(e, chunks) for e in excerpts),
                        "evidence": [{"page": e["page"], "text_sha256": hashlib.sha256(e["text"].encode()).hexdigest()}
                                     for e in excerpts]})
    positives = [r for r in details if r["expected_found"]]
    negatives = [r for r in details if not r["expected_found"]]
    total = sum(r["excerpt_count"] for r in details)
    ready_plans = sum(graph.plan(p["concept"])["status"] == "ready" for p in spec["probes"])
    return {"status": "diagnosed", "documents": len({c.doc for c in chunks}), "chunks": len(chunks),
            "positive_probes": len(positives), "positive_reference_hits": sum(r["passed"] for r in positives),
            "reference_page_hit_rate": sum(r["passed"] for r in positives) / len(positives) if positives else None,
            "negative_probes": len(negatives), "negative_rejections": sum(r["passed"] for r in negatives),
            "source_support_rate": sum(r["grounded_count"] for r in details) / total if total else None,
            "prerequisite_roots": len(graph.edges), "probe_learning_paths": ready_plans,
            "cross_textbook_validation": "not_run_requires_two_real_sources",
            "semantic_comparison_validation": "not_run_requires_independent_review",
            "details": details}


def evaluate(manifest_path, *, root=Path("."), repeats=2):
    if repeats < 2:
        raise ValueError("At least two repeats are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = [{"contracts": evaluate_cases(manifest["cases"]),
             "real_corpus": diagnose_corpus(manifest["real_corpus"], root)} for _ in range(repeats)]
    checks = {"stable": all(r == runs[0] for r in runs[1:]),
              "contracts": runs[0]["contracts"]["passed"] == runs[0]["contracts"]["cases"]}
    real = runs[0]["real_corpus"]
    if real["status"] == "diagnosed":
        checks["real_source_grounding"] = real["source_support_rate"] == 1.0
    blockers = ["No independent semantic comparison review has been completed."]
    if real["status"] == "unavailable":
        blockers.append("Pinned real course corpus is unavailable.")
    else:
        if real["documents"] < 2:
            blockers.append("Only one real textbook is available; cross-textbook quality remains unvalidated.")
        if not real["probe_learning_paths"]:
            blockers.append("Real course materials have no recognized prerequisite declarations for the probes.")
        if real["positive_reference_hits"] < real["positive_probes"]:
            blockers.append("Some positive concept probes still miss their reference pages.")
    return {"purpose": manifest["purpose"], "manifest_sha256": digest(manifest_path), "repeats": repeats,
            "regression_passed": all(checks.values()), "checks": checks, "runs": runs,
            "real_course_acceptance": {"status": "incomplete", "blockers": blockers}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("eval/course-features-v1.json"))
    parser.add_argument("--output", type=Path, default=Path("eval/reports/course-features-v1.json"))
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    report = evaluate(args.manifest, repeats=args.repeats)
    report["code_sha256"] = {str(p): digest(p) for p in
                             [Path(__file__), *sorted(Path("src/course_rag_assistant").rglob("*.py"))]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"regression_passed": report["regression_passed"], "checks": report["checks"],
                      "real_course_acceptance": report["real_course_acceptance"]}, ensure_ascii=False, indent=2))
    if not report["regression_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
