"""Score a locked, assistant-annotated diagnostic corpus without changing quiz rules."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from course_rag_assistant.ingest import Chunk
from course_rag_assistant.quiz import make_questions
from course_rag_assistant.quiz_candidates import (
    candidate_prompt,
    extract_candidates,
    prompt_rejection,
)

DIMENSIONS = {"boundary", "identity", "context", "options"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(text):
    return re.sub(r"\s+", "", text)


def load_corpus(path, lock_path):
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if digest(path) != lock["dataset_sha256"]:
        raise ValueError("Corpus fingerprint changed: preserve v1 and publish a new labeled version.")
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data["cases"]
    if data["schema_version"] != 1 or not cases or len(cases) != lock["case_count"]:
        raise ValueError("Invalid corpus version or case count.")
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Case identifiers must be unique.")
    if {c["dimension"] for c in cases} != DIMENSIONS or set(lock["dimensions"]) != DIMENSIONS:
        raise ValueError("All four diagnostic dimensions are required.")
    if not any(c["dimension"] == "options" and c["expected"]["required_groups"] for c in cases):
        raise ValueError("A corpus must include positive question coverage requirements.")
    for case in cases:
        if not case["chunks"] or case["cohort"] not in {"textbook", "constructed"} or not case["rationale"]:
            raise ValueError(f"Invalid case: {case['id']}")
        for chunk in case["chunks"]:
            if not chunk["text"] or (chunk["source"] and chunk["source"] not in data["sources"]):
                raise ValueError(f"Missing source/text: {case['id']}")
    return data, lock


def verify_sources(data, root):
    import pymupdf

    checked = 0
    with ExitStack() as stack:
        pdfs = {}
        for key, source in data["sources"].items():
            path = root / source["path"]
            if digest(path) != source["sha256"]:
                raise ValueError(f"Source fingerprint changed: {key}")
            pdfs[key] = stack.enter_context(pymupdf.open(path))
        for case in data["cases"]:
            for chunk in case["chunks"]:
                if not chunk["source"]:
                    continue
                pdf = pdfs[chunk["source"]]
                if (not 1 <= chunk["page"] <= len(pdf)
                        or compact(chunk["text"]) not in compact(pdf[chunk["page"] - 1].get_text())):
                    raise ValueError(f"Source passage mismatch: {case['id']}, page {chunk['page']}")
                checked += 1
    return {"verified": True, "passage_occurrences_checked": checked}


def question_errors(case, question):
    expected = case["expected"]
    concept = question["concept"]
    errors = []
    allowed = expected["allowed_options"]
    if concept not in allowed:
        errors.append("unexpected_question")
    options = question["options"]
    index = question["correct_index"]
    if (type(index) is not int or not 0 <= index < len(options) or options[index] != concept
            or len(options) < 2 or len(set(options)) != len(options)):
        errors.append("invalid_answer_key_or_options")
    if any(o != concept and o not in allowed.get(concept, []) for o in options):
        errors.append("unapproved_distractor")
    if any(len(set(group) & set(options)) > 1 for group in expected.get("synonym_groups", [])):
        errors.append("synonym_options")
    prompt = question["prompt"]
    if prompt.count("____") != 1 or concept in prompt:
        errors.append("answer_hint_or_blank")
    if not all(s in prompt for s in expected.get("stem_required", {}).get(concept, [])):
        errors.append("missing_stem_context")
    if not question["sentence"] or not any(
        (question["doc"], question["page"], question["section"]) == (c["doc"], c["page"], c["section"])
        and compact(question["sentence"]) in compact(c["text"]) for c in case["chunks"]
    ):
        errors.append("source_mismatch")
    return errors


def score_case(case, candidates, questions):
    expected = case["expected"]
    dimension = case["dimension"]
    facts = {"candidate_count": len(candidates), "question_count": len(questions)}
    failures = []
    if dimension == "boundary":
        actual, gold = {c.concept for c in candidates}, set(expected["concepts"])
        facts.update(true_positive=len(actual & gold), false_positive=len(actual - gold),
                     false_negative=len(gold - actual), missing=sorted(gold - actual), extra=sorted(actual - gold))
        if actual != gold:
            failures.append("concept_boundary")
    elif dimension == "identity":
        checks = []
        families = []
        for left, right, equivalent in expected["links"]:
            if equivalent:
                family = {left, right}
                for known in families[:]:
                    if family & known:
                        family |= known
                        families.remove(known)
                families.append(family)
            observed = any((c.concept == left and right in c.synonyms)
                           or (c.concept == right and left in c.synonyms) for c in candidates)
            checks.append({"kind": "synonym", "names": [left, right], "expected": equivalent,
                           "actual": observed, "passed": equivalent == observed})
        for concept, aliases in expected["aliases"].items():
            observed = {a.casefold() for c in candidates if c.concept == concept for a in c.aliases}
            checks.append({"kind": "aliases", "concept": concept, "expected": sorted(a.casefold() for a in aliases),
                           "actual": sorted(observed), "passed": observed == {a.casefold() for a in aliases}})
        facts["checks"] = checks
        unexpected = {tuple(sorted((c.concept, synonym))) for c in candidates for synonym in c.synonyms
                      if not any({c.concept, synonym} <= family for family in families)}
        facts["unexpected_synonyms"] = sorted(unexpected)
        if not checks or unexpected or not all(c["passed"] for c in checks):
            failures.append("alias_identity")
    elif dimension == "context":
        matched = [c for c in candidates if c.concept == expected["concept"]]
        stems = []
        for candidate in matched:
            prompt = candidate_prompt(candidate)
            reason = prompt_rejection(candidate, prompt)
            constraints = (all(s in prompt for s in expected["required"])
                           and not any(s in prompt for s in expected["forbidden"]))
            stems.append({"prompt": prompt, "rejection": reason, "constraints_met": constraints})
        accepted = [s for s in stems if s["rejection"] is None]
        passed = (bool(accepted) and all(s["constraints_met"] for s in accepted)) if expected["accept"] else not accepted
        facts.update(stems=stems, expected_accept=expected["accept"], accepted=len(accepted),
                     unsafe_accept=int(not expected["accept"] and bool(accepted)),
                     blocked_at="not_extracted" if not matched else "stem_gate" if not accepted else None)
        if not passed:
            failures.append("unsafe_context_accept" if not expected["accept"] else
                            "context_not_extracted" if not matched else "context_rejected_or_incomplete")
    else:
        valid = []
        diagnostics = []
        used_identities = set()
        for question in questions:
            errors = question_errors(case, question)
            names = next((set(g) for g in expected.get("synonym_groups", []) if question["concept"] in g),
                         {question["concept"]})
            if names & used_identities:
                errors.append("duplicate_question_identity")
            used_identities.update(names)
            if not errors:
                valid.append(question["concept"])
            diagnostics.append({"concept": question["concept"], "errors": errors})
        missing = [g for g in expected["required_groups"] if not set(g) & set(valid)]
        invalid = sum(bool(d["errors"]) for d in diagnostics)
        facts.update(required=len(expected["required_groups"]), covered=len(expected["required_groups"]) - len(missing),
                     missing=missing, invalid_questions=invalid, questions=diagnostics)
        for group in missing:
            matched = [c for c in candidates if c.concept in group]
            failures.append("not_extracted" if not matched else
                            "stem_rejected" if not any(prompt_rejection(c, candidate_prompt(c)) is None for c in matched)
                            else "no_compatible_options_or_selection")
        if invalid:
            failures.append("invalid_questions")
    return {"id": case["id"], "dimension": dimension, "cohort": case["cohort"],
            "passed": not failures, "failures": sorted(set(failures)), "facts": facts}


def summarize(results, lock):
    dimensions = {}
    for dimension in sorted(DIMENSIONS):
        rows = [r for r in results if r["dimension"] == dimension]
        dimensions[dimension] = {"cases": len(rows), "passed": sum(r["passed"] for r in rows),
                                 "pass_rate": sum(r["passed"] for r in rows) / len(rows) if rows else None}
    contexts = [r for r in results if r["dimension"] == "context"]
    positive = [r for r in contexts if r["facts"]["expected_accept"]]
    options = [r for r in results if r["dimension"] == "options"]
    required = sum(r["facts"]["required"] for r in options)
    coverage = sum(r["facts"]["covered"] for r in options)
    unsafe = sum(r["facts"]["unsafe_accept"] for r in contexts)
    invalid = sum(r["facts"]["invalid_questions"] for r in options)
    gate = lock["gates"]
    passed = (all(d["pass_rate"] is not None and d["pass_rate"] >= gate["minimum_case_pass_rate"] for d in dimensions.values())
              and unsafe <= gate["maximum_unsafe_context_accepts"] and invalid <= gate["maximum_invalid_questions"]
              and required > 0 and coverage / required >= gate["minimum_case_pass_rate"])
    return {"passed": passed, "dimensions": dimensions, "unsafe_context_accepts": unsafe,
            "positive_context_cases": len(positive), "positive_context_passed": sum(r["passed"] for r in positive),
            "required_question_groups": required, "covered_question_groups": coverage,
            "question_group_coverage": coverage / required if required else None, "invalid_questions": invalid,
            "failure_cases_by_category": dict(Counter(f for r in results for f in r["failures"]))}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="eval/reports/quiz-benchmark-v1-baseline.json")
    parser.add_argument("--verify-sources", action="store_true")
    args = parser.parse_args()
    path, lock_path = root / "eval/quiz-benchmark-v1.json", root / "eval/quiz-benchmark-v1.lock.json"
    data, lock = load_corpus(path, lock_path)
    output = root / args.output
    if output.exists():
        parser.error("Output already exists; choose a new path to preserve the baseline.")
    source_check = verify_sources(data, root) if args.verify_sources else {"verified": False, "reason": "not_requested"}
    results = []
    for case in data["cases"]:
        chunks = [Chunk(**{k: c[k] for k in ("text", "doc", "page", "section")}) for c in case["chunks"]]
        results.append(score_case(case, extract_candidates(chunks), make_questions(chunks, count=20)))
    summary = summarize(results, lock)
    cohorts = {name: {"cases": sum(r["cohort"] == name for r in results),
                      "passed": sum(r["cohort"] == name and r["passed"] for r in results)}
               for name in ("textbook", "constructed")}
    report = {"benchmark": data["name"], "dataset_sha256": digest(path), "lock_sha256": digest(lock_path),
              "evaluator_sha256": digest(Path(__file__)), "source_verification": source_check,
              "code_sha256": {p.as_posix(): digest(root / p) for p in
                              [Path(f"src/course_rag_assistant/{n}.py") for n in
                               ("quiz", "quiz_candidates", "quiz_content", "quiz_relations", "quiz_terms")]},
              "annotation_policy": data["annotation_policy"], "cohorts": cohorts,
              "summary": summary, "cases": results,
              "limitations": ["Retrospective purposive corpus; not an unseen or representative sample.",
                              "Assistant labels and lexical assertions do not replace independent teacher review.",
                              "Candidate absence and context rejection are reported separately."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": summary, "cohorts": cohorts, "output": str(output)}, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
