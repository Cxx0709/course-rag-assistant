import json
from pathlib import Path

import pytest

from course_rag_assistant.ingest import Chunk
from scripts.evaluate_course_features import (
    diagnose_corpus,
    digest,
    evaluate,
    evaluate_cases,
    grounded,
)

MANIFEST = Path(__file__).resolve().parents[1] / "eval/course-features-v1.json"


def test_public_contract_cases_pass_but_mismatched_expectations_fail():
    cases = json.loads(MANIFEST.read_text(encoding="utf-8"))["cases"]
    result = evaluate_cases(cases)
    assert result["passed"] == result["cases"] == 10
    cases[0]["expected_found"] = [False, False]
    changed = evaluate_cases(cases)
    assert changed["passed"] == 9
    assert not changed["details"][0]["checks"]["found"]


def test_empty_or_duplicate_cases_cannot_claim_success():
    with pytest.raises(ValueError):
        evaluate_cases([])
    with pytest.raises(ValueError):
        evaluate_cases([{"id": "duplicate"}, {"id": "duplicate"}])


def test_grounding_requires_correct_document_page_and_literal_span():
    c = Chunk("原文第一行\n接着是第二行。", "a.pdf", 2, "概念")
    evidence = {"doc": c.doc, "page": c.page, "section": c.section, "text": c.text}
    assert grounded(evidence, [c])
    for update in [{"page": 3}, {"doc": "b.pdf"}, {"section": "其他"}, {"text": c.text.replace("\n", "")}]:
        assert not grounded({**evidence, **update}, [c])


def test_corpus_drift_fails_and_missing_corpus_is_explicit(tmp_path):
    path = tmp_path / "a.md"
    path.write_text("# 过拟合\n\n过拟合是指训练误差远小于验证误差。", encoding="utf-8")
    spec = {"path": ".", "inputs_sha256": {"a.md": digest(path)},
            "probes": [{"concept": "过拟合", "expected_found": True, "reference_pages": [0]}]}
    result = diagnose_corpus(spec, tmp_path)
    assert result["documents"] == result["positive_reference_hits"] == 1
    assert result["source_support_rate"] == 1
    assert result["prerequisite_roots"] == 0
    assert result["cross_textbook_validation"].startswith("not_run")
    assert "text" not in result["details"][0]["evidence"][0]
    assert diagnose_corpus({**spec, "path": "missing"}, tmp_path)["status"] == "unavailable"
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        diagnose_corpus(spec, tmp_path)


def test_contract_success_never_claims_real_course_acceptance(tmp_path):
    result = evaluate(MANIFEST, root=tmp_path)
    assert result["regression_passed"]
    assert result["checks"]["stable"]
    assert result["runs"][0]["real_corpus"]["status"] == "unavailable"
    assert result["real_course_acceptance"]["status"] == "incomplete"
    assert result["real_course_acceptance"]["blockers"]
    with pytest.raises(ValueError):
        evaluate(MANIFEST, root=tmp_path, repeats=1)
