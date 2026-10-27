import json

import pytest

from course_rag_assistant.ingest import extract_markdown
from scripts import evaluate_course_pairs as pairs


def fixture_manifest(tmp_path, monkeypatch):
    sources = []
    for name in ("a.md", "b.md"):
        path = tmp_path / name
        path.write_text("# 过拟合\n\n过拟合是指训练误差远小于验证误差。", encoding="utf-8")
        sources.append({"path": name, "sha256": pairs.digest(path)})
    spec = {"purpose": "test", "sources": sources, "limitations": ["review pending"],
            "cases": [{"concept": "过拟合", "kind": "shared_concept", "expected_found": [True, True],
                       "reference_pages": [[0], [0]], "review_note": "review pending"}]}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(pairs, "extract_pdf", extract_markdown)
    return manifest, spec


def test_pair_success_preserves_pending_semantics_and_hash_only_evidence(tmp_path, monkeypatch):
    manifest, _ = fixture_manifest(tmp_path, monkeypatch)
    result = pairs.evaluate(manifest, root=tmp_path)
    assert result["regression_passed"]
    assert result["semantic_acceptance"] == "pending_independent_review"
    assert result["runs"][0]["prerequisite_roots"] == [0, 0]
    evidence = result["runs"][0]["details"][0]["evidence"]
    assert evidence[0] and evidence[1]
    assert "text" not in evidence[0][0]


def test_pair_wrong_reference_page_fails(tmp_path, monkeypatch):
    manifest, spec = fixture_manifest(tmp_path, monkeypatch)
    spec["cases"][0]["reference_pages"][1] = [99]
    manifest.write_text(json.dumps(spec), encoding="utf-8")
    assert not pairs.evaluate(manifest, root=tmp_path)["regression_passed"]


def test_pair_missing_or_changed_source_cannot_pass(tmp_path, monkeypatch):
    manifest, _ = fixture_manifest(tmp_path, monkeypatch)
    (tmp_path / "a.md").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        pairs.evaluate(manifest, root=tmp_path)
    (tmp_path / "a.md").unlink()
    with pytest.raises(FileNotFoundError):
        pairs.evaluate(manifest, root=tmp_path)


def test_pair_truncated_expectations_are_rejected(tmp_path, monkeypatch):
    manifest, spec = fixture_manifest(tmp_path, monkeypatch)
    spec["cases"][0]["reference_pages"] = []
    manifest.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="both sources"):
        pairs.evaluate(manifest, root=tmp_path)


def test_alignment_mode_scores_decisions_instead_of_only_source_presence(tmp_path, monkeypatch):
    manifest, spec = fixture_manifest(tmp_path, monkeypatch)
    spec["mode"] = "alignment"
    spec["cases"][0]["expected_alignment"] = "role_correspondence"
    manifest.write_text(json.dumps(spec), encoding="utf-8")
    report = pairs.evaluate(manifest, root=tmp_path)
    assert not report["regression_passed"]
    assert not report["runs"][0]["details"][0]["checks"]["alignment_decision"]
    spec["cases"][0]["expected_alignment"] = "literal_name"
    manifest.write_text(json.dumps(spec), encoding="utf-8")
    assert pairs.evaluate(manifest, root=tmp_path)["regression_passed"]


def test_same_term_different_sense_cases_are_distinct_and_pending_is_not_absence(tmp_path, monkeypatch):
    manifest, spec = fixture_manifest(tmp_path, monkeypatch)
    spec["mode"] = "alignment"
    spec["cases"] = [{"concept": "方差", "kind": "sense", "expected_found": [False, False],
                      "reference_pages": [[], []], "review_note": "test", "expected_alignment": "needs_disambiguation"},
                     {"concept": "方差", "sense": "random_variable_variance", "kind": "sense",
                      "expected_found": [False, False], "reference_pages": [[], []],
                      "review_note": "test", "expected_alignment": "sense_selected"}]
    manifest.write_text(json.dumps(spec), encoding="utf-8")
    report = pairs.evaluate(manifest, root=tmp_path)
    assert report["regression_passed"]
    assert report["runs"][0]["details"][0]["facets"] == {}
    assert report["semantic_acceptance"] == "pending_independent_review"
