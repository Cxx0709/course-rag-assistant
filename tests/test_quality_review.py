import json

import pytest

from scripts.export_quality_review import export_review, review_rows


def _report():
    row = {"id": "q1", "question": "训练数据", "answer": "回答",
           "raw_model_output": None, "generation_status": "offline_extractive",
           "abstention_reason": None, "unsupported_claims": [], "evidence": [],
           "expected_answer_points": ["训练数据"], "gold_pages": ["notes.pdf#p.1"]}
    return {"schema_version": 3, "completed": True, "benchmark": "development",
            "run_scope": "full_development_set", "results": {
                "hybrid": {"answerable": {"details": [row]},
                           "unanswerable": {"details": []}}}}


def test_export_is_pending_not_an_automatic_human_judgment():
    rows = list(review_rows(_report(), "hash"))
    assert len(rows) == 1
    assert rows[0]["report_sha256"] == "hash"
    assert rows[0]["review"]["status"] == "pending"
    assert rows[0]["review"]["answer_correct"] is None
    assert rows[0]["expected_abstention"] is False


def test_incomplete_run_cannot_be_exported_as_finished():
    report = _report()
    report["completed"] = False
    with pytest.raises(ValueError, match="completed"):
        list(review_rows(report, "hash"))


def test_repeat_export_preserves_human_review(tmp_path):
    source, target = tmp_path / "report.json", tmp_path / "review.jsonl"
    source.write_text(json.dumps(_report()), encoding="utf-8")
    assert export_review(source, target) == 1
    target.write_text("reviewer's work", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_review(source, target)
    assert target.read_text(encoding="utf-8") == "reviewer's work"
