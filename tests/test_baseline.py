import hashlib
from pathlib import Path

import pytest

from scripts.evaluate_baseline import (
    check_inputs,
    evaluate_integrity,
    stable_predictions,
)


def test_frozen_input_drift_fails_before_evaluation(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_bytes(b"frozen labels")
    manifest = {"inputs_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}}
    check_inputs(manifest)
    path.write_bytes(b"changed labels")
    with pytest.raises(ValueError, match="Frozen baseline input changed"):
        check_inputs(manifest)


def test_repeat_comparison_ignores_timing_but_detects_answer_drift():
    row = {"id": "1", "answer": "source", "abstention_reason": None,
           "selected_sentence_ids": [1], "answer_seconds": 1}
    expected = stable_predictions({"details": [row]})
    assert stable_predictions({"details": [{**row, "answer_seconds": 99}]}) == expected
    assert stable_predictions({"details": [{**row, "answer": "different"}]}) != expected


def test_fixed_integrity_cases_pass():
    result = evaluate_integrity(Path(__file__).resolve().parents[1] / "eval/evidence-integrity-v12.jsonl")
    assert result["passed"] == result["cases"] == 12
