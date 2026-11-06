import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import evaluate_quiz_benchmark as benchmark


@pytest.fixture(autouse=True)
def independent_predictions(monkeypatch):
    # Scorer tests control model outputs; future quiz improvements must not
    # silently change the expected scoring behavior.
    monkeypatch.setattr(benchmark, "candidate_prompt", lambda c: c.prompt)
    monkeypatch.setattr(benchmark, "prompt_rejection", lambda c, _: c.rejection)


def candidate(concept="甲概念", prompt="独立说明____。", rejection=None, synonyms=(), aliases=()):
    return SimpleNamespace(concept=concept, prompt=prompt, rejection=rejection, synonyms=synonyms, aliases=aliases)


def case(dimension, expected):
    return {"id": "sample", "dimension": dimension, "cohort": "constructed", "expected": expected,
            "chunks": [{"text": "甲概念是指一种特定的课程知识。乙概念是指另一种知识。", "doc": "a.md", "page": 0, "section": "课程"}]}


def options_case():
    return case("options", {"required_groups": [["甲概念"]],
                            "allowed_options": {"甲概念": ["乙概念"]}, "stem_required": {}, "synonym_groups": []})


def question():
    return {"concept": "甲概念", "sentence": "甲概念是指一种特定的课程知识。", "doc": "a.md", "page": 0,
            "section": "课程", "prompt": "____是指一种特定的课程知识。", "options": ["甲概念", "乙概念"], "correct_index": 0}


def test_correct_question_is_covered_and_literal_source_is_checked():
    result = benchmark.score_case(options_case(), [candidate()], [question()])
    assert result["passed"] and result["facts"]["covered"] == 1
    assert result["facts"]["invalid_questions"] == 0


@pytest.mark.parametrize("field,value,error", [
    ("doc", "foreign.md", "source_mismatch"),
    ("page", 99, "source_mismatch"),
    ("section", "其他课程", "source_mismatch"),
    ("sentence", "甲概念是指来源没有说过的内容。", "source_mismatch"),
    ("sentence", "", "source_mismatch"),
    ("correct_index", True, "invalid_answer_key_or_options"),
    ("correct_index", 1, "invalid_answer_key_or_options"),
    ("options", ["甲概念", "未知词"], "unapproved_distractor"),
    ("options", ["甲概念", "甲概念"], "invalid_answer_key_or_options"),
    ("prompt", "甲概念是指____。", "answer_hint_or_blank"),
])
def test_invalid_questions_cannot_cover_a_positive_requirement(field, value, error):
    q = {**question(), field: value}
    result = benchmark.score_case(options_case(), [candidate()], [q])
    assert not result["passed"] and result["facts"]["covered"] == 0
    assert error in result["facts"]["questions"][0]["errors"]


def test_missing_context_is_not_hidden_by_other_correct_question_fields():
    item = options_case()
    item["expected"]["stem_required"] = {"甲概念": ["在同分布条件下"]}
    assert "missing_stem_context" in benchmark.question_errors(item, question())


def test_boundary_truncation_is_both_false_positive_and_false_negative():
    result = benchmark.score_case(case("boundary", {"concepts": ["条件独立"]}), [candidate("独立")], [])
    assert not result["passed"]
    assert result["facts"]["false_positive"] == result["facts"]["false_negative"] == 1


def test_missing_synonym_or_extra_alias_does_not_pass():
    item = case("identity", {"links": [["甲概念", "乙概念", True]], "aliases": {"甲概念": ["alpha"]}})
    assert not benchmark.score_case(item, [], [])["passed"]
    assert not benchmark.score_case(item, [candidate(aliases=("alpha",))], [])["passed"]
    assert not benchmark.score_case(item, [candidate(synonyms=("乙概念",), aliases=("alpha", "beta"))], [])["passed"]
    assert benchmark.score_case(item, [candidate(synonyms=("乙概念",), aliases=("ALPHA",))], [])["passed"]
    assert not benchmark.score_case(item, [candidate(synonyms=("乙概念", "幻觉名称"), aliases=("alpha",))], [])["passed"]


def test_synonym_distractors_and_duplicate_question_identities_are_invalid():
    item = options_case()
    item["expected"]["synonym_groups"] = [["甲概念", "乙概念"]]
    assert "synonym_options" in benchmark.question_errors(item, question())
    result = benchmark.score_case(options_case(), [candidate()], [question(), question()])
    assert result["facts"]["invalid_questions"] == 1
    assert "duplicate_question_identity" in result["facts"]["questions"][1]["errors"]


def test_context_acceptance_is_measured_even_without_generated_questions():
    item = case("context", {"concept": "甲概念", "accept": False, "required": [], "forbidden": []})
    result = benchmark.score_case(item, [candidate()], [])
    assert not result["passed"] and result["facts"]["unsafe_accept"] == 1
    absent = benchmark.score_case(item, [], [])
    assert absent["passed"] and absent["facts"]["blocked_at"] == "not_extracted"
    blocked = benchmark.score_case(item, [candidate(rejection="missing_context")], [])
    assert blocked["passed"] and blocked["facts"]["blocked_at"] == "stem_gate"


def test_context_constraints_and_missing_positive_candidate_fail():
    item = case("context", {"concept": "甲概念", "accept": True, "required": ["必要条件"], "forbidden": ["英文提示"]})
    for predictions in ([], [candidate()], [candidate(prompt="必要条件与英文提示____。")]):
        assert not benchmark.score_case(item, predictions, [])["passed"]
    assert benchmark.score_case(item, [candidate(prompt="必要条件下____。")], [])["passed"]


@pytest.mark.parametrize("predictions,stage", [
    ([], "not_extracted"),
    ([candidate(rejection="unsafe")], "stem_rejected"),
    ([candidate()], "no_compatible_options_or_selection"),
])
def test_missing_question_is_attributed_to_the_right_stage(predictions, stage):
    result = benchmark.score_case(options_case(), predictions, [])
    assert result["facts"]["covered"] == 0 and stage in result["failures"]


def test_all_empty_outputs_cannot_pass_even_when_abstention_cases_pass():
    items = [case("boundary", {"concepts": ["甲概念"]}),
             case("identity", {"links": [["甲概念", "乙概念", True]], "aliases": {}}),
             case("context", {"concept": "甲概念", "accept": False, "required": [], "forbidden": []}), options_case()]
    results = [benchmark.score_case(item, [], []) for item in items]
    lock = {"gates": {"minimum_case_pass_rate": .9, "maximum_unsafe_context_accepts": 0, "maximum_invalid_questions": 0}}
    summary = benchmark.summarize(results, lock)
    assert not summary["passed"] and summary["question_group_coverage"] == 0
    assert summary["dimensions"]["context"]["pass_rate"] == 1
    assert summary["dimensions"]["options"]["pass_rate"] == 0


def test_case_weight_does_not_depend_on_number_of_generated_questions():
    result = benchmark.score_case(options_case(), [candidate()], [question()])
    # An enumeration remains one case even if it has several coverage groups.
    result["facts"].update(required=3, covered=3)
    lock = {"gates": {"minimum_case_pass_rate": .9, "maximum_unsafe_context_accepts": 0, "maximum_invalid_questions": 0}}
    summary = benchmark.summarize([result], lock)
    assert summary["dimensions"]["options"]["cases"] == 1
    assert summary["covered_question_groups"] == 3
    assert not summary["passed"]  # Missing dimensions never count as passing.


def test_committed_corpus_lock_and_schema_are_valid():
    root = Path(__file__).parents[1]
    data, lock = benchmark.load_corpus(root / "eval/quiz-benchmark-v1.json", root / "eval/quiz-benchmark-v1.lock.json")
    assert len(data["cases"]) == lock["case_count"] == 34
    assert data["annotation_policy"]["independent_review"] is False


def test_changed_labels_are_rejected_and_duplicate_ids_are_invalid(tmp_path):
    root = Path(__file__).parents[1]
    data = json.loads((root / "eval/quiz-benchmark-v1.json").read_text(encoding="utf-8"))
    lock = json.loads((root / "eval/quiz-benchmark-v1.lock.json").read_text(encoding="utf-8"))
    modified = deepcopy(data)
    modified["cases"][0]["expected"]["concepts"] = []
    path, lock_path = tmp_path / "data.json", tmp_path / "lock.json"
    path.write_text(json.dumps(modified), encoding="utf-8")
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        benchmark.load_corpus(path, lock_path)
    modified["cases"][1]["id"] = modified["cases"][0]["id"]
    path.write_text(json.dumps(modified), encoding="utf-8")
    lock["dataset_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        benchmark.load_corpus(path, lock_path)
