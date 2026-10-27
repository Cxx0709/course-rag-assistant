import pytest

from course_rag_assistant.alignment import (
    aligned_columns,
    dataset_role,
    variance_senses,
)
from course_rag_assistant.compare import compare_excerpts
from course_rag_assistant.ingest import Chunk
from course_rag_assistant.retrieval import HybridRetriever


def retriever(text, section="", doc="notes.pdf", page=1):
    return HybridRetriever([Chunk(text, doc, page, section)])


@pytest.mark.parametrize("query", ["开发集", "验证集"])
def test_aliases_require_role_evidence_on_both_sides_and_keep_original_names(query):
    a = "为了确定候选模型中的最佳模型，我们通常会使用验证集。"
    b = "•开发集（development set）\u200b用于调整参数，选择特征，以及对学习算法作出其它决定。"
    rs = [retriever(a, doc="a.pdf", page=160), retriever(b, doc="b.pdf", page=14)]
    result = aligned_columns(query, rs)
    assert result["alignment"]["status"] == "role_correspondence" and result["comparable"]
    left, right = result["columns"]
    assert left[0]["text"] == a and left[0]["matched_term"] == "验证集" and left[0]["page"] == 160
    assert right[0]["text"] == b and right[0]["matched_term"] == "开发集" and right[0]["page"] == 14
    assert [e["id"] for e in left] == ["1"]


@pytest.mark.parametrize("text", [
    "交叉验证集用于选择模型。", "验证集参数用于选择模型。",
    "验证集不用于选择模型。", "验证集用于选择模型是错误说法。",
    "训练集用于选择模型，另一个术语是验证集。",
    "为了选择模型，我们不会使用验证集。",
    "验证集用于检查数据格式是否正确。",
])
def test_role_evidence_rejects_negation_qualified_terms_and_cooccurrence(text):
    assert not dataset_role("验证集", text)


def test_unsupported_alias_is_reviewable_but_never_highlighted_as_aligned():
    result = aligned_columns("开发集", [retriever("开发集用于选择模型。"),
                                        retriever("验证集用于检查数据格式。")])
    assert all(result["columns"])
    assert not result["comparable"]
    assert result["alignment"]["status"] == "insufficient_evidence"
    rows = compare_excerpts(result["columns"], comparable=False)
    assert any(r["status"] == "alignment_uncertain" for r in rows)
    assert not any("parts" in e for r in rows for cell in r["cells"] for e in cell)


def test_other_names_do_not_expand_and_source_selection_controls_support():
    rs = [retriever("训练集用于学习算法的参数。"), retriever("验证集用于选择模型。")]
    assert aligned_columns("训练集", rs)["alignment"]["status"] == "literal_name"
    assert not aligned_columns("开发集", rs)["comparable"]
    assert not any(aligned_columns("交叉验证集", rs)["columns"])


def test_variance_requires_choice_and_filters_each_excerpt_not_whole_chunk():
    statistics = "方差的平方根被称为标准差。"
    error = "方差指的是算法在测试集上的表现低于训练集的程度。"
    unknown = "方差的概念和之前保持一致。"
    rs = [retriever(statistics + error + unknown, "方差"), retriever(error, "方差")]
    pending = aligned_columns("方差", rs)
    assert pending["needs_disambiguation"] and not any(pending["columns"])
    assert len(pending["alignment"]["sense_options"]) == 2
    stats = aligned_columns("方差", rs, sense="random_variable_variance")
    assert [bool(c) for c in stats["columns"]] == [True, False]
    assert stats["columns"][0][0]["text"] == statistics
    ml = aligned_columns("方差", rs, sense="model_error_variance")
    assert all(c and all(e["text"] == error for e in c) for c in ml["columns"])
    assert variance_senses(unknown) == set()


def test_mixed_sense_unit_and_bare_numbers_are_not_assigned_a_meaning():
    text = "方差的平方根为标准差，方差也在这里描述测试集表现低于训练集的程度。"
    assert len(variance_senses(text)) == 2
    rs = [retriever(text, "方差"), retriever("方差：15%。", "方差")]
    for sense in ("random_variable_variance", "model_error_variance"):
        assert not any(aligned_columns("方差", rs, sense=sense)["columns"])


def test_qualified_query_selects_sense_and_invalid_or_conflicting_choices_fail():
    rs = [retriever("方差的平方根被称为标准差。", "方差")] * 2
    result = aligned_columns("随机变量的方差", rs)
    assert result["alignment"]["sense"] == "random_variable_variance"
    assert all(result["columns"])
    for concept, sense in [("过拟合", "random_variable_variance"), ("方差", "unknown"),
                           ("随机变量的方差", "model_error_variance")]:
        with pytest.raises(ValueError):
            aligned_columns(concept, rs, sense=sense)


def test_sense_filter_runs_before_facet_cap():
    text = "".join(f"方差的定义示例{i}与之前一致。" for i in range(6)) + "方差的平方根被称为标准差。"
    result = aligned_columns("方差", [retriever(text, "方差")] * 2, sense="random_variable_variance")
    assert all(c and len(c) == 1 and "标准差" in c[0]["text"] for c in result["columns"])


def test_both_aliases_in_one_source_keep_bounded_unique_ids():
    text = "".join(f"{term}用于选择模型，示例编号为{i}。" for term in ("开发集", "验证集") for i in range(5))
    result = aligned_columns("验证集", [retriever(text)] * 2)
    for col in result["columns"]:
        assert len(col) == 3 and len({e["id"] for e in col}) == 3
