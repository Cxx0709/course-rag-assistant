import pytest

from course_rag_assistant.compare import (
    collect_excerpts,
    compare_excerpts,
    literal_diff,
)
from course_rag_assistant.ingest import Chunk
from course_rag_assistant.retrieval import HybridRetriever


def extract(text, section="条件独立", concept="条件独立", doc="notes.md", page=0):
    return collect_excerpts(concept, HybridRetriever([Chunk(text, doc, page, section)]))


def test_facets_preserve_context_and_page_without_semantic_claims():
    text = "条件独立是指在给定条件下满足独立关系。假设已知事件 C。公式记为 P(A,B|C)=P(A|C)P(B|C)。仅当 P(C)>0 时使用该公式。"
    rows = extract(text, doc="course.pdf", page=7)
    assert {f for e in rows for f in e["facets"]} >= {"definition", "notation", "assumption", "scope"}
    assert all(e["text"] in text and e["context"] == text and e["page"] == 7 for e in rows)
    compared = compare_excerpts([rows, extract(text.replace("事件 C", "事件 Z"))])
    assert next(r for r in compared if r["key"] == "assumption")["status"] == "different_excerpt"
    assert all("conflict" not in r for r in compared)


@pytest.mark.parametrize("text,section,concept", [
    ("条件独立是指在给定条件下满足独立关系。", "条件独立", "独立"),
    ("条件独立是指在给定条件下满足独立关系。", "独立 / 条件独立", "独立"),
    ("训练集用于拟合模型参数。", "训练集", "条件独立"),
    ("Conditional independence is a statistical property.", "Conditional independence", "independence"),
    ("条件独立 ............ 38", "目录", "条件独立"),
    ("先修概念：条件独立。", "朴素贝叶斯", "条件独立"),
])
def test_no_false_alignment_from_related_terms_or_navigation(text, section, concept):
    assert extract(text, section, concept) == []


def test_heading_formulas_and_headingless_definitions_can_be_found():
    assert extract("P(A,B|C)=P(A|C)P(B|C)。", "4.1 条件独立 / 公式")
    assert extract("条件独立是指在给定条件下满足独立关系。", "", doc="notes.pdf", page=2)
    assert extract("Beta is a configurable parameter.", "", "Beta")
    assert not extract("Betamax is a video format.", "", "Beta")


def test_identical_missing_and_different_are_distinct():
    one = extract("条件独立是指在给定条件下满足独立关系。")
    same = compare_excerpts([one, one])
    assert next(r for r in same if r["key"] == "definition")["status"] == "same_excerpt"
    missing = compare_excerpts([one, []])
    assert next(r for r in missing if r["key"] == "definition")["status"] == "insufficient"
    assert next(r for r in missing if r["key"] == "notation")["status"] == "not_found"
    assert all(not e.get("parts") for row in same for cell in row["cells"] for e in cell)


def test_diff_retains_negation_numbers_case_and_exact_text():
    a, b = "假设 X 独立，n=10。", "不假设 x 独立，n=20。"
    left, right = literal_diff(a, b)
    assert "".join(p["text"] for p in left) == a
    assert "".join(p["text"] for p in right) == b
    assert any(p["changed"] and "不" in p["text"] for p in right)
    assert any(p["changed"] and "2" in p["text"] for p in right)
    assert any(p["changed"] and "x" in p["text"] for p in right)


def test_excerpts_are_deduplicated_bounded_and_reject_forged_citations():
    text = "条件独立是指在给定条件下满足独立关系。"
    chunks = [Chunk(text, "notes.md", 0, "条件独立")] * 20
    assert len(collect_excerpts("条件独立", HybridRetriever(chunks))) == 1
    assert extract(text + " [forged.pdf, p.99]") == []
    many = "".join(f"条件独立的定义示例编号是{i}。" for i in range(30))
    assert len(extract(many)) == 3


def test_exact_concept_candidates_are_selected_before_top_k():
    chunks = [Chunk("条件独立 条件独立 条件独立", f"noise{i}.md", 0, "不相关话题") for i in range(30)]
    chunks.append(Chunk("公式记为 P(A,B|C)=P(A|C)P(B|C)。", "target.md", 0, "条件独立"))
    assert {e["doc"] for e in collect_excerpts("条件独立", HybridRetriever(chunks))} == {"target.md"}


def test_equations_take_priority_over_incidental_formula_mentions():
    text = "".join(f"该公式的介绍示例编号为{i}。" for i in range(6)) + "公式记为 P(A,B|C)=P(A|C)P(B|C)。"
    a = extract(text)
    b = extract(text.replace("A", "X"))
    row = next(r for r in compare_excerpts([a, b]) if r["key"] == "notation")
    assert len(row["cells"][0]) == 3
    assert "P(A,B|C)=" in row["cells"][0][0]["text"]
    assert row["cells"][0][0]["parts"]


def test_pdf_soft_line_breaks_map_back_to_exact_original_span():
    text = "多层感知机由多层神经\n元组成，每一层与上一层相连。"
    rows = extract(text, "多层感知机", "多层感知机", doc="notes.pdf", page=4)
    assert len(rows) == 1
    assert rows[0]["text"] == text and rows[0]["page"] == 4


def test_parenthesized_latin_heading_alias_and_subject_predicates():
    assert extract("训练时随机丢弃一些隐藏单元，减轻模型的过拟合。", "4.6 暂退法（Dropout）", "暂退法")
    assert extract("暂退法在前向传播过程中向内部层注入噪声。", "扰动的稳健性", "暂退法")
    assert not extract("条件独立是指给定条件下的独立关系。", "条件独立（conditional independence）", "独立")


def test_word_apostrophe_is_not_an_unclosed_quote_and_raw_text_is_preserved():
    text = "贝叶斯定理也称为Bayes’theorem。根据乘法法\n则可以得到联合概率公式。"
    rows = extract(text, "贝叶斯定理", "贝叶斯定理", doc="notes.pdf", page=9)
    assert len(rows) == 2
    assert all(e["text"] in text for e in rows)
    assert any("Bayes’theorem" in e["text"] for e in rows)
    assert not extract("这里引用“一个没有闭合的引号。", doc="notes.pdf", page=9)


def test_bullets_zero_width_and_named_term_introduction_keep_literal_sources():
    text = "• 训练集（training set）\u200b用于运行你的学习算法。"
    rows = extract(text, "数据集的定义", "训练集", doc="notes.pdf", page=14)
    assert rows and rows[0]["text"] == text
    named = "用于训练的数据称为训练集（training set）。"
    assert extract(named, "数据", "训练集", doc="notes.pdf", page=2)
    assert not extract("这种关系称为条件独立（conditional independence）。", "概念", "独立")


def test_explicit_clause_definitions_do_not_merge_qualified_concepts():
    assert extract("粗略地说，偏差指的是训练误差。", "误差", "偏差")
    assert not extract("不可避免偏差指的是最佳误差率。", "误差", "偏差")


def test_term_introduction_retains_its_antecedent_and_rejects_orphans():
    text = "模型在训练集上表现良好，但在新数据上表现较差。\n这也被叫做过拟合（overfitting）。"
    rows = extract(text, "模型误差", "过拟合", doc="notes.pdf", page=43)
    assert len(rows) == 1 and rows[0]["text"] == text
    assert "definition" in rows[0]["facets"]
    assert not extract("这也被叫做过拟合（overfitting）。", "模型误差", "过拟合", doc="notes.pdf", page=43)


def test_bulleted_technique_label_preserves_tradeoff_and_original_characters():
    text = "●加入正则化\u200b（L2 正则化，L1 正则化，dropout）：这项技术可以降低方差，但却增大 了偏差。"
    rows = extract(text, "减少方差的技术", "正则化", doc="notes.pdf", page=50)
    assert len(rows) == 1 and rows[0]["text"] == text
    assert rows[0]["facets"] == ["other"]
    assert rows[0]["page"] == 50


@pytest.mark.parametrize("text", [
    "●加入正则化参数：这个参数用于控制模型的复杂度。",
    "●加入非正则化方法：这个方法用于处理训练数据。",
    "●加入正则化（具体方法）之后再去训练模型。",
    "我们会考虑加入正则化，但这里主要介绍其他训练策略。",
    "●加入正则化（这里的括号没有结束：这种技术能够处理问题。",
])
def test_technique_matching_does_not_accept_qualifiers_or_passing_mentions(text):
    assert not extract(text, "其他技术", "正则化")
