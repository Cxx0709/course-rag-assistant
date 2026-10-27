"""Small, evidence-gated terminology profiles for two-source reading.

Profiles describe a reading scope, not universal synonym or equivalence claims.
Only selected sources can supply evidence; unsupported senses are not inferred.
"""
from __future__ import annotations

import re

from .compare import collect_excerpts, normalized

DATASET_TERMS = ("开发集", "验证集")
SENSES = (
    {"id": "random_variable_variance", "label": "随机变量的方差", "description": "随机变量相对期望的离散程度，例如与标准差的关系。"},
    {"id": "model_error_variance", "label": "模型误差分析中的方差", "description": "本次按训练与开发／测试表现差距的操作性定义查找。"},
)
QUALIFIED_VARIANCE = {
    "随机变量方差": "random_variable_variance",
    "随机变量的方差": "random_variable_variance",
    "模型误差中的方差": "model_error_variance",
    "模型误差分析中的方差": "model_error_variance",
}


def variance_senses(text: str) -> set[str]:
    """Conservative cues in the excerpt itself, never the whole document."""
    senses = set()
    for clause in re.split(r"[。！？；;]", normalized(text)):
        if not re.search(r"(?<!协)方差", clause):
            continue
        if "标准差" in clause or ("随机变量" in clause and any(t in clause for t in ("期望", "离散", "偏离"))):
            senses.add("random_variable_variance")
        if ("训练" in clause and any(t in clause for t in ("开发", "测试"))
                and any(t in clause for t in ("误差", "错误率", "表现", "泛化"))):
            senses.add("model_error_variance")
    return senses


def dataset_role(term: str, text: str) -> bool:
    """Recognize explicit affirmative use for tuning/model selection.

    Restrict the term to a subject or the object of a purpose statement. A
    cross-validation term, a negative statement, or co-occurrence is insufficient.
    """
    for clause in re.split(r"[。！？；;\n]", normalized(text)):
        if any(word in clause for word in ("不", "未", "无", "并非", "而非", "错误", "误称", "误以为")):
            continue
        subject = re.search(r"(?:^|[，,：:•●])" + re.escape(term)
                            + r"(?:[（(][^（）()。！？]{1,100}[）)])?"
                            + r"(?:用于|用来|用以)(?:调整参数|调参|选择(?:模型|特征)|模型选择|超参数选择)", clause)
        purpose = re.search(r"为了(?:确定候选模型中的最佳模型|选择最佳模型|选择模型|调整参数|调参)"
                            + r"[，,](?:我们)?(?:通常)?(?:会)?(?:使用|用)" + re.escape(term)
                            + r"(?:[，,]|$)", clause)
        if subject or purpose:
            return True
    return False


def _tag(excerpts, term, sense=None):
    return [{**e, "matched_term": term, "sense": sense} for e in excerpts]


def _merge(items):
    """Deduplicate literal evidence and retain the three-excerpts-per-facet cap."""
    seen, counts, result = set(), {}, []
    for item in items:
        key = (item["doc"], item["page"], item["section"], item["text"])
        if key in seen:
            continue
        seen.add(key)
        facets = [f for f in item["facets"] if counts.get(f, 0) < 3]
        if facets:
            for facet in facets:
                counts[facet] = counts.get(facet, 0) + 1
            result.append({**item, "facets": facets})
    return result


def aligned_columns(concept: str, retrievers, *, sense: str | None = None) -> dict:
    """Return safe comparison columns plus an explicit alignment decision."""
    explicit = QUALIFIED_VARIANCE.get(concept)
    if sense is not None and (concept != "方差" and explicit is None):
        raise ValueError("只有方差对照支持选择这组含义。")
    if sense is not None and sense not in {s["id"] for s in SENSES}:
        raise ValueError("请选择支持的方差含义。")
    if explicit and sense and explicit != sense:
        raise ValueError("概念名称与所选含义不一致，请重新选择。")
    sense = sense or explicit
    if concept == "方差" or explicit:
        if sense is None:
            return {"columns": [[], []], "comparable": False, "needs_disambiguation": True,
                    "alignment": {"status": "needs_disambiguation", "sense_options": list(SENSES),
                                  "note": "方差有不同用法。请先选择含义，再查看所选资料中的对应原文。"}}
        columns = [_tag(collect_excerpts("方差", r, accept=lambda text: variance_senses(text) == {sense}),
                        "方差", sense) for r in retrievers]
        label = next(s["label"] for s in SENSES if s["id"] == sense)
        return {"columns": columns, "comparable": True,
                "alignment": {"status": "sense_selected", "sense": sense, "label": label,
                              "sense_options": list(SENSES),
                              "note": f"当前含义：{label}。仅显示原文中有明确语境依据的摘录；含义不明的片段已排除，未检出不代表原书没有相关内容。"}}
    if concept in DATASET_TERMS:
        role_columns, literal_columns = [], []
        for retriever in retrievers:
            role, literal = [], []
            # Prefer the requested spelling, but allow either term only under
            # this explicit model-selection scope. No transitive alias expansion.
            for term in dict.fromkeys((concept, *DATASET_TERMS)):
                role.extend(_tag(collect_excerpts(term, retriever, matcher=lambda text, t=term: dataset_role(t, text)),
                                 term, "model_selection_data"))
                literal.extend(_tag(collect_excerpts(term, retriever), term))
            # The same physical sentence may describe both terms. Preserve its
            # first spelling label and do not duplicate it or erase provenance.
            role_columns.append(_merge(role))
            literal_columns.append(_merge(literal))
        supported = all(role_columns)
        columns = role_columns if supported else [role or literal for role, literal in zip(role_columns, literal_columns)]
        for col in columns:
            for index, e in enumerate(col, 1):
                e["id"] = str(index)
        return {"columns": columns, "comparable": supported,
                "alignment": {"status": "role_correspondence" if supported else "insufficient_evidence",
                              "label": "开发集／验证集：模型选择与调参", "terms": list(DATASET_TERMS),
                              "note": "按原文中的模型选择或调参用途对应开发集与验证集；这不是所有语境下的同义认定。"
                              if supported else "尚未在两份资料中都找到模型选择或调参用途的明确说明。以下仅列候选原文，不认定术语对应，也不标记跨资料差异。"}}
    return {"columns": [collect_excerpts(concept, r) for r in retrievers], "comparable": True,
            "alignment": {"status": "literal_name", "note": "按原概念名称查找；未推断其他术语或含义。"}}
