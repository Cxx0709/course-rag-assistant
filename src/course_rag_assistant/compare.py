"""Two-source, extractive concept comparison with literal difference marking.

Facet labels are lexical reading aids, not semantic equivalence/contradiction
judgments. Every excerpt and every highlighted character comes from the source.
"""
from __future__ import annotations

import re
from dataclasses import replace
from difflib import SequenceMatcher

from .answer import _CITATION_RE, _source_units
from .prerequisites import concept_name
from .retrieval import HybridRetriever

FACETS = (
    ("definition", "定义与术语", r"是指|指的是|定义|称为|称作|叫作|叫做|\bis\b|\bmeans\b|\bdefined\b"),
    ("notation", "公式与符号", r"[=＝∏∑∩∪⊥]|表示|记为|记作|符号|公式|\bdenot"),
    ("assumption", "假设", r"假设|前提|\bassum"),
    ("scope", "适用条件", r"适用|仅当|当且仅当|给定|要求|条件下|若|如果|\bprovided\b|\bwhen\b|\bif\b"),
    ("other", "其他相关原文", None),
)


def _facet_strength(key: str, text: str) -> int:
    patterns = {
        "definition": r"是指|指的是|定义为|定义是|\bdefined\b|\bmeans\b",
        "notation": r"[=＝∏∑∩∪⊥]|记为|记作|符号",
        "assumption": r"^(?:在此|这里)?假设|前提是|\bassum",
        "scope": r"适用|仅当|当且仅当|要求|\bprovided\b",
    }
    return int(bool(key in patterns and re.search(patterns[key], text, re.IGNORECASE)))


def normalized(text: str) -> str:
    # Keep case, negation, numbers and operators: all may carry meaning.
    return re.sub(r"[\s\u200b\ufeff\u00ad]+", "", text)


def _original_span(text: str, sentence: str) -> str | None:
    """Map whitespace-normalized PDF sentences back to exact source spans."""
    offsets = [index for index, char in enumerate(text) if not char.isspace() and char not in "\u200b\ufeff\u00ad"]
    compact = "".join(text[index] for index in offsets).replace("’", "'")
    target = normalized(sentence).replace("’", "'")
    start = compact.find(target) if target else -1
    if start < 0:
        return None
    return text[offsets[start]:offsets[start + len(target) - 1] + 1]


def _comparison_units(chunk):
    # A typographic apostrophe inside a Latin word (Bayes’theorem, don't)
    # is not an unmatched closing quotation mark. Normalize only for sentence
    # segmentation, then map every result back to the untouched source span.
    text = re.sub(r"(?<=[A-Za-z])’(?=[A-Za-z])", "'", chunk.text)
    text = re.sub(r"[\u200b\ufeff\u00ad]", "", text)
    return _source_units(replace(chunk, text=text))


def _topic_heading(concept: str, section: str) -> bool:
    suffix = r"(?:的)?(?:定义|概念|性质|假设|公式|符号|适用条件|definition|notation)?"
    parts = [concept_name(part) for part in section.split(" / ")]
    # Parenthesized Latin aliases in a heading retain the explicitly named
    # Chinese subject, e.g. 暂退法（Dropout）. This is not synonym inference.
    alias = r"(?:\s*[（(][A-Za-z][A-Za-z0-9 ,_\-]*[）)])?"
    return any(re.fullmatch(re.escape(concept) + alias + suffix, part, re.IGNORECASE)
               and all(re.fullmatch(r"定义|概念|性质|假设|公式|符号|适用条件|示例|definition|notation", child, re.IGNORECASE)
                       for child in parts[index + 1:]) for index, part in enumerate(parts))


def _named_statement(concept: str, text: str) -> bool:
    # Require the concept as the subject, avoiding e.g. 独立 inside 条件独立
    # and similarly named English identifiers. Heading-based passages may also
    # contain follow-on sentences or formulas without repeating the concept.
    text = re.sub(r"[\u200b\ufeff\u00ad]", "", text)
    # PDF line wrapping may insert spaces inside a Chinese concept name.
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    prefix = r"(?:^|[。！？\n；;，,：:])\s*(?:[•●]\s*)?"
    predicate = r"(?:是|指|的|表示|定义|要求|假设|适用|由|在|通过|沿|用于|可以|能够|[：:，,（(]|\s+(?:is|means|denotes|requires)\b)"
    named = re.search(prefix + re.escape(concept) + predicate, text, re.IGNORECASE)
    introduced = re.search(r"(?:称为|称作|叫作|叫做|称之为)\s*[“‘\"']?" + re.escape(concept)
                           + r"(?=[，。；：、\s（(）)”’\"']|$)", text, re.IGNORECASE)
    # Explicit technique labels, e.g. "●加入正则化（L2，L1）：...".
    # Require a bullet and colon; a passing mention or qualified term is not
    # a passage about the queried concept. Keep any caveats in the full unit.
    technique = re.search(r"(?:^|[。！？\n])\s*[•●]\s*(?:加入|添加|使用|采用)\s*"
                          + re.escape(concept) + r"\s*(?:[（(][^（）()\n。！？]{1,160}[）)])?\s*[：:]",
                          text, re.IGNORECASE)
    return bool(named or introduced or technique)


def collect_excerpts(concept: str, retriever: HybridRetriever, *, matcher=None, accept=None) -> list[dict]:
    """Collect literal spans, optionally using a scoped term rule or sense filter.

    Custom matching replaces heading inference; acceptance runs before the
    per-facet limit so unsupported senses cannot crowd out supported evidence.
    """
    matches = matcher or (lambda text: _named_statement(concept, text))
    candidates = []
    for chunk in retriever.chunks:
        in_topic = matcher is None and (_topic_heading(concept, chunk.section) or concept in chunk.concepts)
        if in_topic or matches(chunk.text):
            candidates.append(chunk)
    if not candidates:
        return []
    # Match the concept before limiting ranked results, so incidental mentions
    # cannot crowd out a genuine concept section in a long textbook.
    ranked = HybridRetriever(candidates).search(concept, top_k=12)
    seen = set()
    per_facet = {key: [] for key, _, _ in FACETS}
    for hit in ranked:
        chunk = hit.chunk
        in_topic = matcher is None and (_topic_heading(concept, chunk.section) or concept in chunk.concepts)
        units = _comparison_units(chunk)
        for index, sentence in enumerate(units):
            if not in_topic and not matches(sentence):
                continue
            if re.match(r"^(?:这|它|该|此|上述)", sentence) and re.search(r"称为|称作|叫作|叫做", sentence):
                # "This is called overfitting" alone is not an explanation.
                # Preserve the adjoining antecedent in the same source block.
                if index == 0:
                    continue
                sentence = "".join(units[max(0, index - 2):index + 1])
            sentence = _original_span(chunk.text, sentence)
            # Preserve literal quotations and reject forged citation markers,
            # declarations, incomplete introductions and very large excerpts.
            if (sentence is None or not 8 <= len(sentence) <= 1000
                    or sentence.rstrip().endswith(("：", ":")) or _CITATION_RE.search(sentence)
                    or sentence.startswith("先修概念") or normalized(sentence) in seen):
                continue
            if accept is not None and not accept(sentence):
                continue
            facets = [key for key, _, pattern in FACETS if pattern and re.search(pattern, sentence, re.IGNORECASE)]
            if not facets and _named_statement(concept, sentence) and re.search(r"是|\bis\b", sentence):
                facets = ["definition"]
            facets = facets or ["other"]
            seen.add(normalized(sentence))
            excerpt = {"text": sentence, "doc": chunk.doc, "page": chunk.page, "section": chunk.section,
                       "context": chunk.text if len(chunk.text) <= 8000 else None}
            for key in facets:
                per_facet[key].append(excerpt)
                per_facet[key].sort(key=lambda e: _facet_strength(key, e["text"]), reverse=True)
                del per_facet[key][3:]
    selected = {}
    for key, evidence in per_facet.items():
        for excerpt in evidence:
            item = selected.setdefault(normalized(excerpt["text"]),
                                       {**excerpt, "id": str(len(selected) + 1), "facets": []})
            item["facets"].append(key)
    return list(selected.values())


def literal_diff(left: str, right: str) -> tuple[list[dict], list[dict]]:
    """Mark literal changes without inventing or removing source characters."""
    sides = ([], [])
    for tag, a, b, c, d in SequenceMatcher(None, left, right, autojunk=False).get_opcodes():
        if a != b:
            sides[0].append({"text": left[a:b], "changed": tag != "equal"})
        if c != d:
            sides[1].append({"text": right[c:d], "changed": tag != "equal"})
    return sides


def compare_excerpts(columns: list[list[dict]], *, comparable=True) -> list[dict]:
    rows = []
    for key, label, _ in FACETS:
        cells = [sorted([dict(e) for e in column if key in e["facets"]],
                        key=lambda e: _facet_strength(key, e["text"]), reverse=True) for column in columns]
        texts = [{normalized(e["text"]) for e in cell} for cell in cells]
        if not any(texts):
            status, note = "not_found", "两份资料均未检出本项可核对的原文。"
        elif not all(texts):
            status, note = "insufficient", "有一份资料未检出本项原文，暂不能比较；这不代表该资料没有相关内容。"
        elif not comparable:
            status, note = "alignment_uncertain", "尚无足够依据将两边术语按同一用途对应；请核对原文，暂不标记文字差异。"
        elif texts[0] == texts[1]:
            status, note = "same_excerpt", "本次摘录的文字一致（忽略空白），不代表两份资料的全部论述一致。"
        else:
            status, note = "different_excerpt", "本次摘录存在文字差异，请结合条件与上下文核对；文字不同不等于观点冲突。"
            parts = literal_diff(cells[0][0]["text"], cells[1][0]["text"])
            for index in range(2):
                cells[index][0]["parts"] = parts[index]
        rows.append({"key": key, "label": label, "status": status, "note": note, "cells": cells})
    return rows
