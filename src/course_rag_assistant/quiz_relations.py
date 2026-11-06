"""Bounded source relations for independently readable concept choices.

Each result retains complete, contiguous source sentences as evidence. Stems
select only a member's description; the other member remains available at grade
time. No relation is inferred from proximity or a chapter title alone.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .quiz_terms import name_list

_TERM = r"[\u4e00-\u9fff]{2,20}"
_CONDITION = r"(?:if|when|for|with|without|given|under|not|only|unless)\b"
_ALIAS = r"(?:[（(](?!" + _CONDITION + r")[A-Za-z][A-Za-z \-‐‑–]{0,79}[）)])?"
_PAIR = re.compile(
    r"(?P<scope>在[\u4e00-\u9fff]{2,20}中，)?"
    r"前者[（(](?P<left_body>[^（）()。；]{8,100})[）)]称为(?P<left>" + _TERM + r")" + _ALIAS
    + r"，而后者[（(](?P<right_body>[^（）()。；]{8,100})[）)]称为(?P<right>" + _TERM + r")" + _ALIAS + r"。", re.IGNORECASE
)
_FUNCTION = re.compile(
    r"(?P<order>其中|另外)一个(?P<kind>[\u4e00-\u9fff]{1,8})用来"
    r"(?P<body>[^。；]{6,100})，我们将其称为(?P<term>" + _TERM + r")" + _ALIAS + r"。", re.IGNORECASE
)
_INTRO = re.compile(
    r"[\u4e00-\u9fff]{2,30}引入了(?P<entity>" + _TERM + r")" + _ALIAS
    + r"，或简称为(?P<short>" + _TERM + r")" + _ALIAS + r"。", re.IGNORECASE
)
_NUMBER = r"(?:[一二两三四五六七八九十]|10|[1-9])"
_COUNTED = re.compile(
    r"(?P<scope>通常，|在[\u4e00-\u9fff]{2,20}中，)?"
    r"(?P<verb>涉及|包含|使用)(?P<counts>" + _NUMBER + r"个"
    r"(?:、" + _NUMBER + r"个){0,3}和" + _NUMBER + r"个)"
    r"(?P<object>[\u4e00-\u9fff]{2,25})分别(?:被)?称为(?P<names>.+?)"
    r"(?P<category>模型|方法|过程|算法|函数)。"
)


@dataclass(frozen=True)
class RelatedConcept:
    concept: str
    sentence: str
    prompt: str
    group: int


def _contiguous(text: str, source: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return any(compact in re.sub(r"\s+", "", paragraph)
               for paragraph in re.split(r"\n[ \t\r]*\n", source))


def _distinct(left: str, right: str, descriptions: list[str]) -> bool:
    return (left not in right and right not in left and descriptions[0] != descriptions[1]
            and all(left not in d and right not in d for d in descriptions))


def related_concepts(units: list[str], source: str) -> list[RelatedConcept]:
    results = []
    for index, sentence in enumerate(units):
        counted = _COUNTED.fullmatch(sentence)
        if counted and _contiguous(sentence, source):
            counts = re.split(r"、|和", counted["counts"])
            numbers = [str("一二三四五六七八九十".index(n[:-1]) + 1) if n[:-1] in "一二三四五六七八九十"
                       else "2" if n == "两个" else n[:-1] for n in counts]
            names = name_list(counted["names"])
            if (names and len(names) == len(counts) and len(set(numbers)) == len(numbers)
                    and not any(n.name in counted["object"] or n.name.endswith(counted["category"]) for n in names)):
                for count, name in zip(counts, names):
                    # The common category stays visible; it is not silently
                    # lost from a qualified concept or invented in the options.
                    prompt = ("按资料原文补全空缺：" + (counted["scope"] or "") + counted["verb"]
                              + count + counted["object"] + "被称为____" + counted["category"] + "。")
                    results.append(RelatedConcept(name.name, sentence, prompt, index))
            continue
        pair = _PAIR.fullmatch(sentence)
        if pair and _contiguous(sentence, source):
            descriptions = [pair["left_body"], pair["right_body"]]
            if _distinct(pair["left"], pair["right"], descriptions):
                for side in ("left", "right"):
                    prompt = "按资料原文补全空缺：" + (pair["scope"] or "") + pair[side + "_body"] + "称为____。"
                    results.append(RelatedConcept(pair[side], sentence, prompt, index))
            continue
        first = _FUNCTION.fullmatch(sentence)
        second = _FUNCTION.fullmatch(units[index + 1]) if index + 1 < len(units) else None
        if not (first and second and first["order"] == "其中" and second["order"] == "另外"
                and first["kind"] == second["kind"]
                and all(m["term"].endswith(first["kind"]) for m in (first, second))
                and _distinct(first["term"], second["term"], [first["body"], second["body"]])):
            continue
        # Resolve the shared object only from an explicit introduction plus a
        # control statement in the same paragraph, at most four sentences back.
        for start in range(index - 1, max(-1, index - 5), -1):
            intro = _INTRO.fullmatch(units[start])
            if not intro or any(intro["short"] not in m["body"] for m in (first, second)):
                continue
            control = f"为了控制{intro['entity']}，我们需要许多{first['kind']}。"
            if units[index - 1] != control:
                continue
            context = "".join(units[start:index])
            evidence = context + sentence + units[index + 1]
            if len(evidence) > 600 or not _contiguous(evidence, source):
                continue
            # Remove only the introduced object's English glosses. Parenthesized
            # conditions remain and the normal stem gate will reject ambiguity.
            for term in (intro["entity"], intro["short"]):
                context = re.sub(re.escape(term) + r"[（(](?!" + _CONDITION + r")"
                                 r"[A-Za-z][A-Za-z \-‐‑–]{0,79}[）)]", term, context, flags=re.IGNORECASE)
            for member in (first, second):
                prompt = ("按资料原文补全空缺：" + context + member["kind"] + "用来"
                          + member["body"] + "，我们将其称为____。")
                results.append(RelatedConcept(member["term"], evidence, prompt, index))
            break
    return results
