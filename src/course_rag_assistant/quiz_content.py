"""Conservative source-cloze selection rules, not a semantic quality grader."""
from __future__ import annotations

import re

_ALIAS = r"(?:\s*[（(](?P<alias>[A-Za-z][A-Za-z0-9 ,_\-]{0,79})[）)])?"
_DEFINITION = re.compile(
    r"^[•●\s]*([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9]{1,39}?)"
    + _ALIAS + r"\s*(?:是指|指的是|是一种|是用来)"
)
_LABELS = {"小结", "本章小结", "练习", "目录", "参考文献", "致谢", "前言", "论坛"}


def definition_concept(sentence: str) -> str | None:
    """Read an explicitly defined term, including a parenthesized Latin alias."""
    match = _DEFINITION.match(sentence)
    return match[1] if match else None


def cloze_prompt(concept: str, sentence: str) -> str:
    """Hide a defined term's Latin alias; leave source evidence untouched."""
    definition = _DEFINITION.match(sentence)
    if definition and definition[1] == concept and definition["alias"]:
        alias = definition["alias"]
        # Parenthesized conditions are not aliases. Preserve them even if the
        # permissive definition recognizer accepted their syntax.
        if not re.match(r"(?:if|when|for|with|without|given|under|not)\b", alias, re.I):
            sentence = re.sub(re.escape(concept) + r"\s*[（(]" + re.escape(alias) + r"[）)]",
                              lambda _: concept, sentence)
    return "按资料原文补全空缺：" + sentence.replace(concept, "____")


def teaching_statement(sentence: str) -> bool:
    # Keep knowledge statements containing negation, conditions or limitations.
    # Reject only recognizable questions, reader directions and topic previews.
    if re.search(r"[?？]", sentence):
        return False
    if re.search(r"(?:我们|你们|读者|本[书章节])(?:将会|将|会|需要|首先|先|接下来|在此|进一步|主要|重点|简要|详细|一起|要|再|来|应当|应该|可以|\s)*"
                 r"(?:了解|学习|讨论|介绍|回顾|讲解|复习|探讨)", sentence):
        return False
    if re.search(r"(?:什么|如何|怎样|为何|为什么)[^。！？]{0,60}(?:问题|话题)[。！]?$", sentence):
        return False
    if re.search(r"^(?:为了|下面|接下来)[^。！？]*(?:给出|举|来看)[^。！？]*(?:例子|示例|例题)", sentence):
        return False
    if re.match(r"^[•●\s]*(?:请|试着|尝试)(?:阅读|思考|回答|讨论|完成|证明|计算)", sentence):
        return False
    return True


def usable_concept(concept: str, *, defined: bool = False) -> bool:
    if concept in _LABELS:
        return False
    # Procedure headings are not stand-alone concept names. An explicit
    # definition can override this (e.g. a course that defines “训练模型”).
    return defined or not re.match(r"^(?:初始化|实现|安装|导入|运行|使用|读取|下载|加载|训练模型|创建|绘制)", concept)


def statement_rank(concept: str, sentence: str) -> int:
    if definition_concept(sentence) == concept:
        return 0
    direct = r"^[•●\s]*" + re.escape(concept) + _ALIAS + r"\s*(?:是|用于|表示|控制|提供|决定|衡量|沿着|通过|能够|可以)"
    return 1 if re.search(direct, sentence) else 2


def chapter_number(section: str) -> tuple[int, ...] | None:
    """Numbered chapter/section family used only to rank nearby distractors."""
    match = re.match(r"^(\d+(?:\.\d+)*)\s+\S", section)
    return tuple(map(int, match[1].split(".")[:2])) if match else None
