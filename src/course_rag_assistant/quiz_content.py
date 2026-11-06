"""Conservative source-cloze selection rules, not a semantic quality grader."""
from __future__ import annotations

import re

_ALIAS = r"(?:\s*[（(](?P<alias>[A-Za-z][A-Za-z0-9 ,_\-‐‑–]{0,79})[）)])?"
_DEFINITION = re.compile(
    r"^[•●\s]*([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9]{1,39}?)"
    + _ALIAS + r"\s*(?:是指|指的是|是一种|是用来)"
)
_LABELS = {"小结", "本章小结", "练习", "目录", "参考文献", "致谢", "前言", "论坛"}
# A list label alone can introduce an example or an instruction. Require
# the explanation to explicitly name the same concept again at the end.
_LIST_DEFINITION = re.compile(
    r"^\d{1,2}[.、]\s*(?P<term>[\u4e00-\u9fff]{2,20})" + _ALIAS
    + r"[：:]\s*(?P<body>[^。！？?！]{12,320})[，,](?:即为|这就是|这称为)"
      r"“(?P=term)”(?:原则|性质)?[。！]$"
)


def _listed_definition(sentence: str) -> re.Match | None:
    match = _LIST_DEFINITION.fullmatch(sentence)
    if not match:
        return None
    alias = match["alias"]
    if alias and (re.match(r"(?:if|when|for|with|without|given|under|not)\b", alias, re.IGNORECASE)
                  or alias.casefold() in match["body"].casefold()):
        return None
    return match


def definition_list_groups(units: list[str], source: str) -> dict[int, int]:
    """Identify consecutive, adjacent items; a singleton cannot borrow options."""
    groups = {}
    compact = re.sub(r"\s+", "", source)
    previous_number = None
    for i, unit in enumerate(units):
        if not _listed_definition(unit):
            previous_number = None
            continue
        number = int(re.match(r"\d+", unit)[0])
        follows = (previous_number is not None and number == previous_number + 1
                   and re.sub(r"\s+", "", units[i - 1] + unit) in compact)
        groups[i] = groups[i - 1] if follows else i
        previous_number = number
    return groups
_NAMING = re.compile(
    r"(?:这种|这一)(?:现象|情况|过程|方法)(?:被)?(?:称为|叫做)"
    r"([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9]{1,39}?)" + _ALIAS + r"[。！]$"
)


# Keep the diagnosis in the question, and the full procedure in feedback.
_METHOD = re.compile(r"解决(?:这个|这一|上述)问题的(?:简单|有效)方法(?:即为|称为)"
                     r"([\u4e00-\u9fff]{2,16})(?:[（(][A-Za-z-]+[）)])?[：:](.+[。！])$")
_PROBLEM = re.compile(r"丢失|缺失|不足|无法|失败|冲突|溢出")


def contextual_name(sentence: str) -> str | None:
    match = _NAMING.search(sentence) or _METHOD.search(sentence)
    return match[1] if match else None


def _problem_context(units: list[str], index: int, source: str, method: re.Match) -> str | None:
    # Take only adjacent statements that describe a difficulty. Never stitch
    # over a skipped heading, a blank paragraph, or an intervening sentence.
    start = index
    while start > max(0, index - 3) and _PROBLEM.search(units[start - 1]):
        start -= 1
    prefix = "".join(units[start:index])
    if not prefix or re.match(r"^(?:它|这|其|该|因此|但|此外|然而|上述)", prefix):
        return None
    if method[1] in prefix or re.search(r"仅在|仅当|只有|除非|前提|必须|仅限", method[2]):
        return None
    text = prefix + units[index]
    if len(text) > 400 or not all(teaching_statement(s) for s in units[start:index + 1]):
        return None
    compact = re.sub(r"\s+", "", text)
    return text if any(compact in re.sub(r"\s+", "", paragraph)
                       for paragraph in re.split(r"\n[ \t\r]*\n", source)) else None


def contextual_definitions(sentences: list[str], source: str) -> list[tuple[str, str]]:
    """Keep a naming sentence with up to three contiguous source sentences.

    No page/chunk stitching or removal of intervening headings is allowed.
    This bounded window is a reading aid, not automatic pronoun resolution.
    """
    passages = []
    compact_source = re.sub(r"\s+", "", source)
    for index, sentence in enumerate(sentences):
        method = _METHOD.fullmatch(sentence)
        if method:
            text = _problem_context(sentences, index, source, method)
            if text:
                passages.append((method[1], text))
            continue
        match = _NAMING.fullmatch(sentence)
        if not match or index == 0:
            continue
        context = sentences[max(0, index - 3):index + 1]
        text = "".join(context)
        if (len(text) > 400 or re.sub(r"\s+", "", text) not in compact_source
                or re.match(r"^(?:它|这|其|该|上述|因此|其中|此外|然而|另一方面|然后|最后)", context[0])
                or not all(teaching_statement(s) for s in context)):
            continue
        passages.append((match[1], text))
    return passages


def definition_concept(sentence: str) -> str | None:
    """Read an explicitly defined term, including a parenthesized Latin alias."""
    match = _DEFINITION.match(sentence) or _listed_definition(sentence)
    return match[1] if match else None


def explicit_contrast(concept: str, sentence: str) -> tuple[str, str] | None:
    """Expose a choice only when the source itself contrasts two named terms.

    The saved sentence is untouched. Keep the leading explanation in the quiz;
    the answer and the source's explicitly excluded alternative become options.
    """
    match = re.search(
        r"(?:其实是|实际上是|实际是|实为)(?P<answer>" + re.escape(concept) + r")"
        + r"(?:\s*[（(][A-Za-z]+(?:[-‐‑–][A-Za-z]+)*[）)])?"
          r"[，,]\s*而不是(?P<other>[\u4e00-\u9fff]{2,16})[。！]$", sentence)
    if not match or match.start() < 4:
        return None
    other = match["other"]
    lead = sentence[:match.start("answer")]
    if (other[-2:] != concept[-2:] or concept in other or other in concept
            or concept in lead or other in lead):
        return None
    return "按资料原文补全空缺：" + lead + "____。", other


def cloze_prompt(concept: str, sentence: str) -> str:
    """Hide a defined term's Latin alias; leave source evidence untouched."""
    method = _METHOD.search(sentence)
    if method and sentence[:method.start()].endswith(("。", "！")) and method[1] == concept:
        lead = sentence[:method.start(1)].removeprefix("如上所述，")
        return "按资料原文补全空缺：" + lead + "____。"
    listed = _listed_definition(sentence)
    if listed and listed["term"] == concept:
        body = listed["body"].replace(concept, "____")
        return f"按资料原文补全空缺：____：{body}。"
    # Match aliases at the complete name, including multiword inline names.
    # The candidate extractor owns name boundaries; this helper only masks.
    for definition in re.finditer(re.escape(concept) + _ALIAS, sentence):
        alias = definition["alias"]
        # Parenthesized conditions are not aliases. Preserve them even if the
        # permissive definition recognizer accepted their syntax.
        if alias and not re.match(r"(?:if|when|for|with|without|given|under|not|only|unless)\b", alias, re.IGNORECASE):
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
    if re.match(r"^(?:现在|下面|接下来)?(?:引用|回顾|应用|考虑)(?:上述|前述|前面|以上)", sentence):
        return False
    if re.match(r"^(?:回想|回忆|回顾)(?:一下)?(?:我们)?(?:曾经)?(?:在)?"
                r"(?:第?\d+(?:\.\d+)*[章节]|上一?[章节]|前文|前面)", sentence):
        return False
    if (re.match(r"^(?:然而|不过|但是|因此)?(?:这种|这一|上述|该)(?:方法|过程|情况|模型|结论)", sentence)
            and not _NAMING.fullmatch(sentence)):
        return False
    # Generic association alone supplies no direction, mechanism or condition
    # that distinguishes the missing concept from plausible alternatives.
    if re.search(r"(?:和|与|之间).{0,40}(?:存在|具有|有)(?:着)?"
                 r"(?:(?:一定|某种|密切|直接)的?)?(?:关系|关联|联系)[。！]$", sentence):
        return False
    # A loss plateau alone is not enough context to identify a learning state.
    return not (_DEFINITION.match(sentence) and re.search(
        r"(?:是指|指的是)(?:模型)?(?:无法|不能)继续(?:减少|降低)(?:训练)?(?:误差|损失)[。！]$", sentence))


def usable_concept(concept: str, *, defined: bool = False) -> bool:
    if concept in _LABELS:
        return False
    # Procedure headings are not stand-alone concept names. An explicit
    # definition can override this (e.g. a course that defines “训练模型”).
    return defined or not re.match(r"^(?:从零开始|初始化|实现|安装|导入|运行|使用|读取|下载|加载|训练模型|创建|绘制)", concept)


def statement_rank(concept: str, sentence: str) -> int:
    if definition_concept(sentence) == concept:
        return 0
    direct = r"^[•●\s]*" + re.escape(concept) + _ALIAS + r"\s*(?:是|用于|表示|控制|提供|决定|衡量|沿着|通过|能够|可以)"
    return 1 if re.search(direct, sentence) else 2


def chapter_number(section: str) -> tuple[int, ...] | None:
    """Numbered chapter/section family used only to rank nearby distractors."""
    match = re.match(r"^(\d+(?:\.\d+)*)\s+\S", section)
    return tuple(map(int, match[1].split(".")[:2])) if match else None
