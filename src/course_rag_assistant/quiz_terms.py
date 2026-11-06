"""Explicit source names: synonyms are identities; enumerations are choices."""
from __future__ import annotations

import re
from dataclasses import dataclass

TERM = r"[\u4e00-\u9fff]{2,20}"
CONDITION = r"(?:if|when|for|with|without|given|under|not|only|unless)\b"
GLOSS = r"(?:\s*[（(](?!" + CONDITION + r")(?P<alias>[A-Za-z][A-Za-z0-9 \-‐‑–]{0,79})[）)])?"
_NAME = re.compile(r"(?P<name>" + TERM + r")" + GLOSS, re.IGNORECASE)
_LINK = re.compile(r"，?(?:又称|也称为|亦称|也叫做)")
_DEFINITION = re.compile(r"，?(?:是指|指的是|是一种|是用来)")


@dataclass(frozen=True)
class SourceName:
    name: str
    alias: str | None


def name_list(text: str) -> list[SourceName]:
    """Parse an entire 2–5 member list; never accept a prefix or a condition."""
    pieces = re.split(r"、|和", text)
    if not 2 <= len(pieces) <= 5:
        return []
    matches = [_NAME.fullmatch(piece) for piece in pieces]
    if not all(matches):
        return []
    names = [SourceName(m["name"], m["alias"]) for m in matches]
    if len({n.name for n in names}) != len(names):
        return []
    aliases = [n.alias.casefold() for n in names if n.alias]
    if len(set(aliases)) != len(aliases):
        return []
    return names


def synonym_definition(sentence: str) -> tuple[list[SourceName], str] | None:
    """Read a subject's explicit alternative names before its definition.

    Bare 'or' and 'and' do not assert equivalence. The normalized statement is
    used only for stem construction; citation evidence always stays untouched.
    """
    first = _NAME.match(sentence)
    if not first:
        return None
    names = [SourceName(first["name"], first["alias"])]
    end = first.end()
    while len(names) < 5:
        link = _LINK.match(sentence, end)
        if not link:
            break
        member = _NAME.match(sentence, link.end())
        if not member:
            return None
        names.append(SourceName(member["name"], member["alias"]))
        end = member.end()
    if len(names) < 2 or not _DEFINITION.match(sentence, end):
        return None
    if len({n.name for n in names}) != len(names):
        return None
    return names, names[0].name + sentence[end:].removeprefix("，")
