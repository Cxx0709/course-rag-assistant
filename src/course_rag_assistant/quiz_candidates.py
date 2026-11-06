"""Source-bound concept candidates, kept separate from question construction.

Headings suggest names; they do not establish a definition. These conservative
lexical checks intentionally abstain on mathematical or unresolved passages.
They are not a semantic grader or a guarantee of teaching value.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

from .answer import (
    CitedAnswer,
    _sentences,
    _source_units,
    format_citation,
    verify_citations,
)
from .ingest import Chunk
from .prerequisites import is_prerequisite_statement
from .quiz_content import (
    chapter_number,
    cloze_prompt,
    contextual_definitions,
    contextual_name,
    definition_concept,
    definition_list_groups,
    explicit_contrast,
    teaching_statement,
    usable_concept,
)
from .quiz_relations import related_concepts
from .quiz_terms import synonym_definition
from .retrieval import Hit

_TERM = r"[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9]{1,39}"
_GLOSS = r"\s*[（(]([A-Za-z][A-Za-z0-9 ,_\-‐‑–]{0,79})[）)]"
_NAMED = re.compile(r"(?:称为|叫做|叫作|称作|即为)(?P<term>" + _TERM + r")(?=[（(，,。！：:])")
_ROLE = re.compile(
    r"^(?:[•●]\s*)?(?:(?:当[^。；]{2,60}时|在[^。；]{2,60}条件下)[，,])?"
    r"(?P<term>" + _TERM + r"?)(?:" + _GLOSS + r")?"
    r"(?:用于|控制|衡量|沿着|能够|可以|提供|决定|表示|通过|是一种|是指)"
)
_CONDITION = re.compile(r"^(?:if|when|for|with|without|given|under|not|only|unless)\b", re.IGNORECASE)
# A bounded vocabulary of noun classes, not a list of course-specific answers.
_KINDS = ("误差", "模型", "状态", "函数", "算法", "方法", "过程", "参数", "分布",
          "变量", "运算", "拟合", "集合", "数据集", "门", "层", "率", "性", "集")


@dataclass(frozen=True)
class ConceptCandidate:
    concept: str
    aliases: tuple[str, ...]
    sentence: str
    citation: str
    doc: str
    page: int
    section: str
    chapter: tuple[int, ...] | None
    list_group: tuple[int, int] | None
    origin: str
    kind: str | None
    relation_group: tuple[int, int] | None = None
    stem: str | None = None
    synonyms: tuple[str, ...] = ()

    def evidence(self) -> dict:
        return {key: getattr(self, key) for key in
                ("concept", "sentence", "citation", "doc", "page", "section")}


def aliases_for(concept: str, sentence: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(alias for alias in re.findall(re.escape(concept) + _GLOSS, sentence)
                               if not _CONDITION.match(alias)))


def _kind(concept: str, sentence: str) -> str | None:
    # Explicit source classification takes precedence over a noun ending.
    if re.search(r"(?:这种|这一)(方法|过程|现象|情况)(?:被)?(?:称为|叫做)" + re.escape(concept), sentence):
        category = re.search(r"(?:这种|这一)(方法|过程|现象|情况)(?:被)?(?:称为|叫做)", sentence)[1]
        return category if category in {"方法", "过程"} else next(
            (kind for kind in _KINDS if concept.endswith(kind)), None)
    if re.search(r"解决(?:这个|这一|上述)问题的(?:简单|有效)方法(?:即为|称为)" + re.escape(concept), sentence):
        return "方法"
    match = re.match(re.escape(concept) + r"(?:" + _GLOSS + r")?"
                     r"(?:是指|指的是|是一种)[^。；，,]{0,100}?的(方法|过程|算法|状态|参数|函数)[。；]", sentence)
    if match:
        return match[2]
    return next((kind for kind in _KINDS if concept.endswith(kind)), None)


def extract_candidates(chunks: list[Chunk]) -> list[ConceptCandidate]:
    """Extract complete names and verbatim evidence independently of the target.

    A naming/definition clause owns its full name, so a shorter heading or a
    user-supplied target cannot override it. No context crosses a source chunk.
    """
    candidates = []
    seen = set()
    chapters = {}
    for chunk_index, chunk in enumerate(chunks):
        number = chapter_number(chunk.section)
        if number is not None:
            chapters[chunk.doc] = number
        units = _source_units(chunk)
        for relation in related_concepts(units, chunk.text):
            hit = Hit(chunk, 1, "quiz")
            citation = format_citation(hit)
            verified = verify_citations(CitedAnswer(
                "\n".join(f"{s} {citation}" for s in _sentences(relation.sentence)), [hit], []))
            if (verified.abstention_reason or verified.unsupported_claims
                    or any(is_prerequisite_statement(s) or not teaching_statement(s)
                           for s in _sentences(relation.sentence))):
                continue
            candidates.append(ConceptCandidate(
                relation.concept, aliases_for(relation.concept, relation.sentence), relation.sentence,
                citation, chunk.doc, chunk.page, chunk.section, chapters.get(chunk.doc), None,
                "source_relation", _kind(relation.concept, relation.sentence),
                (chunk_index, relation.group), relation.prompt))
        groups = definition_list_groups(units, chunk.text)
        passages = [(text, None, (chunk_index, groups[i]) if i in groups else None)
                    for i, text in enumerate(units)]
        passages.extend((text, concept, None) for concept, text in contextual_definitions(units, chunk.text))
        headings = {re.sub(r"^\d+(?:\.\d+)*[.、\s]*", "", c).strip()
                    for c in [*chunk.concepts, chunk.section.rsplit(" / ", 1)[-1]]}
        for sentence, contextual, group in passages:
            if not contextual and contextual_name(sentence):
                continue
            synonym = synonym_definition(sentence)
            normalized = synonym[1] if synonym else sentence
            concept = contextual or definition_concept(normalized)
            origin = "contextual" if contextual else "definition"
            if not concept:
                naming = _NAMED.search(sentence)
                role = _ROLE.match(sentence)
                if naming:
                    concept, origin = naming["term"], "naming"
                elif role and (role["term"] in headings or aliases_for(role["term"], sentence)):
                    concept, origin = role["term"], "role"
                else:
                    concept = next((c for c in sorted(headings) if explicit_contrast(c, sentence)), None)
                    origin = "contrast"
            if not concept or not 2 <= len(concept) <= 40 or not 12 <= len(sentence) <= 400:
                continue
            if "的" in concept or re.match(r"^(?:我们|你们|他们|由于|因为|如果|为了|对于)", concept):
                continue
            if not usable_concept(concept, defined=origin == "definition"):
                continue
            if any(is_prerequisite_statement(s) or not teaching_statement(s) for s in _sentences(sentence)):
                continue
            hit = Hit(chunk, 1, "quiz")
            citation = format_citation(hit)
            verified = verify_citations(CitedAnswer(
                "\n".join(f"{s} {citation}" for s in _sentences(sentence)), [hit], []))
            if verified.abstention_reason or verified.unsupported_claims:
                continue
            key = (chunk.doc, chunk.page, chunk.section, concept, re.sub(r"\s+", "", sentence))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(ConceptCandidate(
                concept, tuple(n.alias for n in synonym[0] if n.alias) if synonym else aliases_for(concept, sentence),
                sentence, citation, chunk.doc, chunk.page, chunk.section, chapters.get(chunk.doc), group,
                origin, _kind(concept, normalized), stem=cloze_prompt(concept, normalized) if synonym else None,
                synonyms=tuple(n.name for n in synonym[0][1:]) if synonym else ()))
    return _share_synonyms(candidates)


def _share_synonyms(candidates: list[ConceptCandidate]) -> list[ConceptCandidate]:
    # Only verified, explicit identities in the same document/chapter can
    # propagate. They can exclude an option even when their own stem is unusable.
    groups = {}
    def scope_of(candidate):
        return candidate.doc, candidate.chapter, candidate.section if candidate.chapter is None else None
    for candidate in candidates:
        scope = scope_of(candidate)
        names = {candidate.concept, *candidate.synonyms}
        family = groups.setdefault(scope, [])
        for group in family[:]:
            if names & group:
                names |= group
                family.remove(group)
        family.append(names)
    return [replace(c, synonyms=tuple(sorted(next(g for g in groups[scope_of(c)]
                                                    if c.concept in g) - {c.concept}))) for c in candidates]


def candidate_prompt(candidate: ConceptCandidate) -> str:
    if candidate.stem is not None:
        return candidate.stem
    contrast = explicit_contrast(candidate.concept, candidate.sentence) if candidate.list_group is None else None
    return contrast[0] if contrast else cloze_prompt(candidate.concept, candidate.sentence)


def prompt_rejection(candidate: ConceptCandidate, prompt: str) -> str | None:
    """Check the visible stem without deleting conditions or inventing context."""
    text = prompt.removeprefix("按资料原文补全空缺：")
    if re.match(r"^(?:这|上述|该|其)", candidate.concept):
        return "unresolved_reference"
    if prompt.count("____") != 1:
        return "multiple_or_missing_blanks"
    if (candidate.concept in text or any(n in text for n in candidate.synonyms)
            or any(a.casefold() in text.casefold() for a in candidate.aliases)):
        return "answer_hint"
    if re.match(r"^(?:它|这|其|该|上述|因此|其中|另外一个|前者|后者|相反|接下来|所以|然而|此外)", text):
        return "unresolved_reference"
    if candidate.origin != "contextual" and re.search(r"上述|前述|这类模型|这种方法|这个数量", text):
        return "unresolved_reference"
    if re.search(r"(?:如|见|根据|参考)?[图表式]\s*\d+(?:\.\d+)+|[（(]\d+(?:\.\d+){2,}[）)]", text):
        return "external_figure_or_formula"
    # We cannot reliably resolve notation from surrounding pages. Keep it in
    # source evidence but abstain, rather than stripping it out of the question.
    if re.search(r"[A-Za-zα-ωΑ-Ω∈∑∏=<>≤≥_^]", text.replace("____", "")):
        return "notation_or_latin_context"
    return None


def compatible_options(left: ConceptCandidate, right: ConceptCandidate) -> bool:
    """Require a local list or a recognizable noun class; never pad by proximity."""
    if left.doc != right.doc:
        return False
    if left.chapter != right.chapter:
        return False
    if {left.concept, *left.synonyms} & {right.concept, *right.synonyms}:
        return False
    if {a.casefold() for a in left.aliases} & {a.casefold() for a in right.aliases}:
        return False
    if left.relation_group is not None or right.relation_group is not None:
        return left.relation_group is not None and left.relation_group == right.relation_group
    if left.list_group is not None or right.list_group is not None:
        return left.list_group is not None and left.list_group == right.list_group
    return left.kind is not None and left.kind == right.kind
