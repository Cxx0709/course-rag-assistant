"""Conservative offline answerability screening, not semantic entailment.

The fixed lexical coverage threshold is a development heuristic. Contextual
questions additionally need a record-like source, rather than a textbook example.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .content import navigation_reason
from .facts import record_facts
from .retrieval import Hit, _tokenize
from .text import code_identifiers

MIN_QUERY_COVERAGE = 0.5
_QUESTION_WORDS = re.compile(
    r"有什么不同|有什么区别|什么情况|我想了解|我想知道|请问|能否解释|"
    r"为什么|是什么|什么是|指什么|如何|哪些|哪个|是否|怎么|多少|什么|"
    r"分别|具体|可以|需要|应该|用于|主要|这个|要做|我的|我们"
)
_PERSONAL = re.compile(r"我的|我们|本班|我(?:昨天|今天|上周|本机|上传|训练)|本周|本学期|本次|这次")
_CONTEXT_QUERY = re.compile(r"记录|日志|通知|清单")
_ADVICE = re.compile(r"如何|怎么|为什么|原因|含义|定义|提高|降低|改善|计算")
_RECORD_TYPES = (
    (re.compile(r"截止|教室|几点|考试时间|考试安排|作业.*日期"),
     re.compile(r"课程通知|考试通知|考试安排|作业通知|教学安排|课程安排")),
    (re.compile(r"学号|分组|小组"), re.compile(r"分组名单|分组记录|班级名单")),
    (re.compile(r"准确率|精度|损失|耗时|多少秒|用了.*秒|训练.*结果"),
     re.compile(r"实验日志|实验记录|运行日志|运行记录|训练日志|训练记录")),
    (re.compile(r"数据集.*(?:多少|数量|几张)"),
     re.compile(r"数据集清单|数据集记录|数据集统计|数据统计")),
)


@dataclass
class EvidenceDecision:
    hits: list[Hit]
    reason: str | None
    max_query_coverage: float
    requested_field: str | None = None


def select_evidence(question: str, hits: list[Hit]) -> EvidenceDecision:
    """Keep sufficiently topical passages, and require provenance for local facts.

    A record heading is a routing hint, not proof of ownership or recency. It
    deliberately requires an explicit source role and lexical support together.
    """
    candidates = [h for h in hits if h.score > 0 and h.chunk.text.strip()]
    if not candidates:
        return EvidenceDecision([], "no_evidence", 0.0)
    candidates = [h for h in candidates if navigation_reason(h.chunk.text) is None]
    if not candidates:
        return EvidenceDecision([], "non_body_evidence", 0.0)
    symbols = code_identifiers(question.casefold())
    if symbols:
        candidates = [h for h in candidates if symbols <= code_identifiers(h.chunk.text.casefold())]
        if not candidates:
            return EvidenceDecision([], "missing_code_identifier", 0.0)
    local_record = False
    roles = []
    if (_PERSONAL.search(question) or _CONTEXT_QUERY.search(question)) and not _ADVICE.search(question):
        roles = [role for intent, role in _RECORD_TYPES if intent.search(question)]
        if roles:
            local_record = True
            candidates = [h for h in candidates
                          if all(role.search(f"{h.chunk.doc} {h.chunk.section}") for role in roles)]
            if not candidates:
                return EvidenceDecision([], "missing_context_record", 0.0)
    cleaned = question
    if local_record:
        # A source label routes the question; it need not be repeated in its body.
        for role in roles:
            cleaned = re.sub(rf"(?:{role.pattern})(?:中|里)?", " ", cleaned)
    subjects = re.fullmatch(
        r"(.+?)(?:和|与)(.+?)(?:的(?:作用|用途|功能))?(?:有什么不同|有什么区别|有何区别)[？?]?", cleaned)
    cleaned = " ".join(subjects.groups()) if subjects else cleaned
    terms = set(_tokenize(_QUESTION_WORDS.sub(" ", cleaned)))
    if not terms:
        return EvidenceDecision([], "unclear_question", 0.0)
    coverage = [len(terms & set(_tokenize(h.chunk.text))) / len(terms) for h in candidates]
    selected = [h for h, score in zip(candidates, coverage) if score >= MIN_QUERY_COVERAGE]
    if selected and local_record:
        facts = record_facts(question, candidates)
        if facts.reason:
            return EvidenceDecision([], facts.reason, max(coverage), facts.field)
        selected = [h for h in selected if h in facts.hits]
    return EvidenceDecision(selected, None if selected else "insufficient_query_coverage", max(coverage))
