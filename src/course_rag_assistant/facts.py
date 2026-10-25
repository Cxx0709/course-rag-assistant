"""Conservative field/value checks for explicitly identified local records.

This is a small supported record grammar, not a general semantic extractor.
Values are kept tied to their field, and conflicting records are not resolved
by retrieval rank. Original passages are never changed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from .retrieval import Hit

_METRIC = re.compile(r"(?P<scope>训练|测试|验证)?(?:集)?(?P<name>准确率|精度|损失)")
_UNKNOWN = re.compile(r"尚未|暂未|未测|待测|未统计|待统计|未公布|未知|暂无|没有|未记录|不详|待定|尚无|缺失|未提供|未确认|待确认|不是|并非")
_NONFACT = re.compile(r"目标|计划|预计|预期|假设|例如|举例")
_LINK = r"\s*(?:[:：=]|为|是|达到|等于|测得|共有|有)?\s*"
_NUMBER = r"[+-]?\d+(?:\.\d+)?"


@dataclass(frozen=True)
class Field:
    kind: str
    name: str
    scope: str = ""

    @property
    def label(self):
        return self.scope + self.name


@dataclass
class FactDecision:
    hits: list[Hit]
    reason: str | None = None
    field: str | None = None


def requested_fields(question: str) -> list[Field]:
    fields = [Field("metric", m['name'], m['scope'] or "") for m in _METRIC.finditer(question)]
    if re.search(r"耗时|用时|多少秒|用了.*秒|训练时间|运行时间", question):
        fields.append(Field("duration", "耗时"))
    if "数据集" in question and re.search(r"多少|数量|几张", question):
        fields.append(Field("count", "图片数量"))
    if "截止" in question:
        fields.append(Field("deadline", "截止日期"))
    elif re.search(r"考试.*(?:几点|时间)|几点.*考试", question):
        fields.append(Field("exam_time", "考试时间"))
    return list(dict.fromkeys(fields))


def _metric_values(field: Field, clause: str) -> tuple[set[str], bool]:
    values, unknown = set(), False
    matches = list(_METRIC.finditer(clause))
    for i, match in enumerate(matches):
        if match['name'] != field.name or (field.scope and match['scope'] != field.scope):
            continue
        # Treat planned values as non-evidence, not as contradictory observations.
        prefix = clause[max(0, match.start() - 8):match.start()]
        if _NONFACT.search(prefix):
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(clause)
        tail = clause[match.end():end]
        if _UNKNOWN.search(tail):
            unknown = True
            continue
        number = re.match(_LINK + rf"(?P<number>{_NUMBER})(?P<percent>\s*[%％])?(?![\d.])", tail)
        if not number:
            continue
        if re.match(r"\s*(?:到|至|~|～|±|以上|以下|左右)", tail[number.end():]):
            unknown = True
            continue
        # A numeric prefix followed by another field's unit is not a metric.
        if re.match(r"\s*(?:秒|分钟|小时|毫秒|轮|张|MB|GB|ms|s)", tail[number.end():], re.IGNORECASE):
            continue
        value = Decimal(number['number'])
        if field.name != "损失":
            if number['percent']:
                value /= 100
            if not 0 <= value <= 1:
                continue
        values.add(str(value.normalize()))
    return values, unknown


def _other_values(field: Field, clause: str) -> tuple[set[str], bool]:
    patterns = {
        "duration": r"(?:耗时|用时|训练时间|运行时间|用了)",
        "count": r"(?:图片数量|图像数量|图片总数|图像总数|数据集)",
        "deadline": r"截止(?:日期|时间)?",
        "exam_time": r"考试(?:开始时间|时间)?",
    }
    marker = re.search(patterns[field.kind], clause)
    if marker is None or _NONFACT.search(clause[:marker.start()]):
        return set(), False
    tail = clause[marker.end():]
    if _UNKNOWN.search(tail):
        return set(), True
    if field.kind == "duration":
        match = re.match(_LINK + rf"(?P<n>{_NUMBER})\s*(?P<u>毫秒|秒|分钟|小时|ms|s)", tail)
        if match:
            scale = {"毫秒": Decimal('.001'), "ms": Decimal('.001'), "秒": 1, "s": 1, "分钟": 60, "小时": 3600}
            number = Decimal(match['n']) * scale[match['u']]
            if number >= 0:
                return {str(number.normalize())}, False
    elif field.kind == "count":
        # The counted unit must be an image; a byte count or epoch number is not evidence.
        match = re.search(r"(?:共有|包含|含有|有|共|为|[:：=])\s*(\d+)\s*张(?:图片|图像)?", tail)
        if not match and "数据集" not in marker.group():
            match = re.match(_LINK + r"(\d+)(?=\s*(?:张|[（(]|$))", tail)
        if match:
            return {str(int(match[1]))}, False
    elif field.kind == "deadline":
        match = re.match(_LINK + r"((?:\d{4}年)?\d{1,2}月\d{1,2}日|\d{4}-\d{2}-\d{2})", tail)
        if match:
            return {match[1]}, False
    elif field.kind == "exam_time":
        match = re.match(_LINK + r"(?P<period>上午|下午|晚上)?(?P<h>\d{1,2})(?:点|:|：)(?P<m>\d{1,2})?(?:分)?", tail)
        if match:
            hour, minute = int(match['h']), int(match['m'] or 0)
            if match['period'] in {'下午', '晚上'} and hour < 12:
                hour += 12
            if 0 <= hour < 24 and 0 <= minute < 60:
                return {f"{hour:02}:{minute:02}"}, False
    return set(), False


def record_facts(question: str, hits: list[Hit]) -> FactDecision:
    fields = requested_fields(question)
    allowed = set(range(len(hits)))
    for field in fields:
        all_values, has_unknown = set(), False
        supporting = set()
        for index, hit in enumerate(hits):
            values, unknown = set(), False
            # A clause boundary prevents borrowing a neighboring field's number.
            for clause in re.split(r"[，,。；;\n]", hit.chunk.text):
                found, missing = (_metric_values(field, clause) if field.kind == "metric"
                                  else _other_values(field, clause))
                values.update(found)
                unknown |= missing
            all_values.update(values)
            has_unknown |= unknown
            if values:
                supporting.add(index)
        if len(all_values) > 1 or (all_values and has_unknown):
            return FactDecision([], "conflicting_records", field.label)
        if not all_values:
            return FactDecision([], "missing_requested_fact", field.label)
        allowed &= supporting
    if fields and not allowed:
        return FactDecision([], "incomplete_record", "、".join(f.label for f in fields))
    return FactDecision([h for i, h in enumerate(hits) if i in allowed])
