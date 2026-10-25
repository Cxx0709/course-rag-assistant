"""Conservative navigation-text detection without rewriting indexed sources.

Only explicit dotted TOC leaders, navigation headings, labelled page markers and
page/chapter running titles are recognized. Ordinary numbered headings stay.
"""
from __future__ import annotations

import re

_TOC_ENTRY = re.compile(r"^.{1,180}?(?:[.．·…]\s*){4,}(?:\d{1,5}|[ivxlcdm]{1,8})\s*$", re.IGNORECASE)
_NAVIGATION_HEADING = re.compile(r"(?:目\s*录|table\s+of\s+contents|contents)", re.IGNORECASE)
_PAGE_LABEL = re.compile(
    r"(?:第\s*\d+\s*页(?:\s*[/／]\s*共\s*\d+\s*页)?|page\s+\d+(?:\s+of\s+\d+)?)",
    re.IGNORECASE,
)
_TITLE = r"[A-Za-z\u4e00-\u9fff][A-Za-z\u4e00-\u9fff0-9\s（）()、，\-‐–]{1,70}"
_RUNNING_TITLE = re.compile(
    # These require a distinct page number and a punctuated chapter/section
    # number. A heading alone, equation or ordinary prose must not match.
    rf"(?:\d{{1,5}}\s+\d+(?:\.\d+)*\.\s+{_TITLE}|\d+(?:\.\d+)+\.\s+{_TITLE}\s+\d{{1,5}})"
)


def navigation_reason(text: str) -> str | None:
    """Classify only chunks/sentences composed entirely of explicit navigation."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "empty"
    reasons = []
    for line in lines:
        if _TOC_ENTRY.fullmatch(line):
            reasons.append("toc_entry")
        elif _NAVIGATION_HEADING.fullmatch(line):
            reasons.append("navigation_heading")
        elif _PAGE_LABEL.fullmatch(line):
            reasons.append("page_label")
        elif _RUNNING_TITLE.fullmatch(line):
            reasons.append("running_title")
        else:
            return None
    return "toc_entry" if "toc_entry" in reasons else reasons[0]
