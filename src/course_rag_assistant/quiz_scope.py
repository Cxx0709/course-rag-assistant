"""Document-scoped chapter boundaries and exact practice filters."""
from __future__ import annotations

import hashlib
import re

from .compare import normalized


def section_ranges(document):
    """Use numbered PDF headings or Markdown heading paths, never substring ranges.

    PDF boundaries follow the next recognized peer/ancestor heading. Missing
    headings cannot be reconstructed; expose page bounds for user inspection.
    """
    chunks = document.chunks
    events, seen = [], set()
    pdf = document.source.lower().endswith(".pdf")
    for index, chunk in enumerate(chunks):
        parts = chunk.section.strip().split(" / ")
        titles = [chunk.section.strip()] if pdf else [" / ".join(parts[:i + 1]) for i in range(len(parts))]
        for title in titles:
            if not title or title in seen:
                continue
            number = re.match(r"^(\d+(?:\.\d+)*)\s+\S", title)
            if pdf and not number:
                continue
            seen.add(title)
            events.append((index, title, tuple(map(int, number[1].split("."))) if number else ()))
    entries = []
    for position, (start, title, number) in enumerate(events):
        end = len(chunks)
        for next_start, next_title, next_number in events[position + 1:]:
            descendant = (len(next_number) > len(number) and next_number[:len(number)] == number) if pdf else next_title.startswith(title + " / ")
            if not descendant:
                end = next_start
                break
        identifier = hashlib.sha256(f"{document.source}\0{start}\0{title}".encode()).hexdigest()[:24]
        entries.append({"id": identifier, "document_id": document.id, "document_name": document.name,
                        "title": title, "page_from": chunks[start].page,
                        "page_to": max(c.page for c in chunks[start:end]), "chunks": end - start,
                        "start": start, "end": end})
    return entries


def practice_scope(documents, *, target_concept=None, section_id=None):
    target = target_concept.strip() if target_concept is not None else None
    if target is not None and not 2 <= len(target) <= 40:
        raise ValueError("目标概念须为 2 至 40 个字符，请填写概念名称。")
    section = None
    if section_id is not None:
        section = next((s for d in documents for s in section_ranges(d) if s["id"] == section_id), None)
        if section is None:
            raise ValueError("所选章节不属于当前资料，请重新选择章节。")
        doc = next(d for d in documents if d.id == section["document_id"])
        chunks = doc.chunks[section["start"]:section["end"]]
    else:
        chunks = [c for d in documents for c in d.chunks]
    public = {"target_concept": target,
              "section": {k: v for k, v in section.items() if k not in {"start", "end"}} if section else None,
              "documents": [{"id": d.id, "name": d.name} for d in documents]}
    return chunks, public


def review_matches(row, chunks, scope):
    if scope["target_concept"] and row["concept"] != scope["target_concept"]:
        return False
    if scope["section"]:
        return any((row["doc"], row["page"], row["section"]) == (c.doc, c.page, c.section)
                   and normalized(row["sentence"]) in normalized(c.text) for c in chunks)
    return True
