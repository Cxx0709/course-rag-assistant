"""Ingestion: extract course documents into page-preserving chunks.

Every chunk carries its provenance: source file, page number, section heading.
Page numbers are the foundation of the "precise to the page" citation story,
so extractors must never drop them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    text: str
    doc: str          # source file name
    page: int         # 1-based page number
    section: str = ""  # nearest heading, if known
    concepts: list[str] = field(default_factory=list)  # concept tags, filled later


def extract_pdf(path: Path) -> list[Chunk]:
    """Extract paragraphs from a PDF, one chunk per paragraph block.

    TODO (week 1): implement with PyMuPDF; merge over-split lines, keep
    page boundaries, skip headers/footers, detect headings for `section`.
    """
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RuntimeError("PDF ingestion requires the pymupdf package") from exc

    chunks: list[Chunk] = []
    with fitz.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            blocks = _pdf_text_blocks(page, fitz)
            page_height = page.rect.height
            section = ""
            pending: list[str] = []

            for x0, y0, x1, y1, text, *_ in blocks:
                text = " ".join(text.split())
                if not text or (y1 < page_height * 0.06) or (y0 > page_height * 0.94):
                    continue
                if _looks_like_heading(text):
                    _flush_pdf_pending(chunks, pending, path.name, page_number, section)
                    section = text
                    continue
                # Group adjacent blocks up to a bounded size. This keeps a
                # paragraph and its formula together without making the whole
                # page a single retrieval unit.
                if pending and sum(len(item) for item in pending) + len(text) > 900:
                    _flush_pdf_pending(chunks, pending, path.name, page_number, section)
                pending.append(text)
            _flush_pdf_pending(chunks, pending, path.name, page_number, section)
    return chunks


def _pdf_text_blocks(page, fitz):
    """Remove exact text overlays used to simulate bold, using glyph positions.

    Equal wording elsewhere on the page is retained. Unaffected blocks use the
    existing extraction verbatim, preserving frozen corpus/index alignment.
    """
    blocks = page.get_text("blocks")
    layout = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)
    corrected = {}
    for block in layout["blocks"]:
        if block.get("type") != 0:
            continue
        seen, rows, duplicated = set(), {}, False
        for line in block["lines"]:
            for span in line["spans"]:
                key = (span["text"], tuple(span["bbox"]), span["font"], span["size"])
                if key in seen:
                    duplicated = True
                    continue
                seen.add(key)
                baseline = (round(span["origin"][1], 2), tuple(line["dir"]))
                rows.setdefault(baseline, []).append(span)
        if duplicated:
            # Overlay rendering can split one printed line into several API
            # lines. Reunite spans on the same baseline before joining lines.
            corrected[block["number"]] = "\n".join(
                "".join(s["text"] for s in sorted(spans, key=lambda s: s["origin"][0]))
                for _, spans in sorted(rows.items()))
    return [(*b[:4], corrected.get(b[5], b[4]), *b[5:]) for b in blocks]


def _flush_pdf_pending(chunks: list[Chunk], pending: list[str], doc: str, page: int, section: str) -> None:
    if pending:
        chunks.append(Chunk("\n".join(pending), doc, page, section))
        pending.clear()


def extract_pptx(path: Path) -> list[Chunk]:
    """Extract slide text; slide number plays the role of `page`.

    TODO (week 1): implement with python-pptx; keep speaker notes if present.
    """
    try:
        from pptx import Presentation
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PPTX ingestion requires python-pptx; install it to read slides") from exc

    chunks: list[Chunk] = []
    for slide_number, slide in enumerate(Presentation(path).slides, start=1):
        texts = [shape.text.strip() for shape in slide.shapes if hasattr(shape, "text") and shape.text.strip()]
        if texts:
            chunks.append(Chunk("\n".join(texts), path.name, slide_number))
    return chunks


def extract_markdown(path: Path) -> list[Chunk]:
    """Extract markdown sections; heading hierarchy fills `section`.

    TODO (week 1): implement; page numbers are N/A, use section anchors.
    """
    # Markdown has no physical pages.  Page zero is an explicit sentinel; callers
    # should cite the section anchor instead of rendering it as a PDF page.
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    chunks: list[Chunk] = []
    headings: list[str] = []
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            chunks.append(Chunk(text, path.name, 0, " / ".join(headings)))
        buffer.clear()

    for line in lines:
        fence = re.match(r"^\s*(```+|~~~+)", line)
        if fence:
            in_fence = not in_fence
            buffer.append(line.rstrip())
            continue
        heading = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading and not in_fence:
            flush()
            level, title = len(heading.group(1)), heading.group(2).strip()
            headings[:] = headings[: level - 1]
            headings.append(title)
            continue
        if not line.strip() and not in_fence:
            flush()
        elif line.strip() or buffer:
            buffer.append(line.rstrip())
    flush()
    return chunks


def _looks_like_heading(text: str) -> bool:
    if len(text) > 80 or any(mark in text for mark in "，,；;"):
        return False
    return text.endswith(":") or bool(re.match(r"^(?:\d+(?:\.\d+)*\s+)?[^。！？.!?]{2,50}$", text))


def ingest_dir(root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() == ".pdf":
            chunks.extend(extract_pdf(path))
        elif path.suffix.lower() == ".pptx":
            chunks.extend(extract_pptx(path))
        elif path.suffix.lower() == ".md":
            chunks.extend(extract_markdown(path))
    return chunks


if __name__ == "__main__":
    import sys

    chunks = ingest_dir(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/samples"))
    print(f"ingested {len(chunks)} chunks")
    for index, chunk in enumerate(chunks, start=1):
        location = f"section={chunk.section or '-'}"
        if chunk.page:
            location = f"page={chunk.page} {location}"
        print(f"[{index}] {chunk.doc} {location}\n{chunk.text}\n")
