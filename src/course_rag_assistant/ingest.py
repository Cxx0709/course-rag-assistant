"""Ingestion: extract course documents into page-preserving chunks.

Every chunk carries its provenance: source file, page number, section heading.
Page numbers are the foundation of the "precise to the page" citation story,
so extractors must never drop them.
"""
from __future__ import annotations

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
    raise NotImplementedError


def extract_pptx(path: Path) -> list[Chunk]:
    """Extract slide text; slide number plays the role of `page`.

    TODO (week 1): implement with python-pptx; keep speaker notes if present.
    """
    raise NotImplementedError


def extract_markdown(path: Path) -> list[Chunk]:
    """Extract markdown sections; heading hierarchy fills `section`.

    TODO (week 1): implement; page numbers are N/A, use section anchors.
    """
    raise NotImplementedError


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
