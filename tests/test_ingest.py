"""Tests for ingestion provenance and the Markdown chunk contract."""

from pathlib import Path


def test_chunk_carries_page():
    from course_rag_assistant.ingest import Chunk

    c = Chunk(text="贝叶斯定理", doc="notes.md", page=12, section="4.4 朴素贝叶斯")
    assert c.page == 12
    assert c.doc == "notes.md"


def test_markdown_sections_and_paragraphs(tmp_path: Path):
    from course_rag_assistant.ingest import extract_markdown

    source = tmp_path / "notes.md"
    source.write_text(
        "# Course\n\n## Bayes\n\nFirst paragraph.\n\nSecond paragraph.\n\n"
        "```python\nprint('kept')\n```\n",
        encoding="utf-8",
    )
    chunks = extract_markdown(source)

    assert [chunk.text for chunk in chunks] == [
        "First paragraph.",
        "Second paragraph.",
        "```python\nprint('kept')\n```",
    ]
    assert all(chunk.page == 0 for chunk in chunks)
    assert all(chunk.section == "Course / Bayes" for chunk in chunks)


def test_pdf_bold_overlays_are_deduplicated_by_position_not_wording(tmp_path):
    import pymupdf

    from course_rag_assistant.ingest import _pdf_text_blocks, extract_pdf

    path = tmp_path / "overlay.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        sentence = "Training data is used to train the model."
        page.insert_text((72, 120), sentence)
        page.insert_text((72, 120), sentence)
        page.insert_text((72, 180), sentence)
        blocks = _pdf_text_blocks(page, pymupdf)
        assert sum(b[4].count(sentence) for b in blocks) == 2
        pdf.save(path)
    assert sum(c.text.count(sentence) for c in extract_pdf(path)) == 2


def test_overlay_fragments_on_one_baseline_are_joined_without_invented_spaces():
    from types import SimpleNamespace

    from course_rag_assistant.ingest import _pdf_text_blocks

    def span(text, x):
        return {"text": text, "bbox": (x, 100, x + 20, 115), "origin": (x, 112), "font": "test", "size": 12}
    fragments = [span("训练", 72), span("训练", 72), span("集", 92), span("集", 92)]
    layout = {"blocks": [{"type": 0, "number": 0,
                          "lines": [{"dir": (1, 0), "spans": [s]} for s in fragments]}]}
    class Page:
        def get_text(self, kind, **kwargs):
            return [(72, 100, 112, 115, "训练\n训练\n集\n集", 0, 0)] if kind == "blocks" else layout
    assert _pdf_text_blocks(Page(), SimpleNamespace(TEXTFLAGS_TEXT=0))[0][4] == "训练集"
