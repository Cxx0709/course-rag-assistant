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
