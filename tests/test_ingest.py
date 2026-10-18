"""Smoke tests for ingestion provenance (page numbers must survive)."""


def test_chunk_carries_page():
    from course_rag_assistant.ingest import Chunk

    c = Chunk(text="贝叶斯定理", doc="notes.md", page=12, section="4.4 朴素贝叶斯")
    assert c.page == 12
    assert c.doc == "notes.md"
