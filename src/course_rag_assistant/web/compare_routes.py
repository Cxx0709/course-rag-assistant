"""Independent retrieval per selected source for concept comparison."""
from fastapi import APIRouter, HTTPException

from ..alignment import aligned_columns
from ..compare import compare_excerpts
from .schemas import CompareRequest


def create_compare_router(library):
    router = APIRouter()

    @router.post("/api/compare")
    def compare(payload: CompareRequest):
        concept = payload.concept.strip()
        if len(concept) < 2 or len(set(payload.document_ids)) != 2:
            raise HTTPException(400, "请输入一个概念，并选择两份不同的资料。")
        documents = library.selected(payload.document_ids)
        try:
            result = aligned_columns(concept, [library.retriever((d.id,)) for d in documents], sense=payload.sense)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        columns = result["columns"]
        for index, (document, excerpts) in enumerate(zip(documents, columns), start=1):
            for excerpt in excerpts:
                excerpt.update(id=f"{index}-{excerpt['id']}", document_id=document.id,
                               document_name=document.name, type=document.public()["type"],
                               source_url=f"/api/documents/{document.id}/file" + (
                                   f"#page={excerpt['page']}" if excerpt["page"] else ""))
        found = sum(bool(c) for c in columns)
        pending = result.get("needs_disambiguation", False)
        status = ("needs_disambiguation" if pending else "needs_review" if found == 2 and not result["comparable"]
                  else "ready" if found == 2 else "partial" if found else "not_found")
        return {"concept": concept, "status": status, "alignment": result["alignment"],
                "documents": [{"id": d.id, "name": d.name, "label": f"资料 {i + 1}",
                               "status": "pending" if pending else "found" if columns[i] else "not_found"}
                              for i, d in enumerate(documents)],
                "rows": [] if pending else compare_excerpts(columns, comparable=result["comparable"]),
                "note": "仅比较本次检出的原文。分类用于辅助阅读，标记仅表示每项首条引文的文字变化；含义是否相同或冲突需结合上下文判断。"}

    return router
