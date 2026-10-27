"""Source-backed practice and review HTTP routes."""
from functools import lru_cache

from fastapi import APIRouter, HTTPException

from ..quiz import QuizStore
from ..quiz_scope import practice_scope, review_matches, section_ranges
from .prerequisites import learning_plans
from .schemas import QuizGradeRequest, QuizRequest


def create_quiz_router(library):
    router = APIRouter()

    @lru_cache(maxsize=1)
    def quiz_store():
        return QuizStore(library.root / "quiz.sqlite3")

    def quiz_sources(rows, documents):
        by_source = {d.source: d for d in documents}
        for row in rows:
            document = by_source.get(row["doc"])
            if document:
                row["document_name"] = document.name
                row["source_url"] = f"/api/documents/{document.id}/file" + (
                    f"#page={row['page']}" if row["page"] else "")
        return rows

    @router.post("/api/quiz", status_code=201)
    def create_quiz(payload: QuizRequest):
        documents = library.selected(payload.document_ids)
        try:
            chunks, scope = practice_scope(documents, target_concept=payload.target_concept, section_id=payload.section_id)
            return quiz_store().create(chunks, [d.id for d in documents], payload.count,
                                       target_concept=scope["target_concept"], scope=scope)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/api/quiz/sections")
    def chapters(payload: QuizRequest):
        documents = library.selected(payload.document_ids)
        return {"sections": [{k: v for k, v in s.items() if k not in {"start", "end"}}
                             for d in documents for s in section_ranges(d)],
                "note": "PDF 范围按已识别的编号标题划分，请核对起止页；未识别标题的内容可能缺失或归入相邻章节。"}

    @router.post("/api/quiz/{quiz_id}/grade")
    def grade_quiz(quiz_id: str, payload: QuizGradeRequest):
        try:
            quiz = quiz_store().load(quiz_id)
            documents = library.selected(quiz["document_ids"])
            result = quiz_store().grade(quiz_id, payload.choices)
        except KeyError as exc:
            raise HTTPException(404, exc.args[0]) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        quiz_sources(result["results"], documents)
        result["learning_plans"] = learning_plans(documents, [r["concept"] for r in result["results"] if not r["correct"]])
        return result

    @router.post("/api/review")
    def review_quiz(payload: QuizRequest):
        documents = library.selected(payload.document_ids)
        try:
            chunks, scope = practice_scope(documents, target_concept=payload.target_concept, section_id=payload.section_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        gaps = quiz_sources([g for g in quiz_store().review([d.id for d in documents]) if review_matches(g, chunks, scope)], documents)
        return {"gaps": gaps, "scope": scope, "learning_plans": learning_plans(documents, [g["concept"] for g in gaps]),
                "note": "复习清单根据原文识记练习中的错题整理，不代表完整的知识掌握评估。"}

    return router
