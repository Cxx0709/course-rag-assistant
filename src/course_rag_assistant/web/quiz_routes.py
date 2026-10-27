"""Source-backed practice and review HTTP routes."""
from functools import lru_cache

from fastapi import APIRouter, HTTPException

from ..quiz import QuizStore
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
            return quiz_store().create([c for d in documents for c in d.chunks],
                                  [d.id for d in documents], payload.count)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

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
        gaps = quiz_sources(quiz_store().review([d.id for d in documents]), documents)
        return {"gaps": gaps, "learning_plans": learning_plans(documents, [g["concept"] for g in gaps]),
                "note": "复习清单根据原文识记练习中的错题整理，不代表完整的知识掌握评估。"}

    return router
