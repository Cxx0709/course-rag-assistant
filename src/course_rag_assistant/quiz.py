"""Source-backed concept cloze quizzes and persistent review history.

These measure recognition of a literal source statement, not general mastery.
No generated claim or semantic grading is presented as a verified fact.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from .ingest import Chunk
from .quiz_candidates import (
    candidate_prompt,
    compatible_options,
    extract_candidates,
    prompt_rejection,
)
from .quiz_content import explicit_contrast, statement_rank


def make_questions(chunks: list[Chunk], count: int = 5, *, target_concept: str | None = None) -> list[dict]:
    def rank(candidate):
        return 0 if candidate.origin in {"contextual", "source_relation", "definition"} else statement_rank(candidate.concept, candidate.sentence)

    # Invalid stems cannot supply distractors either. Extraction is independent
    # of the requested target; filtering cannot turn a substring into a name.
    eligible = []
    for candidate in sorted(extract_candidates(chunks), key=rank):
        contrast = explicit_contrast(candidate.concept, candidate.sentence) if (
            candidate.list_group is None and candidate.relation_group is None) else None
        prompt = candidate_prompt(candidate)
        if not prompt_rejection(candidate, prompt):
            eligible.append((candidate, prompt, contrast))
    questions = []
    used_concepts = set()
    for candidate, prompt, contrast in eligible:
        concept, sentence = candidate.concept, candidate.sentence
        if target_concept and concept != target_concept:
            continue
        if concept in used_concepts:
            continue
        nearby = sorted((c for c, _, _ in eligible if compatible_options(candidate, c)), key=lambda c: (
            abs(c.page - candidate.page) // 20, rank(c), abs(c.page - candidate.page),
            c.section != candidate.section, c.concept))
        concepts = []
        option_names = set()
        for other in nearby:
            names = {other.concept, *other.synonyms}
            if not names & option_names:
                concepts.append(other.concept)
                option_names.update(names)
        visible = prompt if candidate.relation_group is not None else sentence
        distractors = [c for c in concepts if c != concept and c not in visible
                       and c not in concept and concept not in c][:3]
        if contrast:
            if contrast[1] in candidate.synonyms:
                continue
            distractors = [contrast[1]]
        if not distractors:
            continue
        options = sorted([concept, *distractors],
                         key=lambda c: hashlib.sha256((sentence + c).encode()).hexdigest())
        questions.append({**candidate.evidence(), "id": str(len(questions) + 1),
                          "prompt": prompt, "options": options, "correct_index": options.index(concept)})
        used_concepts.update((concept, *candidate.synonyms))
        if len(questions) >= count:
            break
    return questions


class QuizStore:
    def __init__(self, path: Path):
        self.path = path
        with closing(self.connect()) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS quizzes (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            db.execute("""CREATE TABLE IF NOT EXISTS attempts (
                quiz_id TEXT, question_id TEXT, choice INTEGER, correct INTEGER,
                PRIMARY KEY (quiz_id, question_id))""")

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def create(self, chunks: list[Chunk], document_ids: list[str], count: int, *, target_concept=None, scope=None) -> dict:
        questions = make_questions(chunks, count, target_concept=target_concept)
        if not questions:
            raise ValueError("当前范围没有足够的完整概念句或可区分的选项。请调整概念或章节，或补充课程笔记；不会从范围外凑题。")
        quiz_id = uuid4().hex
        payload = {"document_ids": sorted(set(document_ids)), "questions": questions, "scope": scope}
        with closing(self.connect()) as db, db:
            db.execute("INSERT INTO quizzes VALUES (?, ?)", (quiz_id, json.dumps(payload, ensure_ascii=False)))
        return {"quiz_id": quiz_id, "kind": "source_cloze", "scope": scope, "requested_count": count,
                "note": f"当前范围生成 {len(questions)} / {count} 道原文识记题。同一概念不重复出题；题数不足时不扩大范围。",
                "questions": [
            {k: q[k] for k in ("id", "prompt", "options")} for q in questions]}

    def load(self, quiz_id: str) -> dict:
        with closing(self.connect()) as db:
            row = db.execute("SELECT payload FROM quizzes WHERE id = ?", (quiz_id,)).fetchone()
        if not row:
            raise KeyError("练习不存在，请重新出题。")
        return json.loads(row[0])

    def grade(self, quiz_id: str, choices: dict[str, int]) -> dict:
        quiz = self.load(quiz_id)
        if set(choices) != {q["id"] for q in quiz["questions"]}:
            raise ValueError("请完成所有题目后提交，且不要提交练习外的题号。")
        results = []
        for question in quiz["questions"]:
            choice = choices[question["id"]]
            if type(choice) is not int or not 0 <= choice < len(question["options"]):
                raise ValueError("选项编号无效。")
            correct = choice == question["correct_index"]
            results.append({**question, "choice": choice, "correct": correct})
        with closing(self.connect()) as db, db:
            # A retry of the same POST must not inflate wrong-answer counts.
            # First submitted choices remain immutable; practice again via a new quiz.
            db.executemany("INSERT OR IGNORE INTO attempts VALUES (?, ?, ?, ?)",
                           [(quiz_id, q["id"], q["choice"], int(q["correct"])) for q in results])
            saved = dict(db.execute("SELECT question_id, choice FROM attempts WHERE quiz_id = ?", (quiz_id,)))
        for q in results:
            q["choice"] = saved[q["id"]]
            q["correct"] = q["choice"] == q["correct_index"]
        return {"quiz_id": quiz_id, "score": sum(q["correct"] for q in results),
                "total": len(results), "results": results, "scope": quiz.get("scope")}

    def review(self, document_ids: list[str]) -> list[dict]:
        selected = set(document_ids)
        with closing(self.connect()) as db:
            rows = db.execute("""SELECT q.payload, a.question_id, a.correct
                FROM attempts a JOIN quizzes q ON q.id = a.quiz_id ORDER BY a.rowid""").fetchall()
        gaps = {}
        for raw, question_id, correct in rows:
            payload = json.loads(raw)
            if not set(payload["document_ids"]) <= selected:
                continue
            q = next(q for q in payload["questions"] if q["id"] == question_id)
            key = (q["doc"], q["page"], q["section"], q["concept"])
            item = gaps.setdefault(key, {k: q[k] for k in
                                       ("concept", "doc", "page", "section", "sentence", "citation")})
            item["attempts"] = item.get("attempts", 0) + 1
            item["mistakes"] = item.get("mistakes", 0) + (not correct)
            item["latest_correct"] = bool(correct)
        return sorted([g for g in gaps.values() if g["mistakes"] and not g["latest_correct"]],
                      key=lambda g: (-g["mistakes"], g["concept"]))
