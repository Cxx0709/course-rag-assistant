"""Source-backed concept cloze quizzes and persistent review history.

These measure recognition of a literal source statement, not general mastery.
No generated claim or semantic grading is presented as a verified fact.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from .answer import CitedAnswer, _source_units, format_citation, verify_citations
from .ingest import Chunk
from .retrieval import Hit


def make_questions(chunks: list[Chunk], count: int = 5) -> list[dict]:
    candidates = []
    seen = set()
    for chunk in chunks:
        # Prefer explicit tags or the innermost heading. Headingless PDFs can
        # contribute simple, explicitly phrased Chinese definitions.
        concepts = chunk.concepts or [chunk.section.rsplit(" / ", 1)[-1]]
        for sentence in _source_units(chunk):
            definitions = re.findall(r"^([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9]{1,19}?)(?:是指|指的是|是一种|是用来)", sentence)
            for concept in [*concepts, *definitions]:
                concept = re.sub(r"^\d+(?:\.\d+)*[.、\s]*", "", concept).strip()
                if not 2 <= len(concept) <= 30 or concept not in sentence:
                    continue
                if not 12 <= len(sentence) <= 400 or re.match(r"^(?:它|这|其|该|上述|因此|其中)", sentence):
                    continue
                hit = Hit(chunk, 1, "quiz")
                citation = format_citation(hit)
                verified = verify_citations(CitedAnswer(f"{sentence} {citation}", [hit], []))
                if verified.abstention_reason or verified.unsupported_claims:
                    continue
                key = re.sub(r"\s+", "", sentence)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({"concept": concept, "sentence": sentence, "citation": citation,
                                   "doc": chunk.doc, "page": chunk.page, "section": chunk.section})
                break
    concepts = sorted({c["concept"] for c in candidates})
    questions = []
    used_concepts = set()
    for candidate in candidates:
        concept, sentence = candidate["concept"], candidate["sentence"]
        if concept in used_concepts:
            continue
        # Distractors that already appear in the sentence give away or confuse
        # the task. The prompt explicitly asks for the original source wording.
        distractors = [c for c in concepts if c != concept and c not in sentence
                       and c not in concept and concept not in c][:3]
        if not distractors:
            continue
        options = sorted([concept, *distractors],
                         key=lambda c: hashlib.sha256((sentence + c).encode()).hexdigest())
        questions.append({**candidate, "id": str(len(questions) + 1),
                          "prompt": "按资料原文补全空缺：" + sentence.replace(concept, "____"),
                          "options": options, "correct_index": options.index(concept)})
        used_concepts.add(concept)
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

    def create(self, chunks: list[Chunk], document_ids: list[str], count: int) -> dict:
        questions = make_questions(chunks, count)
        if not questions:
            raise ValueError("所选资料没有足够的完整概念句或可区分的选项。请添加带章节标题的课程笔记。")
        quiz_id = uuid4().hex
        payload = {"document_ids": sorted(set(document_ids)), "questions": questions}
        with closing(self.connect()) as db, db:
            db.execute("INSERT INTO quizzes VALUES (?, ?)", (quiz_id, json.dumps(payload, ensure_ascii=False)))
        return {"quiz_id": quiz_id, "kind": "source_cloze", "questions": [
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
                "total": len(results), "results": results}

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
