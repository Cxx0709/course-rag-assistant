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
from .compare import _named_statement, _topic_heading
from .ingest import Chunk
from .prerequisites import is_prerequisite_statement
from .quiz_content import chapter_number, cloze_prompt, definition_concept, statement_rank, teaching_statement, usable_concept
from .retrieval import Hit


def make_questions(chunks: list[Chunk], count: int = 5, *, target_concept: str | None = None) -> list[dict]:
    candidates = []
    seen = set()
    chapters = {}
    for chunk in chunks:
        number = chapter_number(chunk.section)
        if number is not None:
            chapters[chunk.doc] = number
        # Prefer explicit tags or the innermost heading. Headingless PDFs can
        # contribute simple, explicitly phrased Chinese definitions.
        concepts = chunk.concepts or [chunk.section.rsplit(" / ", 1)[-1]]
        for sentence in _source_units(chunk):
            # A declared reading order is metadata for the review path, not a
            # lesson assessing knowledge of the concept itself.
            if is_prerequisite_statement(sentence) or not teaching_statement(sentence):
                continue
            defined = definition_concept(sentence)
            definitions = [defined] if defined else []
            target = [target_concept] if target_concept and (not defined or defined == target_concept) and (
                _named_statement(target_concept, sentence) or _topic_heading(target_concept, chunk.section)
                or target_concept in chunk.concepts) else []
            for concept in [*target, *definitions, *concepts]:
                concept = re.sub(r"^\d+(?:\.\d+)*[.、\s]*", "", concept).strip()
                if not usable_concept(concept, defined=concept == defined):
                    continue
                if not 2 <= len(concept) <= 40 or concept not in sentence:
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
                                   "doc": chunk.doc, "page": chunk.page, "section": chunk.section,
                                   "_chapter": chapters.get(chunk.doc)})
                break
    # Prefer substantive statements in every mode, not merely the first
    # sentence mentioning a heading. Stable sorting retains source order ties.
    candidates.sort(key=lambda c: statement_rank(c["concept"], c["sentence"]))
    questions = []
    used_concepts = set()
    for candidate in candidates:
        concept, sentence = candidate["concept"], candidate["sentence"]
        if target_concept and concept != target_concept:
            continue
        if concept in used_concepts:
            continue
        # Distractors that already appear in the sentence give away or confuse
        # the task. The prompt explicitly asks for the original source wording.
        # Prefer the same recognized numbered section, then defined concepts
        # in nearby 20-page bands. This ranks candidates; scope remains strict.
        nearby = sorted(candidates, key=lambda c: (
            c["doc"] != candidate["doc"],
            candidate["_chapter"] is not None and c["_chapter"] != candidate["_chapter"],
            abs(c["page"] - candidate["page"]) // 20,
            statement_rank(c["concept"], c["sentence"]), abs(c["page"] - candidate["page"]),
            c["section"] != candidate["section"], c["concept"]))
        concepts = dict.fromkeys(c["concept"] for c in nearby)
        distractors = [c for c in concepts if c != concept and c not in sentence
                       and c not in concept and concept not in c][:3]
        if not distractors:
            continue
        options = sorted([concept, *distractors],
                         key=lambda c: hashlib.sha256((sentence + c).encode()).hexdigest())
        questions.append({**{k: v for k, v in candidate.items() if k != "_chapter"}, "id": str(len(questions) + 1),
                          "prompt": cloze_prompt(concept, sentence),
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
