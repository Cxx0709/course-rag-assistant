"""Evaluation: prove the system retrieves and cites correctly.

Labeled set format (eval/labeled-questions.jsonl, one JSON object per line):
    {"question": "...", "gold_pages": ["notes.md#sec"], "answer_points": ["..."]}

Metrics:
- retrieval: recall@k, MRR over gold chunks
- citation: precision of [doc, p.N] citations against gold passages
- abstention: on unanswerable questions, did it refuse instead of hallucinating?
- ablation: dense-only vs hybrid vs +rerank on the same set (the portfolio's
  most persuasive table)
"""
from __future__ import annotations

import json
from pathlib import Path


def load_labeled(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / max(len(gold), 1)


def run_eval(labeled_path: Path) -> dict:
    """TODO (week 4): wire retriever + answerer, compute all metrics."""
    questions = load_labeled(labeled_path)
    raise NotImplementedError(f"{len(questions)} questions loaded; eval not implemented yet")


if __name__ == "__main__":
    import sys

    print(run_eval(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/labeled-questions.jsonl")))
