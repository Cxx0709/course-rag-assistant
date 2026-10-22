"""Score generated answer JSONL files against a labeled question set."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def score(labels_path: Path, answers_path: Path) -> dict:
    labels = {item["question"]: item for item in load_jsonl(labels_path)}
    rows: list[dict] = []
    for answer in load_jsonl(answers_path):
        label = labels.get(answer["question"], {})
        text = answer.get("answer", "")
        points = [str(point) for point in label.get("answer_points", [])]
        matched = [point for point in points if point.casefold() in text.casefold()]
        citations = len(re.findall(r"\[[^,\]]+,\s*(?:p\.\d+|section=[^\]]+)\]", text))
        rows.append({
            "question": answer["question"],
            "point_coverage": len(matched) / max(len(points), 1),
            "matched_points": matched,
            "answer_points": points,
            "citations": citations,
            "abstained": bool(answer.get("abstained")),
            "unsupported_claims": len(answer.get("unsupported_claims", [])),
        })
    count = len(rows)
    return {
        "questions": count,
        "answer_point_coverage": sum(row["point_coverage"] for row in rows) / max(count, 1),
        "citation_rate": sum(row["citations"] > 0 for row in rows) / max(count, 1),
        "abstention_rate": sum(row["abstained"] for row in rows) / max(count, 1),
        "unsupported_claims": sum(row["unsupported_claims"] for row in rows),
        "details": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score generated answers against JSONL labels")
    parser.add_argument("labels", type=Path)
    parser.add_argument("answers", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score(args.labels, args.answers)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
