"""Export an unreviewed JSONL packet from a completed quality report.

Human judgments are intentionally blank; automatic scores are not human labels.
Existing review files are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def review_rows(report, report_sha256):
    if report.get("schema_version", 0) < 3 or not report.get("completed"):
        raise ValueError("A completed schema v3 quality report is required")
    for mode, results in report["results"].items():
        for split in ("answerable", "unanswerable"):
            for row in results[split]["details"]:
                yield {
                    "review_id": f"{mode}/{split}/{row['id']}",
                    "report_sha256": report_sha256,
                    "benchmark": report["benchmark"], "run_scope": report["run_scope"],
                    "mode": mode, "split": split, "question": row["question"],
                    "expected_abstention": split == "unanswerable",
                    "answer": row["answer"], "raw_model_output": row["raw_model_output"],
                    "generation_status": row["generation_status"],
                    "selected_sentence_ids": row.get("selected_sentence_ids", []),
                    "abstention_reason": row["abstention_reason"],
                    "unsupported_claims": row["unsupported_claims"],
                    "expected_answer_points": row["expected_answer_points"],
                    "gold_pages": row["gold_pages"],
                    "evidence": row["evidence"],
                    "review": {
                        "status": "pending", "reviewer": None, "reviewed_at": None,
                        "answer_correct": None, "answer_complete": None,
                        "citations_support_claims": None, "relevant_to_question": None,
                        "abstention_appropriate": None, "notes": None,
                    },
                }


def export_review(report_path, output_path):
    content = report_path.read_bytes()
    report = json.loads(content)
    rows = list(review_rows(report, hashlib.sha256(content).hexdigest()))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Never erase the reviewer's work on a repeat invocation.
    with output_path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = export_review(args.report, args.output)
    print(f"Exported {count} pending reviews to {args.output}")


if __name__ == "__main__":
    main()
