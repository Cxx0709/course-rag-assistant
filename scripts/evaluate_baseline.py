"""Verify frozen inputs and repeat offline evaluation; fail on drift/regression.

Run from the repository root. The manifest is deliberately checked, never
automatically rewritten. Thresholds describe a development regression floor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from course_rag_assistant.answer import CitedAnswer, answer, verify_citations
from course_rag_assistant.dense import NumpyVectorIndex
from course_rag_assistant.evaluate import evaluate_retriever, load_labeled
from course_rag_assistant.ingest import Chunk, ingest_dir
from course_rag_assistant.retrieval import Hit, HybridRetriever
from scripts.evaluate_quality import CachedEmbedder, RankedResults, answer_metrics


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_predictions(result):
    return [{k: row[k] for k in ("id", "answer", "abstention_reason", "selected_sentence_ids")}
            for row in result["details"]]


def check_inputs(manifest):
    for path, expected in manifest["inputs_sha256"].items():
        if fingerprint(path) != expected:
            raise ValueError(f"Frozen baseline input changed: {path}")


def evaluate_integrity(path):
    rows = []
    for case in load_labeled(Path(path)):
        hits = [Hit(Chunk(**p), 1, "bm25") for p in case["passages"]]
        result = (verify_citations(CitedAnswer(case["claim"], hits, [])) if "claim" in case
                  else answer(case["question"], hits))
        passed = (result.text == case["expected_text"] if "expected_text" in case
                  else result.abstention_reason == case["expected_reason"])
        rows.append({"id": case["id"], "passed": passed, "answer": result.text,
                     "reason": result.abstention_reason})
    return {"cases": len(rows), "passed": sum(r["passed"] for r in rows), "details": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("eval/baseline-v12-manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("eval/reports/baseline-v12.json"))
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("At least two repeats are required to measure stability")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    check_inputs(manifest)
    labels = load_labeled(Path(manifest["labels"]))
    negatives = load_labeled(Path(manifest["unanswerable"]))
    chunks = ingest_dir(Path(manifest["data"]))
    index = NumpyVectorIndex.load(Path(manifest["index"]))
    if chunks != index.chunks:
        raise ValueError("Dense index differs from corpus")
    import numpy as np

    queries = [q["question"] for q in labels + negatives]
    metadata = json.loads(Path(manifest["cache"]).with_suffix(".json").read_text(encoding="utf-8"))
    if metadata != {"model": manifest["embedding_model"], "queries": queries}:
        raise ValueError("Query cache metadata mismatch")
    vectors = np.load(manifest["cache"], allow_pickle=False)
    if vectors.shape != (len(queries), index.vectors.shape[1]) or not np.isfinite(vectors).all():
        raise ValueError("Invalid query cache")
    retriever = HybridRetriever(chunks, index, CachedEmbedder(queries, vectors))
    runs = []
    for _ in range(args.repeats):
        ranked = RankedResults(queries, retriever.search_many(queries, top_k=8))
        runs.append({"retrieval": evaluate_retriever(ranked, labels, 8),
                     "answerable": answer_metrics(ranked, labels, 8),
                     "unanswerable": answer_metrics(ranked, negatives, 8)})
    first = runs[0]
    integrity = evaluate_integrity(manifest["integrity_cases"])
    stable = all(all(stable_predictions(run[split]) == stable_predictions(first[split])
                     for split in ("answerable", "unanswerable")) and run["retrieval"] == first["retrieval"]
                 for run in runs[1:])
    floors = manifest["regression_floors"]
    checks = {"stable_predictions": stable,
              "evidence_integrity": integrity["cases"] > 0 and integrity["passed"] == integrity["cases"],
              "retrieval": first["retrieval"]["recall_at_k"] >= floors["recall_at_k"],
              "coverage": first["answerable"]["literal_answer_point_coverage"] >= floors["coverage"],
              "reference_pages": first["answerable"]["gold_page_citation_precision"] >= floors["reference_pages"],
              "no_positive_refusals": first["answerable"]["abstention_rate"] == 0,
              "negative_refusals": first["unanswerable"]["abstention_rate"] == 1,
              "exact_support": first["answerable"]["exact_source_support_rate"] == 1,
              "no_duplicates": first["answerable"]["duplicate_claims"] == 0}
    report = {"purpose": "Frozen development regression baseline, not independent semantic acceptance",
              "manifest_sha256": fingerprint(args.manifest), "repeats": args.repeats,
              "checks": checks, "passed": all(checks.values()), "runs": runs, "integrity": integrity,
              "code_sha256": {str(p): fingerprint(p) for p in
                              [Path(__file__), Path("scripts/evaluate_quality.py"),
                               *sorted(Path("src/course_rag_assistant").rglob("*.py"))]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": checks}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
