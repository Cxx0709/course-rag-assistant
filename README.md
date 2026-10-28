# Course RAG Assistant

A course-focused RAG system that answers from *your own course materials* — with
page-exact citations, prerequisite-aware explanations, and a quiz loop that maps
wrong answers back to knowledge gaps.

## 网页版：直接使用

双击项目里的 **`start-web.cmd`**，然后打开 **http://127.0.0.1:8000**。
点击“用一份机器学习示例试试”即可体验；也可以上传自己的 PDF、PPTX 或 Markdown，输入问题，点击引用查看原文和页码。资料保存在本机，重启后仍可使用。

默认无需模型；已有 Ollama 时可运行 `.\start-web.ps1 -Model qwen3.5:9b`，再在页面勾选“模型辅助”。首次安装运行 `uv sync --extra dev`。
详细说明和实际截图见 [网页版使用说明](docs/local-web.md)。

## Why this is not "yet another PDF chatbot"

1. **Verifiable answers.** Every factual claim is anchored to a retrieved
   passage; answers cite the exact source file, page number, and paragraph.
   Claims without a supporting passage are flagged, not hallucinated.
2. **Prerequisite linking.** Explicit prerequisite declarations in selected notes
   form a dependency graph. Web answers and wrong-answer review show a foundation-first
   study order, source excerpts for each relation, and links to prerequisite lessons.
   Missing lessons and cyclic dependencies are reported; relationships are not inferred
   from co-occurrence. See [the supported note format](docs/local-web.md#先修概念与复习顺序).
3. **Cross-source comparison.** Select two documents and enter a concept to compare
   literal excerpts by definition, notation, assumptions and applicability. Each
   excerpt links to its source; literal differences are highlighted without claiming
   semantic conflict. Missing evidence is shown explicitly. Try the two Markdown
   files in [`data/comparison-samples/`](data/comparison-samples/).
4. **Source-backed practice.** Quiz mode creates concept cloze multiple-choice
   questions from complete source statements, grades the selected options, and
   saves mistakes as a review list with source links. This measures literal
   recognition; general concept mastery and semantic free-answer grading remain planned.
5. **Evaluated, not demoed.** A labeled question set measures retrieval
   accuracy, citation correctness, and abstention quality, with an ablation
   study (dense-only vs hybrid vs +rerank).

## Architecture

```
course materials (PDF / PPTX / MD, page numbers preserved)
        │ ingest.py: extract → clean → page-preserving chunk
        ▼
  local index (BM25 + optional BGE-M3 vectors, stored under data/index/)
        │ retrieval.py: BM25 → dense (BGE-M3) → reranker
        ▼
  answer.py: constrained generation — every claim cites [doc p.N]
        │ evaluate.py: recall@k / MRR / citation precision / abstention
        ▼
  quiz mode → error-gap analysis → review list
```

## Status

The [assistant learning-session review](eval/reviews/learning-session-v1.md)
found and fixed indirect offline definition answers, PDF fraction fragments and
English aliases revealing cloze answers. All 347 tests, 24 scoped-practice checks,
the learning flow, frozen baseline and Chrome walkthrough pass. Two question
quality limitations remain documented; this is not human learning-effectiveness validation.

The [quiz quality update](docs/quiz-quality-v1.md) replaces observed topic previews
with definition questions, recognizes definitions with parenthesized English
terms, and favors concepts from the same numbered section over procedural labels.
All 334 tests, 22 real-PDF checks, the existing learning flow and browser checks pass.
These are development regressions; teaching quality still awaits human review.

The [targeted practice update](docs/targeted-practice-v1.md) adds concept and chapter
scopes, scoped wrong-answer review, and explicit insufficient-content feedback.
The UI defaults to a named concept; it never pads a scoped quiz with other topics.
Real-PDF checks cover the overfitting definition and D2L section 4.4 (PDF pages
157–167), plus scope persistence and browser/mobile behavior. Run
`python -m scripts.evaluate_targeted_practice` to reproduce the isolated API check.
Chapter extraction and the teaching value of source clozes still need human review.

The [real learning-flow acceptance](docs/learning-flow-v1.md) exercises questions,
citations, practice, wrong-answer review, restart recovery and a correct retry
using the real textbooks plus a separately selected, unreviewed course note.
It fixes prerequisite declarations being used as definition answers or quiz
content. At that stage, all 310 tests and the frozen answer baseline passed.
Targeted practice is now available; human semantic and teaching-order review remain
pending in the [review checklist](eval/reviews/learning-acceptance-v1.md).
Run `python -m scripts.evaluate_learning_flow` to reproduce the isolated API flow.

The [terminology alignment record](docs/course-alignment-v1.md) adds scoped
development/validation-set correspondence backed by model-selection evidence
from each selected source. Variance requires a choice of meaning before comparison;
unsupported senses and uncertain correspondences are not highlighted as aligned.
All 302 tests and 11 repeated real-textbook development checks pass. These are
limited reading rules; independent semantic review and broader coverage remain pending.
Run `python -m scripts.evaluate_course_pairs --manifest eval/course-alignment-v1.json --output eval/reports/course-alignment-v1.json`.
A [separate reading-plan draft](docs/course-pairs-v2.md) has checked graph orders
and PDF page links, with human review pending. Run
`python -m scripts.evaluate_course_paths` for its structure check.

Configuration now lives in `config.py`; the local Web app is organized under
`web/` with separate storage, routes, answering, security and static resources.
Existing startup commands and API URLs remain supported. See
[module layout and configuration](docs/project-structure.md) for settings and
the 233-test / installed-wheel validation record.

The [v12 evidence and baseline report](docs/evidence-baseline-v12.md) describes
per-citation source verification, deduplication in all answer modes, quotation
and English sentence integrity, and a frozen, repeated offline regression gate.
The web app now includes source-backed quizzes and persistent wrong-answer review.
Answers, quiz results and review lists also include source-declared prerequisite paths.
Run `python -m scripts.evaluate_baseline` from the repository root to verify the
frozen local corpus, labels, index and cached query vectors before repeated evaluation.
This is a development regression baseline, not independent semantic acceptance.

Ingestion, local BM25/dense retrieval, deterministic citation checking, and
retrieval evaluation are implemented. See
[`docs/implementation-plan.md`](docs/implementation-plan.md) for the full
6–8 week build plan, data design, and acceptance criteria.

The [v5 record-facts report](docs/record-facts-v5.md) evaluates the same 30 real-textbook
questions and 10 unanswerable questions as the [v4 report](docs/matching-v4.md).
Hybrid retrieval reaches Recall@8 0.80, rejects all 10 unanswerable development
questions, and no longer rejects the 30 answerable questions. Answer-point
coverage is 57.78%; not refusing is not proof of correctness. An additional
20-case frozen synthetic validation set improves from 9/20 to 20/20 with explicit
field/value and conflict checks; the previous v4 cases now serve as development
regressions. These results do not establish general answer-quality acceptance.
Personal results require an explicitly titled record source; this is a heuristic,
not verification of ownership or recency. Supported record fields must contain
an explicit value, and conflicting candidate records cause abstention.
The offline citation verifier accepts
complete source sentences only (ignoring whitespace). Free paraphrases require
a future semantic verifier. Citation validity does not establish answer relevance.

The [v6 evaluation workflow](docs/quality-evaluation-v6.md) adds explicit `--chat`
evaluation, per-question model output and fallback traces, and pending human-review
exports. The default remains offline; missing chat configuration or an explicitly
requested unavailable reranker fails instead of substituting another mode.
Small `--limit` runs are marked as smoke tests, not acceptance results.
The [v7 sentence-selection report](docs/sentence-selection-v7.md) fixes the v6
chat citation-format failure: the model selects source sentence IDs, and the
application renders their original text and citations before verification.
On the same 30+10 development questions, local `qwen3.5:9b` with hybrid retrieval
now rejects 0/30 answerable questions and 10/10 unanswerable questions, with no
generation failures. Literal answer-point coverage is 73.33%, and reference-page
citation precision is 60.49%. Some selections still use table-of-contents or
incomplete evidence; general answer-quality acceptance is not established.
The offline baseline is unchanged. Custom generators must now return
`{"sentence_ids":[1,3]}` (or an empty array to abstain), rather than answer prose.

The [v8 content-filter report](docs/content-filter-v8.md) excludes explicit
TOC and running-title fragments before retrieval truncation and sentence selection.
On the same development set, hybrid Recall@8 is 83.33% and chat answer-point
coverage is 78.33%, with 0/30 answerable refusals and 10/10 unanswerable refusals.
Reference-page citation precision decreases from 60.49% to 57.65%; human review
is still pending. Real BGE reranking remains untested without a configured service.

The [v9 answer-completeness report](docs/answer-completeness-v9.md) adds
optional chat-model passage selection with `answer --chat-rerank` and
`search --chat-rerank`. In a same-configuration development comparison,
answer-point coverage increases from 75.00% to 85.28%; all 30 answerable
questions receive answers and all 10 unanswerable questions are rejected.
Reference-page citation precision decreases from 58.54% to 50.41%, and the
40-question run takes 164 seconds instead of 45 seconds. This mode remains
opt-in; these results do not establish general answer correctness.
See [actual answer examples](docs/answer-examples-v9.md) for the improved
definitions and remaining repetition.

The [v10 concise-answer report](docs/concise-answers-v10.md) preserves
the model's answer order, removes identical selected sentences, and repairs PDF
line breaks around parenthetical terms. With the same development questions,
model configuration, and passage selections, average answer text is 17.4%
shorter; literal answer-point coverage changes from 85.28% to 81.39%, while
reference-page citation precision increases from 50.41% to 56.38%.
The report explains the keyword mismatches and remaining incomplete excerpts.
See [before-and-after answers](docs/answer-examples-v10.md).

The latest [v11 PDF sentence-integrity report](docs/pdf-sentence-integrity-v11.md)
filters obvious PDF fragments before selection, fallback answering, and citation
verification. Complete list items and code remain eligible; PDF passages with no
usable units no longer crowd out later complete evidence. Both tested chat modes
answer all 30 positive development questions and reject all 10 negatives.
Chat-reranked answer-point coverage decreases from 81.39% to 78.61%; the report
records missing selections and model variability instead of claiming overall
quality acceptance. See [actual answers](docs/answer-examples-v11.md).

## Quick start

```bash
uv sync --extra dev
course-rag ingest data/samples/
course-rag search "朴素贝叶斯 条件独立" --data data/samples
course-rag answer "朴素贝叶斯为什么需要条件独立假设？" --data data/samples
course-rag evaluate eval/labeled-questions.jsonl --data data/samples
course-rag evaluate eval/unanswerable-questions.jsonl --data data/samples --unanswerable
python scripts/batch_answer.py eval/labeled-questions.jsonl --output eval/answers-bm25.jsonl
python scripts/score_answers.py eval/labeled-questions.jsonl eval/answers-bm25.jsonl
```

`python -m course_rag_assistant ...` provides the same unified interface. To
enable hosted generation, set `COURSE_RAG_LLM_BASE_URL`,
`COURSE_RAG_LLM_API_KEY`, and `COURSE_RAG_LLM_MODEL`. Dense retrieval and BGE
reranking are enabled automatically when their corresponding environment
variables and persisted index are present.

For a local Ollama embedding model, start Ollama and set the model name before
building the index:

```powershell
$env:COURSE_RAG_OLLAMA_MODEL="nomic-embed-text"
python scripts/build_dense_index.py --data data/raw/datamining-notes
```

To enable local answer generation, set the chat model separately. A smaller
`COURSE_RAG_OLLAMA_CHAT_NUM_PREDICT` keeps CPU-only responses responsive; failed
local generation automatically falls back to the cited extractive answer.

```powershell
$env:COURSE_RAG_OLLAMA_CHAT_MODEL="qwen3.5:9b"
$env:COURSE_RAG_OLLAMA_CHAT_THINK="false"
$env:COURSE_RAG_OLLAMA_CHAT_NUM_PREDICT="512"
```

## Tech stack

Python, FastAPI · PyMuPDF ingestion · BGE-M3 embeddings · BM25 + dense hybrid
retrieval · BGE reranker · local numpy indexes · OpenAI-compatible LLM API
(pluggable, local option supported)

## License

MIT
