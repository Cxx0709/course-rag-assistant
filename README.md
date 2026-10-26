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
2. **Prerequisite linking.** Concepts are linked in a dependency graph
   (e.g. "Naive Bayes → conditional independence → Bayes' theorem"), so
   answers point back to what you should review first.
3. **Cross-source comparison.** The same concept from different textbooks or
   lecture decks is aligned side-by-side, with definitions and notation
   differences made explicit.
4. **Error-gap analysis.** Quiz mode generates questions from the corpus,
   grades answers, and maps mistakes to missing concepts — producing a review
   list instead of just a score.
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
