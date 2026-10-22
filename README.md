# Course RAG Assistant

A course-focused RAG system that answers from *your own course materials* — with
page-exact citations, prerequisite-aware explanations, and a quiz loop that maps
wrong answers back to knowledge gaps.

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

## Tech stack

Python, FastAPI · PyMuPDF ingestion · BGE-M3 embeddings · BM25 + dense hybrid
retrieval · BGE reranker · local numpy indexes · OpenAI-compatible LLM API
(pluggable, local option supported)

## License

MIT
