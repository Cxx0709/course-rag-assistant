import json

import numpy as np
import pytest

from course_rag_assistant.answer import answer
from course_rag_assistant.content import navigation_reason
from course_rag_assistant.dense import NumpyVectorIndex
from course_rag_assistant.ingest import Chunk
from course_rag_assistant.retrieval import Hit, HybridRetriever


@pytest.mark.parametrize("text", [
    "3.2 支持向量机 . . . . . . 41",
    "Contents\n1. Introduction .......... 3\n2. Methods .......... iv",
    "目录\n5.1 聚类分析…………89",
    "第 9 页 / 共 20 页", "Page 9 of 20", "目录",
    "388 10. 注意力机制", "6.1. 从全连接层到卷积 219",
])
def test_explicit_navigation_is_excluded(text):
    assert navigation_reason(text) is not None


@pytest.mark.parametrize("text", [
    "3.2 支持向量机", "模型的核心由输入层、隐藏层和输出层构成。",
    "x = 1.2345", "1\n2\n3", "x = [..., 3]", "第9页给出了定义。",
    "Table of contents can help readers navigate.",
    "3.2 支持向量机 . . . . . . 41\n支持向量机寻找最大间隔的分离超平面。",
    "本次实验精度为95%。", "图3.1: 支持向量机的间隔示意图",
    "10. 注意力机制", "6.1. 从全连接层到卷积", "1 2.5 + 3 = 6.5",
    "1. 记录温度为 300", "388 10. 注意力机制可以聚合信息。",
])
def test_body_code_formulas_and_mixed_chunks_are_retained(text):
    assert navigation_reason(text) is None


def _corpus():
    # More than top-k navigation entries must not prevent replenishing results.
    toc = [Chunk(f"3.{i} 支持向量机 . . . . . . {i + 10}", "book.pdf", 2) for i in range(12)]
    body = Chunk("支持向量机寻找最大间隔。", "book.pdf", 40)
    return toc + [body]


class Embedder:
    def embed(self, texts):
        return np.asarray([[1.0, 0.0]] * len(texts))


def test_dense_filters_before_top_k_for_single_and_batch_search():
    chunks = _corpus()
    # Navigation has a higher dense score than the real body.
    vectors = np.asarray([[1.0, 0.0]] * 12 + [[0.8, 0.6]])
    index = NumpyVectorIndex(chunks, vectors)
    single = index.search("支持向量机", Embedder(), 1)
    batch = index.search_many(["支持向量机"], Embedder(), 1)[0]
    assert single == batch == [(chunks[-1], 0.8)]
    assert index.chunks == chunks  # Filtering does not rewrite saved provenance.


@pytest.mark.parametrize("dense", [False, True])
def test_retrieval_replenishes_body_results_and_batch_matches_single(dense):
    chunks = _corpus()
    index = NumpyVectorIndex(chunks, np.asarray([[1.0, 0.0]] * len(chunks))) if dense else None
    retriever = HybridRetriever(chunks, index, Embedder() if dense else None)
    single = retriever.search("支持向量机", 1)
    assert len(single) == 1 and single[0].chunk is chunks[-1]
    assert retriever.search_many(["支持向量机"], 1)[0] == single


def test_only_navigation_refuses_before_generation():
    class UnexpectedGenerator:
        def generate(self, prompt):
            pytest.fail("Navigation-only evidence must not reach the model")

    result = answer("支持向量机", [Hit(_corpus()[0], 1, "dense")], UnexpectedGenerator())
    assert result.abstention_reason == "non_body_evidence"


def test_running_titles_cannot_crowd_out_attention_body():
    chunks = [Chunk(f"{page} 10. 注意力机制", "book.pdf", page) for page in (388, 426, 420)]
    body = Chunk("注意力机制有选择地聚合值。", "book.pdf", 404)
    hits = HybridRetriever(chunks + [body]).search("注意力机制的作用是什么", 3)
    assert [h.chunk for h in hits] == [body]
    assert "聚合值" in answer("注意力机制的作用是什么", hits).text


def test_mixed_chunk_retains_body_and_excludes_navigation_from_both_modes():
    chunk = Chunk("3.2 支持向量机 . . . . . . 41\n支持向量机寻找最大间隔。\nPage 41 of 90\n41 3. 支持向量机",
                  "book.pdf", 41)
    hit = Hit(chunk, 1, "bm25")

    class Generator:
        def generate(self, prompt):
            payload = json.loads(prompt.split("\n")[-1])
            assert payload["candidates"] == [{"id": 1, "text": "支持向量机寻找最大间隔。",
                                               "source": "[book.pdf, p.41]"}]
            return '{"sentence_ids":[1]}'

    for generator in (None, Generator()):
        result = answer("支持向量机", [hit], generator)
        assert result.text == "支持向量机寻找最大间隔。 [book.pdf, p.41]"
        assert result.unsupported_claims == []


def test_reranker_receives_candidate_pool_not_pretruncated_top_k():
    chunks = [Chunk("支持向量机" * 5, "book.pdf", 2),
              Chunk("支持向量机寻找最大间隔。", "book.pdf", 40)]

    class Reranker:
        def rerank(self, query, hits, top_k):
            assert len(hits) == 2
            return sorted(hits, key=lambda hit: hit.chunk.page, reverse=True)[:top_k]

    assert HybridRetriever(chunks, reranker=Reranker()).search("支持向量机", 1)[0].chunk.page == 40
