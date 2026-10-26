import json

import pytest

from course_rag_assistant.answer import answer
from course_rag_assistant.evidence import select_evidence
from course_rag_assistant.ingest import Chunk
from course_rag_assistant.rerank import ChatReranker
from course_rag_assistant.retrieval import Hit


class Generator:
    def __init__(self, output):
        self.output = output

    def generate(self, prompt):
        self.prompt = json.loads(prompt.split("\n")[-1])
        return self.output


def hits():
    return [Hit(Chunk("随机森林的实现示例。", "a.pdf", 1), .1, "fused"),
            Hit(Chunk("随机森林由多个决策树组成。", "b.pdf", 2), .09, "fused")]


def test_chat_rerank_selects_original_sources_and_keeps_scores():
    candidates = hits()
    ranker = ChatReranker(Generator('{"passage_ids":[2,1]}'))
    result = ranker.rerank("随机森林的组成是什么？", candidates, 2)
    assert result[0].chunk is candidates[1].chunk
    assert result[0].score == .09 and result[0].stage == "chat_reranked"
    assert ranker.diagnostics[0]["status"] == "selected"


@pytest.mark.parametrize("raw", [
    '{"passage_ids":[0]}', '{"passage_ids":[3]}', '{"passage_ids":[true]}',
    '{"passage_ids":[1,1]}', '{"passage_ids":[1.0]}', '{"passage_ids":["1"]}',
    '{"passage_ids":[1],"text":"伪造内容"}', '{"passage_ids":[1],"passage_ids":[2]}',
    '[]', '{"passage_ids":[', '{"passage_ids":[1]}解释',
])
def test_invalid_rerank_falls_back_and_records_failure(raw):
    ranker = ChatReranker(Generator(raw))
    candidates = hits()
    assert ranker.rerank("随机森林", candidates, 2) == candidates
    assert ranker.diagnostics[0]["status"] == "fallback_original_ranking"


def test_empty_selection_can_refuse_without_fabricating_evidence():
    ranker = ChatReranker(Generator('{"passage_ids":[]}'))
    assert ranker.rerank("不相关的问题", hits(), 2) == []
    assert ranker.diagnostics[0]["status"] == "selected"


def test_service_failure_is_recorded_without_leaking_error_text():
    class BrokenGenerator:
        def generate(self, prompt):
            raise RuntimeError("secret token")

    ranker = ChatReranker(BrokenGenerator())
    assert ranker.rerank("随机森林", hits(), 1) == hits()[:1]
    assert ranker.diagnostics[0]["error_type"] == "RuntimeError"
    assert "secret token" not in str(ranker.diagnostics)


def test_excerpt_budget_and_candidate_limit_do_not_rewrite_sources():
    candidates = hits()
    candidates[0].chunk.text = "开头" + "文" * 1000 + "结尾"
    generator = Generator('{"passage_ids":[1]}')
    ranker = ChatReranker(generator, max_candidates=1, excerpt_chars=100)
    result = ranker.rerank("问题", candidates, 1)
    preview = generator.prompt["passages"][0]["text"]
    assert preview.startswith("开头") and preview.endswith("结尾") and "省略" in preview
    assert len(generator.prompt["passages"]) == 1
    assert result[0].chunk is candidates[0].chunk and len(result[0].chunk.text) > 1000


def test_structure_question_does_not_require_literal_structure_word():
    candidates = [hits()[1]]
    candidates[0].stage = "chat_reranked"
    assert select_evidence("随机森林的组成是什么？", candidates).hits == candidates
    assert not select_evidence("支持向量机的组成是什么？", candidates).hits


def test_valid_selection_keeps_other_candidates_available_to_evidence_guard():
    candidates = hits()
    result = ChatReranker(Generator('{"passage_ids":[2]}')).rerank("随机森林", candidates, 2)
    assert [h.chunk for h in result] == [candidates[1].chunk, candidates[0].chunk]


def test_ranked_answer_is_bounded_and_preserves_answer_order():
    text = "".join(f"随机森林的第{i}个性质有说明。" for i in range(1, 9))
    candidate = Hit(Chunk(text, "book.pdf", 1), 1, "chat_reranked")
    result = answer("随机森林", [candidate], Generator('{"sentence_ids":[8,7,6,5,4,3,2,1]}'))
    assert result.selected_sentence_ids == [8, 7, 6, 5, 4, 3]
    assert "第1个" not in result.text


def test_conclusion_late_in_source_can_lead_the_answer():
    text = "训练集由训练样本组成。测试集包含未参与训练的数据。训练集用于拟合参数，测试集用于评估模型。"
    candidate = Hit(Chunk(text, "book.pdf", 7), 1, "chat_reranked")
    result = answer("训练集和测试集的用途有什么不同？", [candidate],
                    Generator('{"sentence_ids":[3,2]}'))
    assert result.text.splitlines()[0] == "训练集用于拟合参数，测试集用于评估模型。 [book.pdf, p.7]"
    assert result.selected_sentence_ids == [3, 2]
    assert not result.unsupported_claims


def test_identical_sentences_across_sources_are_rendered_once():
    candidates = [Hit(Chunk("训练集用于拟合参数。", "a.pdf", 1), 1, "chat_reranked"),
                  Hit(Chunk("训练集用于拟合参数。测试集不用于拟合参数。", "b.pdf", 2), 1, "chat_reranked")]
    result = answer("训练集", candidates, Generator('{"sentence_ids":[2,1,3]}'))
    assert result.selected_sentence_ids == [2, 3]
    assert result.text.count("训练集用于拟合参数。") == 1
    assert "[b.pdf, p.2]" in result.text
    assert "测试集不用于拟合参数。" in result.text
    assert not result.unsupported_claims


def test_english_parenthetical_soft_wrap_is_not_rendered_as_two_fragments():
    text = "这种模型称为随机森林（random\nforest）。"
    result = answer("随机森林", [Hit(Chunk(text, "book.pdf", 1), 1, "bm25")])
    assert result.text == "这种模型称为随机森林（random forest）。 [book.pdf, p.1]"
    assert not result.unsupported_claims


def test_personal_record_guards_still_apply_after_chat_reranking():
    candidates = [Hit(Chunk("模型测试准确率为95%。", "textbook.pdf", 1), 1, "fused")]
    selected = ChatReranker(Generator('{"passage_ids":[1]}')).rerank("我的模型测试准确率是多少", candidates, 1)
    assert answer("我的模型测试准确率是多少", selected).abstention_reason == "missing_context_record"


def test_parenthetical_pdf_wraps_keep_definition_and_qualifier_together():
    text = "样本也称为数据点\n（data point），由一组称为特征（features）\n的属性组成。"
    candidate = Hit(Chunk(text, "book.pdf", 38), 1, "chat_reranked")
    result = answer("什么是样本", [candidate], Generator('{"sentence_ids":[1]}'))
    assert result.text == "样本也称为数据点（data point），由一组称为特征（features）的属性组成。 [book.pdf, p.38]"
    assert not result.unsupported_claims
