"""Regressions from development cases; frozen validation cases are separate."""
import json

from course_rag_assistant.answer import answer
from course_rag_assistant.ingest import Chunk
from course_rag_assistant.retrieval import Hit, HybridRetriever, _tokenize


def test_relu_reordered_polarity_retains_real_answer():
    text = "当输入为负时，ReLU函数的导数为0，而当输入为正时，ReLU函数的导数为1。"
    result = answer("ReLU在负输入和正输入处的导数分别是多少？", [Hit(Chunk(text, "notes.pdf", 149), 1, "bm25")])
    assert result.abstention_reason is None
    assert text in result.text


def test_polarity_is_not_erased_and_words_survive_pdf_wrapping():
    assert set(_tokenize("负输入")) == set(_tokenize("输入为负"))
    assert set(_tokenize("正输入")) != set(_tokenize("负输入"))
    assert _tokenize("指定输入形状") == _tokenize("指\n定输入形状")


def test_question_separators_do_not_create_extra_terms():
    assert "用法" not in _tokenize("使用 法则")
    result = answer("对复合函数求导可以使用什么法则？", [
        Hit(Chunk("链式法则可以用来微分复合函数。", "notes.pdf", 87), 1, "bm25")])
    assert result.abstention_reason is None


def test_api_explanation_beats_repeated_incidental_usage():
    explanation = Chunk("nn.Linear的两个参数指定输入特征形状和输出特征形状。", "guide.md", 0)
    noise = Chunk("nn.Linear(2, 1)\n" * 20, "example.md", 0)
    wrong = Chunk("nn.Sequential的两个参数指定输入特征形状和输出特征形状。", "other.md", 0)
    retriever = HybridRetriever([noise, wrong, explanation])
    assert retriever.search("nn.Linear的两个参数各指定什么？", 1)[0].chunk == explanation


def test_missing_exact_symbol_does_not_use_similar_api():
    result = answer("nn.Linear的参数是什么？", [Hit(Chunk("nn.Sequential的参数是网络层。", "notes.md", 0), 1, "bm25")])
    assert result.abstention_reason == "missing_code_identifier"


def test_pdf_line_wrap_does_not_insert_citation_inside_word():
    text = "第一个指\n定输入特征形状，第二个指定输出特征形状。"
    result = answer("输入特征形状", [Hit(Chunk(text, "notes.pdf", 120), 1, "bm25")])
    assert result.text == "第一个指定输入特征形状，第二个指定输出特征形状。 [notes.pdf, p.120]"


def test_old_index_rebuilds_stale_tokenization(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"version": 1, "chunks": [
        {"text": "nn.Linear定义全连接层。", "doc": "notes.md", "page": 0}],
        "tokens": [["stale"]]}), encoding="utf-8")
    retriever = HybridRetriever.load(path)
    assert retriever.search("nn.Linear", 1)[0].score > 0
