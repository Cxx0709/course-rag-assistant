from dataclasses import replace

import pytest

from course_rag_assistant.ingest import Chunk
from course_rag_assistant.quiz import make_questions
from course_rag_assistant.quiz_candidates import (
    candidate_prompt,
    extract_candidates,
    prompt_rejection,
)
from course_rag_assistant.quiz_terms import name_list, synonym_definition

ENUMERATION = ("通常，涉及一个、两个和三个变量的概率公式分别被称为一元语法（unigram）、"
               "二元语法（bigram）和三元语法（trigram）模型。")
SYNONYM = "验证误差（validation error），又称开发误差（development error），是指模型在留出验证数据上的误差。"
PEER = Chunk("训练误差是指模型在训练数据上计算得到的误差。", "notes.pdf", 8, "4.4 误差")


def test_counted_enumeration_preserves_order_category_scope_and_full_evidence():
    rows = make_questions([Chunk(ENUMERATION, "book.pdf", 323, "8.3 语言模型")])
    assert [q["concept"] for q in rows] == ["一元语法", "二元语法", "三元语法"]
    for q, count in zip(rows, ("一个", "两个", "三个")):
        assert q["sentence"] == ENUMERATION and q["page"] == 323
        assert set(q["options"]) == {"一元语法", "二元语法", "三元语法"}
        assert q["prompt"] == f"按资料原文补全空缺：通常，涉及{count}变量的概率公式被称为____模型。"
        assert q["options"][q["correct_index"]] == q["concept"]
    assert not make_questions([Chunk(ENUMERATION, "book.pdf", 323)], target_concept="语法")


@pytest.mark.parametrize("source", [
    ENUMERATION.replace("一个、两个和三个", "一个和两个"),
    ENUMERATION.replace("一个、两个和三个", "一个、1个和三个"),
    ENUMERATION.replace("一元语法", "二元语法"),
    ENUMERATION.replace("bigram", "unigram"),
    ENUMERATION.replace("bigram", "when data matches"),
    ENUMERATION.replace("模型。", "模型，但仅在特定条件下成立。"),
    ENUMERATION.replace("变量的概率公式", "上述变量的概率公式"),
    ENUMERATION.replace("分别被称为", "有时也许被称为"),
    ENUMERATION.replace("一元语法（unigram）", "一元语法模型（unigram）"),
])
def test_enumeration_does_not_drop_conditions_guess_mapping_or_duplicate_members(source):
    assert make_questions([Chunk(source, "book.pdf", 323)]) == []


def test_enumeration_keeps_conditions_in_description_and_supports_other_topics():
    source = "在缓存系统中，使用一个和两个备用副本的恢复过程分别被称为单副本恢复（single recovery）和双副本恢复（double recovery）方法。"
    rows = make_questions([Chunk(source, "notes.md", 0)])
    assert len(rows) == 2
    assert rows[0]["prompt"] == "按资料原文补全空缺：在缓存系统中，使用一个备用副本的恢复过程被称为____方法。"
    assert all(q["sentence"] == source for q in rows)


def test_explicit_synonyms_share_identity_but_preserve_original_evidence():
    chunks = [Chunk(SYNONYM, "notes.pdf", 8, "4.4 误差"),
              Chunk("开发误差是指在开发数据上的模型误差。", "notes.pdf", 8, "4.4 误差"), PEER]
    candidates = extract_candidates(chunks)
    first = candidates[0]
    assert first.concept == "验证误差" and first.synonyms == ("开发误差",)
    assert first.aliases == ("validation error", "development error")
    assert candidates[1].synonyms == ("验证误差",)
    assert first.sentence == SYNONYM
    assert candidate_prompt(first) == "按资料原文补全空缺：____是指模型在留出验证数据上的误差。"
    assert prompt_rejection(first, candidate_prompt(first)) is None
    rows = make_questions(chunks)
    assert {q["concept"] for q in rows} == {"验证误差", "训练误差"}
    assert all(not {"验证误差", "开发误差"} <= set(q["options"]) for q in rows)
    assert make_questions(chunks[:2]) == []


def test_synonyms_propagate_transitively_only_inside_their_source_scope():
    a = Chunk("验证误差，又称开发误差，是指模型在验证数据上的误差。", "notes.pdf", 8, "4.4 误差")
    b = Chunk("开发误差，也称为留出误差，是指模型在留出数据上的误差。", "notes.pdf", 8, "4.4 误差")
    c = Chunk("留出误差是指模型在留出样本上的误差。", "notes.pdf", 8, "4.4 误差")
    candidates = extract_candidates([a, b, c])
    assert all({x.concept, *x.synonyms} == {"验证误差", "开发误差", "留出误差"} for x in candidates)
    assert make_questions([a, b, c]) == []
    assert extract_candidates([a, replace(c, doc="other.pdf")])[-1].synonyms == ()
    assert extract_candidates([a, replace(c, section="5.1 误差")])[-1].synonyms == ()
    assert extract_candidates([replace(a, section="第一篇"), replace(c, section="第二篇")])[-1].synonyms == ()


@pytest.mark.parametrize("link", ["或", "和", "不等于", "不是", "有时称为", "在特定条件下称为"])
def test_equivalence_is_never_inferred_from_disjunction_or_conditional_names(link):
    assert synonym_definition(f"验证误差，{link}开发误差，是指模型在开发数据上的误差。") is None


def test_synonym_conditions_and_negation_remain_in_stem_and_feedback():
    source = SYNONYM.replace("是指", "是指在样本同分布时，").replace("误差。", "误差，不能直接用于拟合参数。")
    q, = make_questions([Chunk(source, "notes.pdf", 8, "4.4 误差"), PEER], target_concept="验证误差")
    assert "在样本同分布时" in q["prompt"] and "不能直接用于拟合参数" in q["prompt"]
    assert q["sentence"] == source
    assert not synonym_definition(SYNONYM.replace("development error", "when samples match"))


def test_same_identity_is_not_listed_twice_as_distractors_for_a_third_concept():
    rows = make_questions([Chunk(SYNONYM, "notes.pdf", 8, "4.4 误差"),
                           Chunk("开发误差是指模型在开发数据上的误差。", "notes.pdf", 8, "4.4 误差"), PEER])
    training = next(q for q in rows if q["concept"] == "训练误差")
    assert len(training["options"]) == 2


def test_synonym_mentioned_elsewhere_in_stem_cannot_hint_at_the_answer():
    source = SYNONYM.replace("上的误差。", "上的误差，也就是开发误差。")
    candidate = extract_candidates([Chunk(source, "notes.pdf", 8, "4.4 误差")])[0]
    assert prompt_rejection(candidate, candidate_prompt(candidate)) == "answer_hint"


def test_list_parser_consumes_all_text_including_qualifications():
    assert [n.name for n in name_list("甲模型（alpha）和乙模型（beta）")] == ["甲模型", "乙模型"]
    assert not name_list("甲模型和乙模型（when enabled）")
    assert not name_list("甲模型和乙模型，但仅限特殊情况")
