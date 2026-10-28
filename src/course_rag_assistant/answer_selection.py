"""Bounded, literal definition selection for offline answers, not semantic QA."""
import re


def definition_sentence_ids(question: str, sentences: list[str]) -> list[int]:
    question = question.strip().rstrip("？?")
    single = re.fullmatch(r"(?:什么是|何为)(.{2,40})|(.{2,40}?)(?:是什么|是指什么)", question)
    pair = re.fullmatch(r"(.{2,40}?)(?:和|与)(.{2,40}?)(?:有什么区别|有什么不同|有何区别)", question)
    subjects = list(pair.groups()) if pair else [next(g for g in single.groups() if g)] if single else []
    if not subjects:
        return []
    ids = []
    for subject in subjects:
        term = re.escape(subject)
        direct = re.compile(r"^[•●\s]*" + term +
                            r"(?:\s*[（(][A-Za-z][A-Za-z0-9 ,_\-]{0,79}[）)])?"
                            r"\s*(?:是指|指的是|是一种|是用来|用于)")
        named = re.compile(r"(?:现象|过程|方法|模型|函数|算法|性质|关系|变量|事件|集合)"
                           r"(?:称为|叫做)" + term + r"(?=[（(，。；]|$)")
        choices = [(0 if direct.search(sentence) else 1, i)
                   for i, sentence in enumerate(sentences, 1)
                   if direct.search(sentence) or (named.search(sentence)
                       and not re.match(r"^(?:它|这|其|该|上述|因此|其中)", sentence))]
        if not choices:
            # A partial match must not masquerade as a complete comparison.
            return []
        ids.append(min(choices)[1])
    return list(dict.fromkeys(ids))
