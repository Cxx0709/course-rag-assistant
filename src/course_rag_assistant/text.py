"""Small lexical normalizations; original passages remain untouched for citations."""
import re


def normalize_matching(text: str) -> str:
    # PDF extraction often inserts line breaks inside Chinese words.
    # Keep intentional spaces, including separators inserted by question cleaning.
    text = re.sub(r"(?<=[\u4e00-\u9fff])[ \t]*\r?\n[ \t]*(?=[\u4e00-\u9fff])", "", text)
    # Canonicalize only this constrained adjective/predicate alternation. Keep
    # positive and negative separate; do not remove negation or numeric values.
    nouns = "输入|输出|权重|梯度|数值"
    text = re.sub(rf"({nouns})(?:为|是)(正|负)", r" \2 \1 ", text)
    return re.sub(rf"(正|负)({nouns})", r" \1 \2 ", text)


def code_identifiers(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+", text, flags=re.ASCII))
