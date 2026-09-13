from __future__ import annotations

import re


_ASCII_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*%?")
_CJK_RUN = re.compile(r"[\u3400-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """Return deterministic tokens for a dependency-free Chinese BM25 baseline.

    ASCII words are lower-cased. Chinese runs contribute both characters and
    adjacent character bigrams. This is intentionally simple and will later be
    compared with a proper tokenizer rather than presented as a final method.
    """

    tokens = [match.group(0).lower() for match in _ASCII_TOKEN.finditer(text)]
    for match in _CJK_RUN.finditer(text):
        run = match.group(0)
        tokens.extend(run)
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens
