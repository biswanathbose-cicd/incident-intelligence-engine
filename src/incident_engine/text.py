"""Tokenization and n-gram analysis shared by every vectorizer."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def analyze(text: str, ngram_range: tuple[int, int] = (1, 2)) -> list[str]:
    """Tokens followed by space-joined n-grams for each n in ``ngram_range`` (n=1 are the tokens)."""
    lo, hi = ngram_range
    if lo < 1 or hi < lo:
        raise ValueError(f"invalid ngram_range {ngram_range}")
    toks = tokenize(text)
    terms = list(toks) if lo == 1 else []
    for n in range(max(lo, 2), hi + 1):
        terms.extend(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return terms
