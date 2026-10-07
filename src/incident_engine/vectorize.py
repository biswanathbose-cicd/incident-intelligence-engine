"""TF-IDF vectorizers: a readable reference implementation and a sparse, vectorized one.

Both use scikit-learn's conventions (smooth idf ``ln((1+n)/(1+df)) + 1``, raw term counts, L2 row
normalization, alphabetically sorted vocabulary) so they can be tested against each other *and*
against ``sklearn.feature_extraction.text.TfidfVectorizer`` for exact numeric parity."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse

from .text import analyze


class ReferenceTfidf:
    """Obviously-correct pure-Python TF-IDF: the test oracle and the benchmark baseline.

    ``transform`` returns one ``{feature_index: weight}`` dict per document."""

    def __init__(self, ngram_range: tuple[int, int] = (1, 2), min_df: int = 1):
        self.ngram_range, self.min_df = ngram_range, min_df

    def fit(self, docs: Iterable[str]) -> "ReferenceTfidf":
        n, df = 0, Counter()
        for d in docs:
            n += 1
            df.update(set(analyze(d, self.ngram_range)))
        terms = sorted(t for t, c in df.items() if c >= self.min_df)
        self.terms_ = terms
        self.vocab_ = {t: i for i, t in enumerate(terms)}
        self.idf_ = [math.log((1 + n) / (1 + df[t])) + 1.0 for t in terms]
        return self

    def transform(self, docs: Iterable[str]) -> list[dict[int, float]]:
        out = []
        for d in docs:
            counts = Counter(t for t in analyze(d, self.ngram_range) if t in self.vocab_)
            vec = {self.vocab_[t]: c * self.idf_[self.vocab_[t]] for t, c in counts.items()}
            norm = math.sqrt(sum(v * v for v in vec.values()))
            out.append({i: v / norm for i, v in vec.items()} if norm else vec)
        return out

    def fit_transform(self, docs: Sequence[str]) -> list[dict[int, float]]:
        return self.fit(docs).transform(docs)


class SparseTfidf:
    """CSR-based TF-IDF. Documents are tokenized once; idf scaling and L2 normalization are single
    vectorized passes over the ``nnz`` entries (O(nnz) time, O(nnz) memory)."""

    def __init__(self, ngram_range: tuple[int, int] = (1, 2), min_df: int = 1):
        self.ngram_range, self.min_df = ngram_range, min_df

    # -- fitting -----------------------------------------------------------------------------
    def fit(self, docs: Sequence[str]) -> "SparseTfidf":
        self._fit_from_counters([Counter(analyze(d, self.ngram_range)) for d in docs])
        return self

    def fit_transform(self, docs: Sequence[str]) -> sparse.csr_matrix:
        counters = [Counter(analyze(d, self.ngram_range)) for d in docs]  # tokenize once, reuse for transform
        self._fit_from_counters(counters)
        return self._to_tfidf(self._counters_to_csr(counters))

    def _fit_from_counters(self, counters: list[Counter]) -> None:
        df: Counter = Counter()
        for c in counters:
            df.update(c.keys())
        self.terms_ = np.array(sorted(t for t, k in df.items() if k >= self.min_df), dtype=object)
        self.vocab_ = {t: i for i, t in enumerate(self.terms_)}
        n = len(counters)
        dfs = np.array([df[t] for t in self.terms_], dtype=np.float64)
        self.idf_ = np.log((1.0 + n) / (1.0 + dfs)) + 1.0

    # -- transforming ------------------------------------------------------------------------
    def transform(self, docs: Sequence[str]) -> sparse.csr_matrix:
        self._check_fitted()
        return self._to_tfidf(self._counters_to_csr([Counter(analyze(d, self.ngram_range)) for d in docs]))

    def _counters_to_csr(self, counters: list[Counter]) -> sparse.csr_matrix:
        vocab = self.vocab_
        indptr, indices, data = [0], [], []
        for c in counters:
            for t, k in c.items():
                j = vocab.get(t)
                if j is not None:
                    indices.append(j)
                    data.append(k)
            indptr.append(len(indices))
        X = sparse.csr_matrix((np.asarray(data, dtype=np.float64), np.asarray(indices, dtype=np.int32),
                               np.asarray(indptr, dtype=np.int64)), shape=(len(counters), len(vocab)))
        X.sort_indices()
        return X

    def _to_tfidf(self, X: sparse.csr_matrix) -> sparse.csr_matrix:
        X.data *= self.idf_[X.indices]
        row_ids = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
        norms = np.sqrt(np.bincount(row_ids, weights=X.data * X.data, minlength=X.shape[0]))
        norms[norms == 0.0] = 1.0
        X.data /= norms[row_ids]
        return X

    def _check_fitted(self) -> None:
        if not hasattr(self, "idf_"):
            raise RuntimeError("SparseTfidf is not fitted")
