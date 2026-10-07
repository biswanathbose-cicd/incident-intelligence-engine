"""Similar-incident search. Three interchangeable ``SimilarityIndex`` strategies:

    ReferenceIndex   pure-Python brute force over dict vectors  (oracle + baseline)
    SparseExactIndex chunked sparse matrix product + O(n) top-k (exact; the production default)
    LSHIndex         random-hyperplane LSH, candidates re-ranked exactly (approximate)

Vectors are L2-normalized TF-IDF rows, so a dot product *is* cosine similarity. Every ``query``
returns ``(indices, scores)`` of shape (n_queries, k): descending scores, strictly positive only,
padded with (-1, 0.0)."""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from scipy import sparse

from .topk import topk_partition, topk_rows

_INDEXES: dict[str, Callable[..., "SimilarityIndex"]] = {}


def register_index(name: str):
    def deco(cls):
        _INDEXES[name] = cls
        return cls
    return deco


def create_index(name: str, **params) -> "SimilarityIndex":
    if name not in _INDEXES:
        raise ValueError(f"unknown index '{name}'; available: {sorted(_INDEXES)}")
    return _INDEXES[name](**params)


class SimilarityIndex(ABC):
    @abstractmethod
    def build(self, X) -> "SimilarityIndex": ...

    @abstractmethod
    def query(self, Q, k: int) -> tuple[np.ndarray, np.ndarray]: ...

    @staticmethod
    def _check_k(k: int) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")


@register_index("reference")
class ReferenceIndex(SimilarityIndex):
    def build(self, X: list[dict[int, float]]) -> "ReferenceIndex":
        self.docs = X
        return self

    def query(self, Q: list[dict[int, float]], k: int):
        self._check_k(k)
        idx = np.full((len(Q), k), -1, dtype=np.int64)
        val = np.zeros((len(Q), k))
        for r, q in enumerate(Q):
            scored = []
            for i, d in enumerate(self.docs):
                small, large = (q, d) if len(q) <= len(d) else (d, q)
                s = sum(w * large[j] for j, w in small.items() if j in large)
                if s > 0.0:
                    scored.append((s, -i))
            for c, (s, negi) in enumerate(heapq.nlargest(k, scored)):
                idx[r, c], val[r, c] = -negi, s
        return idx, val


@register_index("sparse_exact")
class SparseExactIndex(SimilarityIndex):
    """Exact cosine search. Queries are processed in chunks so the dense score block
    (chunk x n_docs float64) never exceeds ``max_block_bytes``: bounded memory at any corpus size."""

    def __init__(self, max_block_bytes: int = 64 * 1024 * 1024):
        self.max_block_bytes = max_block_bytes

    def build(self, X: sparse.csr_matrix) -> "SparseExactIndex":
        self.docs = sparse.csr_matrix(X)
        self.docs_t = self.docs.T.tocsr()  # transposed once so each query block is one sparse x sparse product
        return self

    def query(self, Q: sparse.csr_matrix, k: int):
        self._check_k(k)
        Q = sparse.csr_matrix(Q)
        if Q.shape[1] != self.docs.shape[1]:
            raise ValueError(f"query has {Q.shape[1]} features, index has {self.docs.shape[1]}")
        n_docs = self.docs.shape[0]
        chunk = max(1, self.max_block_bytes // (8 * max(n_docs, 1)))
        idx = np.full((Q.shape[0], k), -1, dtype=np.int64)
        val = np.zeros((Q.shape[0], k))
        for a in range(0, Q.shape[0], chunk):
            block = (Q[a:a + chunk] @ self.docs_t).toarray()
            idx[a:a + chunk], val[a:a + chunk] = topk_rows(block, k)
        return idx, val


@register_index("lsh")
class LSHIndex(SimilarityIndex):
    """Random-hyperplane (SimHash) LSH for cosine similarity.

    ``n_tables`` hash tables of ``n_bits`` sign bits each; a document is a candidate if it shares a
    full bucket with the query in any table. Candidates are re-ranked with exact cosine, so scores
    are always exact; only *recall* is approximate. ``last_candidate_fraction`` records how much of
    the corpus was actually scored."""

    def __init__(self, n_bits: int = 6, n_tables: int = 16, seed: int = 0):
        if not 1 <= n_bits <= 62:
            raise ValueError("n_bits must be in [1, 62]")
        self.n_bits, self.n_tables, self.seed = n_bits, n_tables, seed

    def _keys(self, X: sparse.csr_matrix) -> np.ndarray:
        proj = np.asarray(X @ self.planes) > 0  # (n, tables*bits)
        bits = proj.reshape(X.shape[0], self.n_tables, self.n_bits)
        return (bits * (1 << np.arange(self.n_bits, dtype=np.int64))).sum(axis=2)  # (n, tables)

    def build(self, X: sparse.csr_matrix) -> "LSHIndex":
        self.docs = sparse.csr_matrix(X)
        rng = np.random.default_rng(self.seed)
        self.planes = rng.standard_normal((X.shape[1], self.n_tables * self.n_bits)).astype(np.float32)
        keys = self._keys(self.docs)
        self.tables = []
        for t in range(self.n_tables):
            order = np.argsort(keys[:, t], kind="stable")
            self.tables.append((keys[order, t], order))
        self.last_candidate_fraction = 0.0
        return self

    def query(self, Q: sparse.csr_matrix, k: int):
        self._check_k(k)
        Q = sparse.csr_matrix(Q)
        qkeys = self._keys(Q)
        idx = np.full((Q.shape[0], k), -1, dtype=np.int64)
        val = np.zeros((Q.shape[0], k))
        scored = 0
        for r in range(Q.shape[0]):
            parts = []
            for t, (skeys, order) in enumerate(self.tables):
                lo, hi = np.searchsorted(skeys, qkeys[r, t], "left"), np.searchsorted(skeys, qkeys[r, t], "right")
                parts.append(order[lo:hi])
            cand = np.unique(np.concatenate(parts))
            if cand.size == 0:
                continue
            scored += cand.size
            scores = (self.docs[cand] @ Q[r].T).toarray().ravel()
            top = topk_partition(scores, k)
            idx[r, :top.size], val[r, :top.size] = cand[top], scores[top]
        self.last_candidate_fraction = scored / (max(Q.shape[0], 1) * self.docs.shape[0])
        return idx, val
