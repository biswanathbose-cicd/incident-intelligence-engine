"""Three ways to pick the top-k scores, with one shared contract so they can be tested for equality
and benchmarked against each other.

Contract: return indices of the (at most) ``k`` highest scores that are strictly greater than
``min_score``, ordered by score descending and, for exactly equal scores, by index ascending.

    topk_sort       O(n log n)        full sort
    topk_heap       O(n log k)        pure-Python heap (``heapq.nlargest``)
    topk_partition  O(n + m log m)    threshold + selection on the m candidates; the default
"""

from __future__ import annotations

import heapq

import numpy as np


def _empty() -> np.ndarray:
    return np.empty(0, dtype=np.int64)


def topk_sort(scores: np.ndarray, k: int, min_score: float = 0.0) -> np.ndarray:
    if k <= 0 or scores.size == 0:
        return _empty()
    order = np.lexsort((np.arange(scores.size), -scores))  # last key is primary: score desc, then index asc
    order = order[scores[order] > min_score]
    return order[:k].astype(np.int64)


def topk_heap(scores: np.ndarray, k: int, min_score: float = 0.0) -> np.ndarray:
    if k <= 0 or scores.size == 0:
        return _empty()
    best = heapq.nlargest(k, ((s, -i) for i, s in enumerate(scores.tolist()) if s > min_score))
    return np.asarray([-negi for _, negi in best], dtype=np.int64)


def topk_partition(scores: np.ndarray, k: int, min_score: float = 0.0) -> np.ndarray:
    if k <= 0 or scores.size == 0:
        return _empty()
    pos = np.flatnonzero(scores > min_score)  # most documents score 0 for sparse text: drop them first
    if pos.size == 0:
        return _empty()
    if pos.size > k:
        vals = scores[pos]
        kth = np.partition(vals, pos.size - k)[pos.size - k]  # k-th largest value
        keep = vals >= kth  # includes every tie at the boundary so the tie-break below is exact
        pos, vals = pos[keep], vals[keep]
    else:
        vals = scores[pos]
    order = np.lexsort((pos, -vals))  # pos is ascending already; lexsort keeps index ascending on ties
    return pos[order][:k].astype(np.int64)


def topk_rows(S: np.ndarray, k: int, min_score: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise top-k over a dense score matrix. Returns (indices, scores) padded with (-1, 0.0)."""
    idx = np.full((S.shape[0], k), -1, dtype=np.int64)
    val = np.zeros((S.shape[0], k), dtype=np.float64)
    for r in range(S.shape[0]):
        top = topk_partition(S[r], k, min_score)
        idx[r, :top.size] = top
        val[r, :top.size] = S[r, top]
    return idx, val
