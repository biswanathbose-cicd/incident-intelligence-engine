"""Evaluation metrics (small, dependency-free, unit-tested)."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and the same length")
    return float((y_true == y_pred).mean())


def macro_f1(y_true: Sequence, y_pred: Sequence) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for c in np.unique(np.concatenate([y_true, y_pred])):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(scores))


def retrieval_scores(retrieved_groups: np.ndarray, query_groups: np.ndarray, k: int) -> dict[str, float]:
    """``retrieved_groups[i, r]`` is the group id of the rank-r hit for query i (-1 = no hit).
    A hit is relevant iff its group equals the query's group. Returns hit@k and MRR."""
    rel = retrieved_groups[:, :k] == query_groups[:, None]
    hit = rel.any(axis=1)
    first = np.where(hit, rel.argmax(axis=1) + 1, 0)
    mrr = np.where(hit, 1.0 / np.maximum(first, 1), 0.0)
    return {f"hit@{k}": float(hit.mean()), "mrr": float(mrr.mean())}


def overlap_at_k(approx_idx: np.ndarray, exact_idx: np.ndarray) -> float:
    """Mean fraction of the exact top-k that an approximate search also returned."""
    fracs = []
    for a, e in zip(approx_idx, exact_idx):
        e = set(e[e >= 0].tolist())
        if e:
            fracs.append(len(e & set(a[a >= 0].tolist())) / len(e))
    return float(np.mean(fracs)) if fracs else 1.0


def score_ratio_at_k(approx_scores: np.ndarray, exact_scores: np.ndarray) -> float:
    """Mean over queries of sum(approx top-k scores) / sum(exact top-k scores). Unlike set overlap it is
    not distorted by exact ties (many incidents with identical similarity), 1.0 = as good as exact."""
    ratios = [a.sum() / e.sum() for a, e in zip(approx_scores, exact_scores) if e.sum() > 0]
    return float(np.mean(ratios)) if ratios else 1.0
