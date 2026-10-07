"""Classifiers behind one interface.

Design patterns used (and why):
  * Template Method - ``Classifier.fit/predict*`` validate inputs and handle label encoding once;
    subclasses implement only ``_fit``, ``_joint_log_proba`` and ``_term_scores``.
  * Strategy + Registry/Factory - ``create("naive_bayes")`` lets the pipeline pick an algorithm from
    configuration without importing concrete classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.special import logsumexp


class NotFittedError(RuntimeError):
    pass


_REGISTRY: dict[str, type["Classifier"]] = {}


def register(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        if name in _REGISTRY:
            raise ValueError(f"classifier '{name}' already registered")
        _REGISTRY[name] = cls
        cls.registry_name = name
        return cls
    return deco


def create(name: str, **params) -> "Classifier":
    try:
        return _REGISTRY[name](**params)
    except KeyError:
        raise ValueError(f"unknown classifier '{name}'; available: {sorted(_REGISTRY)}") from None


def available() -> list[str]:
    return sorted(_REGISTRY)


class Classifier(ABC):
    classes_: np.ndarray

    # -- public API (template methods) ------------------------------------------------------
    def fit(self, X, y) -> "Classifier":
        X = self._as_csr(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]} labels")
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        if self.classes_.size < 2:
            raise ValueError("need at least two classes to fit a classifier")
        self.n_features_ = X.shape[1]
        self._fit(X, y_idx)
        self._fitted = True
        return self

    def predict_log_proba(self, X) -> np.ndarray:
        X = self._check(X)
        jll = self._joint_log_proba(X)
        return jll - logsumexp(jll, axis=1, keepdims=True)

    def predict_proba(self, X) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X) -> np.ndarray:
        X = self._check(X)
        return self.classes_[np.argmax(self._joint_log_proba(X), axis=1)]

    def explain(self, x_row: sparse.csr_matrix, class_label, terms: np.ndarray, top: int = 5) -> list[tuple[str, float]]:
        """Terms of one document that push the decision towards ``class_label`` the most."""
        self._check(x_row)
        c = int(np.flatnonzero(self.classes_ == class_label)[0])
        x_row = sparse.csr_matrix(x_row)
        contrib = x_row.data * self._term_scores(c)[x_row.indices]
        order = np.argsort(-contrib)[:top]
        return [(str(terms[x_row.indices[i]]), float(contrib[i])) for i in order if contrib[i] > 0]

    # -- hooks ----------------------------------------------------------------------------------
    @abstractmethod
    def _fit(self, X: sparse.csr_matrix, y_idx: np.ndarray) -> None: ...

    @abstractmethod
    def _joint_log_proba(self, X: sparse.csr_matrix) -> np.ndarray: ...

    @abstractmethod
    def _term_scores(self, class_idx: int) -> np.ndarray:
        """Per-feature evidence for a class, relative to the other classes."""

    # -- helpers ------------------------------------------------------------------------------
    @staticmethod
    def _as_csr(X) -> sparse.csr_matrix:
        if not sparse.issparse(X):
            X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        X = sparse.csr_matrix(X, dtype=np.float64)
        if X.shape[0] == 0:
            raise ValueError("X has no rows")
        return X

    def _check(self, X) -> sparse.csr_matrix:
        if not getattr(self, "_fitted", False):
            raise NotFittedError(f"{type(self).__name__} is not fitted")
        X = self._as_csr(X)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"expected {self.n_features_} features, got {X.shape[1]}")
        return X


@register("naive_bayes")
class MultinomialNaiveBayes(Classifier):
    """Multinomial NB on (tf-idf) feature mass. Training is two sparse matrix products: O(nnz).

    ``alpha`` defaults to 0.1, not the textbook 1.0: tf-idf mass per class is small relative to the
    vocabulary size, so Laplace smoothing swamps the evidence (validation-split sweep on the synthetic
    data: category 96.2% -> 97.4%, priority 70.4% -> 73.4% going from alpha=1.0 to 0.1)."""

    def __init__(self, alpha: float = 0.1):
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.alpha = alpha

    def _fit(self, X, y_idx):
        k, n = self.classes_.size, X.shape[0]
        onehot = sparse.csr_matrix((np.ones(n), (y_idx, np.arange(n))), shape=(k, n))  # k x n
        feature_mass = (onehot @ X).toarray()  # k x d
        smoothed = feature_mass + self.alpha
        self.feature_log_prob_ = np.log(smoothed) - np.log(smoothed.sum(axis=1, keepdims=True))
        self.class_log_prior_ = np.log(np.bincount(y_idx, minlength=k) / n)

    def _joint_log_proba(self, X):
        return np.asarray(X @ self.feature_log_prob_.T) + self.class_log_prior_

    def _term_scores(self, c):
        return self.feature_log_prob_[c] - self.feature_log_prob_.mean(axis=0)


@register("softmax")
class SoftmaxRegression(Classifier):
    """Multinomial logistic regression trained with mini-batch Adam; gradients cost O(nnz * k)."""

    def __init__(self, l2: float = 1e-5, epochs: int = 25, batch_size: int = 256, lr: float = 0.05, seed: int = 0):
        self.l2, self.epochs, self.batch_size, self.lr, self.seed = l2, epochs, batch_size, lr, seed

    @staticmethod
    def loss_and_grad(W: np.ndarray, b: np.ndarray, X: sparse.csr_matrix, y_idx: np.ndarray, l2: float):
        """Mean cross-entropy + 0.5*l2*||W||^2 and its gradients (checked numerically in the tests)."""
        n = X.shape[0]
        logits = np.asarray(X @ W) + b
        logp = logits - logsumexp(logits, axis=1, keepdims=True)
        loss = -logp[np.arange(n), y_idx].mean() + 0.5 * l2 * np.sum(W * W)
        d = np.exp(logp)
        d[np.arange(n), y_idx] -= 1.0
        d /= n
        gW = np.asarray(X.T @ d) + l2 * W
        return loss, gW, d.sum(axis=0)

    def _fit(self, X, y_idx):
        n, d = X.shape
        k = self.classes_.size
        rng = np.random.default_rng(self.seed)
        W, b = np.zeros((d, k)), np.zeros(k)
        mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        self.loss_history_: list[float] = []
        for _ in range(self.epochs):
            perm, total = rng.permutation(n), 0.0
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                loss, gW, gb = self.loss_and_grad(W, b, X[idx], y_idx[idx], self.l2)
                total += loss * idx.size
                t += 1
                mW = b1 * mW + (1 - b1) * gW
                vW = b2 * vW + (1 - b2) * gW * gW
                mb = b1 * mb + (1 - b1) * gb
                vb = b2 * vb + (1 - b2) * gb * gb
                c1, c2 = 1 - b1 ** t, 1 - b2 ** t
                W -= self.lr * (mW / c1) / (np.sqrt(vW / c2) + eps)
                b -= self.lr * (mb / c1) / (np.sqrt(vb / c2) + eps)
            self.loss_history_.append(total / n)
        self.coef_, self.intercept_ = W, b

    def _joint_log_proba(self, X):
        return np.asarray(X @ self.coef_) + self.intercept_

    def _term_scores(self, c):
        return self.coef_[:, c] - self.coef_.mean(axis=1)
