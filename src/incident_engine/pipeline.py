"""End-to-end triage pipeline: text -> category, priority, similar past incidents, explanation.

Composition over inheritance: the pipeline owns a vectorizer, two classifiers and an index, each
chosen by name from ``PipelineConfig`` (Strategy + Factory), so swapping an algorithm is a config
change rather than a code change."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from . import models
from .data import Incident
from .index import SimilarityIndex, create_index
from .vectorize import SparseTfidf


@dataclass(frozen=True)
class PipelineConfig:
    category_model: str = "softmax"
    priority_model: str = "naive_bayes"
    index: str = "sparse_exact"
    index_params: dict = field(default_factory=dict)
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2


@dataclass(frozen=True)
class SimilarIncident:
    incident_id: str
    similarity: float
    text: str
    category: str


@dataclass(frozen=True)
class TriageResult:
    category: str
    category_confidence: float
    priority: str
    priority_confidence: float
    similar: list[SimilarIncident]
    top_terms: list[tuple[str, float]]


class TriagePipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.vectorizer = SparseTfidf(self.config.ngram_range, self.config.min_df)
        self.category_model = models.create(self.config.category_model)
        self.priority_model = models.create(self.config.priority_model)
        self.index: SimilarityIndex = create_index(self.config.index, **self.config.index_params)
        self._fitted = False

    def fit(self, incidents: Sequence[Incident]) -> "TriagePipeline":
        if not incidents:
            raise ValueError("cannot fit on zero incidents")
        X = self.vectorizer.fit_transform([i.text for i in incidents])
        self.category_model.fit(X, [i.category for i in incidents])
        self.priority_model.fit(X, [i.priority for i in incidents])
        self.index.build(X)
        self.incidents = list(incidents)
        self._fitted = True
        return self

    def triage(self, text: str, k: int = 5) -> TriageResult:
        return self.triage_many([text], k)[0]

    def triage_many(self, texts: Sequence[str], k: int = 5) -> list[TriageResult]:
        """Batched: one vectorization and one index query for all texts (amortizes Python overhead)."""
        if not self._fitted:
            raise models.NotFittedError("pipeline is not fitted")
        if any(not t or not t.strip() for t in texts):
            raise ValueError("incident text must be non-empty")
        X = self.vectorizer.transform(texts)
        cat_p, pri_p = self.category_model.predict_proba(X), self.priority_model.predict_proba(X)
        idx, sim = self.index.query(X, k)
        out = []
        for r in range(X.shape[0]):
            ci, pi = int(cat_p[r].argmax()), int(pri_p[r].argmax())
            category = str(self.category_model.classes_[ci])
            similar = [SimilarIncident(self.incidents[j].id, float(s), self.incidents[j].text, self.incidents[j].category)
                       for j, s in zip(idx[r], sim[r]) if j >= 0]
            out.append(TriageResult(
                category, float(cat_p[r, ci]), str(self.priority_model.classes_[pi]), float(pri_p[r, pi]), similar,
                self.category_model.explain(X[r], category, self.vectorizer.terms_)))
        return out
