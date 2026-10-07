"""FastAPI service: ``uvicorn --factory incident_engine.api:create_app``.

NOT executed in the sandbox this repository was written in (no package index access); the test in
``tests/test_models_index_pipeline.py::ApiTests`` exercises it and runs in CI."""

from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .data import generate_incidents
from .pipeline import TriagePipeline


class TriageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    k: int = Field(default=5, ge=1, le=20)


def create_app(n_train: int | None = None) -> FastAPI:
    n = n_train or int(os.environ.get("TRAIN_INCIDENTS", "20000"))
    pipeline = TriagePipeline().fit(generate_incidents(n))  # demo model trained on synthetic data at start-up
    app = FastAPI(title="Incident Intelligence Engine", version="1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "trained_on": n}

    @app.post("/triage")
    def triage(req: TriageRequest) -> dict:
        r = pipeline.triage(req.text, req.k)
        return {
            "category": r.category, "category_confidence": r.category_confidence,
            "priority": r.priority, "priority_confidence": r.priority_confidence,
            "top_terms": [{"term": t, "weight": w} for t, w in r.top_terms],
            "similar": [s.__dict__ for s in r.similar],
        }

    return app
