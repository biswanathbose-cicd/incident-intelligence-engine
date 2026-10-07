"""Incident Intelligence Engine: ITSM ticket triage + similar-incident search.

Hand-written, performance-minded ML components (TF-IDF, Naive Bayes, softmax regression, top-k
selection, exact and LSH nearest-neighbour search) behind small, swappable interfaces."""

from .data import CATEGORIES, Incident, generate_incidents
from .pipeline import PipelineConfig, TriagePipeline, TriageResult

__all__ = ["CATEGORIES", "Incident", "generate_incidents", "PipelineConfig", "TriagePipeline", "TriageResult"]
