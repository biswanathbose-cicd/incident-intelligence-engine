# Incident Intelligence Engine

A small, hand-built ML system that **triages IT-service-management incidents**: it predicts a ticket's category and priority, explains the decision with the terms that drove it, and retrieves similar past incidents. The point of the project is the engineering underneath: algorithms written from scratch, checked against references, profiled and benchmarked.

> **Data is synthetic.** Tickets come from a seeded generator (`data.py`) with typos, filler text, cross-category confusers, vague tickets and a deliberately noisy priority label. Accuracy numbers describe *this generator*, not real ITSM data. The generator's `problem_id` is the retrieval ground truth.

## What's inside

| Piece | What it demonstrates |
|---|---|
| `vectorize.py` | TF-IDF twice: a pure-Python reference and a sparse CSR version. Both match scikit-learn exactly (tested). |
| `models.py` | Multinomial Naive Bayes and softmax regression (mini-batch Adam) written from scratch. NB matches `sklearn.MultinomialNB`; the softmax gradient is verified by finite differences. |
| `topk.py` | Top-k by sort O(n log n), heap O(n log k), partition O(n + m log m) with an explicit deterministic tie-break contract. |
| `index.py` | Similar-incident search: brute-force reference, chunked exact sparse search (bounded memory), random-hyperplane LSH with exact re-ranking. |
| `pipeline.py` | `TriagePipeline` composed from a `PipelineConfig`; algorithms are swapped by name. |
| `api.py`, `Dockerfile` | FastAPI service and container (**not executed here**, see limitations). |

Design patterns: Template Method (`Classifier.fit/predict`), Strategy + Registry/Factory (`models.create`, `create_index`), composition for the pipeline.

## Measured results (`results/`, reproducible with the scripts)

Held-out split: 16,000 train / 4,000 test synthetic tickets.

| Task | Majority | Naive Bayes | Softmax |
|---|---|---|---|
| Category accuracy | 16.8% | **97.6%** | 97.0% |
| Priority accuracy (noisy labels) | 41.7% | **75.3%** | 73.6% |

Similar-incident retrieval (hit = same underlying problem): exact search hit@1 74.6%, hit@10 91.3%, MRR 0.804; random baseline hit@10 3.1%.

Performance (12k tickets / 11k docs x 500 queries, one machine, Python 3.13; see `results/benchmarks.md`):
- Top-10 of 1M scores: partition 1.4 ms vs full sort 23 ms (about 16x); heap is slowest in Python at these sizes because of interpreter overhead.
- Vectorizing 12k tickets: tokenizing once (`fit_transform`) 0.14 s vs fit-then-transform 0.21 s; pure-Python reference 0.25 s; scikit-learn 0.17 s. The profile (`results/profile_vectorizer.txt`) shows the time is dominated by Python-level n-gram generation and CSR assembly, so the gains from vectorization alone are modest on short tickets.
- Chunked exact search keeps peak memory bounded: 87.6 MiB with a 256 MiB block budget vs 12.5 MiB with a 4 MiB budget, same results.
- **LSH did not pay off at this scale**: it scored ~25% of the corpus, reached score ratio 0.96 (set overlap 0.77) and was ~6.5x *slower* than exact sparse search (0.55 s vs 0.084 s). LSH starts to matter when the corpus is much larger than what one sparse product handles; this repo shows the trade-off rather than hiding it.

## Run it

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests -v          # 40 tests
PYTHONPATH=src python benchmarks/bench.py
PYTHONPATH=src python experiments/run_experiments.py
PYTHONPATH=src python -m incident_engine.cli "vpn keeps disconnecting error 809"
```

## Limitations (read before quoting numbers)
- Synthetic data only; real tickets are messier and the numbers would be lower.
- Priority labels are noisy by construction (20% flipped), so ~75% is near the ceiling here, not a weakness of the model.
- The FastAPI service, Dockerfile and CI workflow were written without package-index access, so they have **not been run**. The API test is skipped when fastapi/httpx are missing.
- Naive Bayes `alpha=0.1` was chosen on a validation split of the same synthetic distribution.
- Benchmarks are single-machine timings; compare ratios, not absolute seconds.
