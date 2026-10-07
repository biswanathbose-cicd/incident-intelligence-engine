"""Held-out experiments on SYNTHETIC tickets. Run: PYTHONPATH=src python experiments/run_experiments.py"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from incident_engine import PipelineConfig, TriagePipeline, generate_incidents
from incident_engine.metrics import accuracy, macro_f1, retrieval_scores

OUT = Path(__file__).resolve().parent.parent / "results"
data = generate_incidents(20000, seed=7)
train, test = data[:16000], data[16000:]
res = {"n_train": len(train), "n_test": len(test), "note": "synthetic data; problem_id ground truth is generator-defined"}

# classification: majority baseline vs the two hand-written models
for task in ("category", "priority"):
    y_tr, y_te = [getattr(i, task) for i in train], [getattr(i, task) for i in test]
    maj = Counter(y_tr).most_common(1)[0][0]
    res[task] = {"majority": {"accuracy": accuracy(y_te, [maj] * len(y_te))}}
    for model in ("naive_bayes", "softmax"):
        kw = {"category_model": model, "priority_model": model}
        p = TriagePipeline(PipelineConfig(**kw)).fit(train)
        pred = [getattr(r, task) for r in p.triage_many([i.text for i in test], k=1)]
        res[task][model] = {"accuracy": accuracy(y_te, pred), "macro_f1": macro_f1(y_te, pred)}

# retrieval: a hit is relevant if it belongs to the same underlying problem
pid = {i.id: i.problem_id for i in train}
for name, params in (("sparse_exact", {}), ("lsh", {})):
    p = TriagePipeline(PipelineConfig(index=name, index_params=params)).fit(train)
    rs = p.triage_many([i.text for i in test], k=10)
    grp = np.full((len(rs), 10), -1)
    for r, x in enumerate(rs):
        for c, s in enumerate(x.similar):
            grp[r, c] = pid[s.incident_id]
    q = np.array([i.problem_id for i in test])
    res.setdefault("retrieval", {})[name] = {**retrieval_scores(grp, q, 10), "hit@1": retrieval_scores(grp, q, 1)["hit@1"]}
# random baseline: chance a random train ticket shares the problem
rng = np.random.default_rng(0)
tp = np.array([i.problem_id for i in test]); trp = np.array([i.problem_id for i in train])
res["retrieval"]["random_hit@10"] = float(np.mean([(trp[rng.integers(0, len(trp), 10)] == q).any() for q in tp]))

(OUT / "experiments.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
