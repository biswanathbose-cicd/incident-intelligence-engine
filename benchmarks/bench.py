"""Micro-benchmarks with real measurements. Run: PYTHONPATH=src python benchmarks/bench.py
Writes results/benchmarks.json and results/benchmarks.md. Timings are machine-dependent; ratios are the point."""
from __future__ import annotations

import cProfile, io, json, platform, pstats, time, tracemalloc
from pathlib import Path

import numpy as np

from incident_engine import generate_incidents
from incident_engine.index import LSHIndex, ReferenceIndex, SparseExactIndex
from incident_engine.metrics import overlap_at_k, score_ratio_at_k
from incident_engine.topk import topk_heap, topk_partition, topk_sort
from incident_engine.vectorize import ReferenceTfidf, SparseTfidf

OUT = Path(__file__).resolve().parent.parent / "results"


def best(fn, repeat=3):
    ts = []
    for _ in range(repeat):
        t = time.perf_counter(); r = fn(); ts.append(time.perf_counter() - t)
    return min(ts), r


def main():
    res: dict = {"python": platform.python_version(), "machine": platform.machine()}
    texts = [i.text for i in generate_incidents(12000, seed=1)]

    n = 4000
    t_ref, _ = best(lambda: ReferenceTfidf((1, 2), 2).fit_transform(texts), 2)
    t_two, _ = best(lambda: (lambda v: (v.fit(texts), v.transform(texts)))(SparseTfidf((1, 2), 2)), 2)
    t_one, _ = best(lambda: SparseTfidf((1, 2), 2).fit_transform(texts), 2)
    from sklearn.feature_extraction.text import TfidfVectorizer
    t_sk, _ = best(lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=2, token_pattern=r"[a-z0-9]+").fit_transform(texts), 2)
    res["vectorize_12k_docs_s"] = {"reference_pure_python": t_ref,
                                  "sparse_fit_then_transform(tokenizes twice)": t_two,
                                  "sparse_fit_transform(tokenizes once)": t_one, "sklearn": t_sk}

    rng = np.random.default_rng(0)
    topk = {}
    for size in (10_000, 100_000, 1_000_000):
        s = rng.random(size) * (rng.random(size) > 0.9)
        row = {}
        for name, f in (("sort", topk_sort), ("heap", topk_heap), ("partition", topk_partition)):
            if name == "heap" and size > 100_000:
                continue
            row[name] = best(lambda: f(s, 10), 3)[0]
        topk[str(size)] = row
    res["topk_k10_s"] = topk

    vec = SparseTfidf((1, 2), 2)
    X = vec.fit_transform(texts)
    docs, Q = X[:11000], X[11000:11500]
    exact = SparseExactIndex().build(docs)
    t_exact, (ei, es) = best(lambda: exact.query(Q, 10), 3)
    lsh = LSHIndex().build(docs)
    t_lsh, (li, ls) = best(lambda: lsh.query(Q, 10), 2)
    ref_vec = ReferenceTfidf((1, 2), 2); rd = ref_vec.fit_transform(texts[:1000]); rq = ref_vec.transform(texts[11000:11020])
    t_ref_q, _ = best(lambda: ReferenceIndex().build(rd).query(rq, 10), 1)
    res["search_500_queries_11k_docs"] = {"sparse_exact_s": t_exact, "lsh_6x16_s": t_lsh,
        "lsh_score_ratio": score_ratio_at_k(ls, es), "lsh_overlap": overlap_at_k(li, ei),
        "lsh_candidate_fraction": lsh.last_candidate_fraction,
        "reference_ms_per_query_1k_docs": t_ref_q / 20 * 1000, "sparse_exact_ms_per_query_11k_docs": t_exact / 500 * 1000}

    mem = {}
    for budget in (256 * 1024 * 1024, 4 * 1024 * 1024):
        ix = SparseExactIndex(max_block_bytes=budget).build(docs)
        tracemalloc.start(); ix.query(Q, 10); _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        mem[f"block_budget_{budget // 2**20}MiB"] = {"peak_MiB": peak / 2**20}
    res["peak_memory_500q_11k_docs"] = mem

    pr = cProfile.Profile(); pr.enable()
    SparseTfidf((1, 2), 2).fit_transform(texts)
    pr.disable()
    buf = io.StringIO(); pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(8)
    (OUT / "profile_vectorizer.txt").write_text(buf.getvalue())

    (OUT / "benchmarks.json").write_text(json.dumps(res, indent=2))
    v = res["vectorize_12k_docs_s"]; s = res["search_500_queries_11k_docs"]
    md = [f"# Benchmarks (measured; Python {res['python']})\n", "| Vectorize 12k tickets, 1-2 grams | seconds |", "|---|---|"]
    md += [f"| {k} | {x:.2f} |" for k, x in v.items()]
    md += ["", "| Top-10 of n scores (ms) | " + " | ".join(topk) + " |", "|---|---|---|---|"]
    for name in ("sort", "heap", "partition"):
        md.append(f"| {name} | " + " | ".join(f"{topk[z][name]*1000:.2f}" if name in topk[z] else "-" for z in topk) + " |")
    md += ["", "| Search (500 queries x 11k docs) | value |", "|---|---|"]
    md += [f"| {k} | {x:.4g} |" for k, x in s.items()]
    md += ["", "| Peak memory, 500 queries | MiB |", "|---|---|"]
    md += [f"| {k} | {x['peak_MiB']:.1f} |" for k, x in mem.items()]
    (OUT / "benchmarks.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
