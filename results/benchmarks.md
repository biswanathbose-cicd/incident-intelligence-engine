# Benchmarks (measured; Python 3.13.16)

| Vectorize 12k tickets, 1-2 grams | seconds |
|---|---|
| reference_pure_python | 0.25 |
| sparse_fit_then_transform(tokenizes twice) | 0.21 |
| sparse_fit_transform(tokenizes once) | 0.14 |
| sklearn | 0.17 |

| Top-10 of n scores (ms) | 10000 | 100000 | 1000000 |
|---|---|---|---|
| sort | 0.14 | 1.78 | 22.65 |
| heap | 0.42 | 4.46 | - |
| partition | 0.03 | 0.21 | 1.44 |

| Search (500 queries x 11k docs) | value |
|---|---|
| sparse_exact_s | 0.08252 |
| lsh_6x16_s | 0.5296 |
| lsh_score_ratio | 0.9639 |
| lsh_overlap | 0.768 |
| lsh_candidate_fraction | 0.2482 |
| reference_ms_per_query_1k_docs | 1.204 |
| sparse_exact_ms_per_query_11k_docs | 0.165 |

| Peak memory, 500 queries | MiB |
|---|---|
| block_budget_256MiB | 87.6 |
| block_budget_4MiB | 12.5 |
