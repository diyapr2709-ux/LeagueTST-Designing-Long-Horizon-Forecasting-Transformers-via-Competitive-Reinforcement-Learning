# E3 replay on E1 ETTh1 T=336

2368 configs, 74 architectures, pooled seed SD 0.0132; 50 configs lie within one SD of the best multi-seed mean.
Regret of the incumbent in pooled seed-SD units, scored on the seeds the search never saw where the incumbent has them and in-sample otherwise (held-out share per budget in summary.json): mean over search seeds with a 95% bootstrap interval. Budgets in full-run equivalents of 30 epochs. AUC is the mean regret over budgets 1 to 100.

| method | seeds | @10 | @25 | @50 | @100 | AUC |
|---|---|---|---|---|---|---|
| qwen_prompted | 10 | 1.80 [1.13, 2.52] | 1.16 [0.94, 1.37] | 1.16 [0.94, 1.39] | 1.31 [1.17, 1.47] | 1.30 |
| pg | 30 | 1.24 [1.02, 1.48] | 1.16 [0.95, 1.37] | 1.21 [1.07, 1.35] | 1.26 [1.14, 1.38] | 1.32 |
| tpe | 30 | 1.58 [1.37, 1.81] | 1.19 [0.97, 1.39] | 1.17 [0.97, 1.38] | 1.18 [1.00, 1.36] | 1.37 |
| pg_league_charged | 30 | 1.37 [1.13, 1.62] | 1.32 [1.10, 1.55] | 1.23 [1.05, 1.44] | 1.30 [1.16, 1.43] | 1.39 |
| pg_uniform_rivals | 30 | 1.24 [1.02, 1.48] | 1.30 [1.13, 1.48] | 1.31 [1.17, 1.45] | 1.28 [1.18, 1.39] | 1.39 |
| pg_league | 30 | 1.24 [1.02, 1.48] | 1.26 [0.99, 1.56] | 1.37 [1.21, 1.54] | 1.26 [1.16, 1.35] | 1.41 |
| re | 30 | 1.91 [1.60, 2.23] | 1.51 [1.28, 1.75] | 1.34 [1.18, 1.50] | 1.37 [1.17, 1.58] | 1.57 |
| asha | 30 | 1.87 [1.44, 2.38] | 1.46 [1.26, 1.66] | 1.35 [1.20, 1.50] | 1.41 [1.27, 1.54] | 1.57 |
| random | 30 | 1.91 [1.60, 2.23] | 1.46 [1.18, 1.75] | 1.53 [1.18, 1.94] | 1.51 [1.26, 1.81] | 1.61 |
| referee | 30 | 1.42 [1.21, 1.65] | 1.36 [1.14, 1.58] | 1.62 [1.39, 1.91] | 1.44 [1.25, 1.72] | 1.63 |
| epochk | 30 | 1.42 [1.21, 1.65] | 1.59 [1.25, 2.01] | 1.63 [1.39, 1.92] | 1.41 [1.22, 1.68] | 1.68 |

No claims from this table: incumbents are scored in-sample; see the pick reseeds.
