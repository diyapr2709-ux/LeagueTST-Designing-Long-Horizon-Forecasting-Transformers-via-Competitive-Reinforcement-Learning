# E4 replay on E1 ILI T=24

2368 configs, 74 architectures, pooled seed SD 0.009787; 59 configs lie within one SD of the best multi-seed mean.
Regret of the incumbent, scored on the seeds the search never saw, in pooled seed-SD units: mean over search seeds with a 95% bootstrap interval. Budgets in full-run equivalents of 30 epochs. AUC is the mean regret over budgets 1 to 100.

| method | seeds | @10 | @25 | @50 | @100 | AUC |
|---|---|---|---|---|---|---|
| tpe_warm_from_ETTh1 | 30 | 2.98 [2.55, 3.41] | 1.62 [1.27, 2.01] | 1.47 [1.18, 1.78] | 1.09 [0.90, 1.31] | 1.65 |
| grpo_league_from_ETTh1 | 10 | 3.63 [2.99, 4.12] | 3.21 [2.67, 3.66] | 3.01 [2.42, 3.60] | 3.02 [2.30, 3.72] | 3.14 |
| grpo_none_from_ETTh1 | 10 | 3.65 [3.31, 3.97] | 3.19 [2.79, 3.62] | 3.41 [3.09, 3.77] | 2.93 [2.33, 3.52] | 3.23 |
| portfolio_from_ETTh1 | 30 | 3.88 [3.66, 4.10] | 2.65 [2.36, 2.97] | 3.29 [2.90, 3.69] | 3.67 [3.30, 4.02] | 3.28 |
