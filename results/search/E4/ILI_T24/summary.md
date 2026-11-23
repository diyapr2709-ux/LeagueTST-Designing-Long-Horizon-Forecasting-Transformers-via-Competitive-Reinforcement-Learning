# E4 replay on E1 ILI T=24

2368 configs, 74 architectures, pooled seed SD 0.009787; 59 configs lie within one SD of the best multi-seed mean.
Regret of the incumbent, scored on the seeds the search never saw, in pooled seed-SD units: mean over search seeds with a 95% bootstrap interval. Budgets in full-run equivalents of 30 epochs. AUC is the mean regret over budgets 1 to 100.

| method | seeds | @10 | @25 | @50 | @100 | AUC |
|---|---|---|---|---|---|---|
| tpe_warm_from_ETTh1 | 30 | 2.98 [2.55, 3.41] | 1.62 [1.27, 2.01] | 1.47 [1.18, 1.78] | 1.09 [0.90, 1.31] | 1.65 |
| portfolio_from_ETTh1 | 30 | 3.88 [3.66, 4.10] | 2.65 [2.36, 2.97] | 3.29 [2.90, 3.69] | 3.67 [3.30, 4.02] | 3.28 |
