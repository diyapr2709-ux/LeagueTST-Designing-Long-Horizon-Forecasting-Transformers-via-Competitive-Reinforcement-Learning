# E4 replay on E1 ETTh1 T=336

2368 configs, 74 architectures, pooled seed SD 0.0132; 50 configs lie within one SD of the best multi-seed mean.
Regret of the incumbent in pooled seed-SD units, scored on the seeds the search never saw where the incumbent has them and in-sample otherwise (held-out share per budget in summary.json): mean over search seeds with a 95% bootstrap interval. Budgets in full-run equivalents of 30 epochs. AUC is the mean regret over budgets 1 to 100.

| method | seeds | @10 | @25 | @50 | @100 | AUC |
|---|---|---|---|---|---|---|
| tpe_warm_from_ILI | 30 | 6.90 [6.37, 7.33] | 1.57 [1.35, 1.80] | 1.37 [1.24, 1.53] | 1.34 [1.20, 1.49] | 2.11 |
| portfolio_from_ILI | 30 | 7.40 [7.32, 7.48] | 7.35 [7.23, 7.48] | 7.21 [7.12, 7.31] | 3.91 [3.58, 4.30] | 5.94 |

No claims from this table: incumbents are scored in-sample; see the pick reseeds.
