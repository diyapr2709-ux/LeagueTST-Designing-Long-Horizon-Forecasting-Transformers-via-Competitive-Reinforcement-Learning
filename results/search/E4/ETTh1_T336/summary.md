# E4 replay on E1 ETTh1 T=336

2368 configs, 74 architectures, pooled seed SD 0.0132; 50 configs lie within one SD of the best multi-seed mean.
Regret of the incumbent in pooled seed-SD units, scored on the seeds the search never saw where the incumbent has them and in-sample otherwise (held-out share per budget in summary.json): mean over search seeds with a 95% bootstrap interval. Budgets in full-run equivalents of 30 epochs. AUC is the mean regret over budgets 1 to 100.

| method | seeds | @10 | @25 | @50 | @100 | AUC |
|---|---|---|---|---|---|---|
| tpe_warm_from_ILI | 30 | 6.90 [6.37, 7.33] | 1.57 [1.35, 1.80] | 1.37 [1.24, 1.53] | 1.34 [1.20, 1.49] | 2.11 |
| grpo_uniform_from_ILI | 5 | 4.35 [1.92, 6.78] | 3.04 [1.37, 5.00] | 1.45 [0.63, 2.74] | 1.08 [0.25, 2.06] | 2.22 |
| grpo_league_from_ILI | 10 | 3.67 [1.95, 5.43] | 3.35 [1.84, 5.06] | 3.17 [1.68, 4.87] | 2.62 [1.38, 4.23] | 3.51 |
| grpo_none_from_ILI | 10 | 5.23 [3.51, 6.89] | 4.78 [3.03, 6.49] | 3.75 [2.15, 5.40] | 2.73 [1.54, 4.29] | 4.04 |
| portfolio_from_ILI | 30 | 7.40 [7.32, 7.48] | 7.35 [7.23, 7.48] | 7.21 [7.12, 7.31] | 3.91 [3.58, 4.30] | 5.94 |

No claims from this table: incumbents are scored in-sample; see the pick reseeds.
