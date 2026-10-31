# Progress ledger

The proposal plans twelve weeks of work. This ledger assigns each component a
weight so that progress can be stated as one number. Weights sum to 100. The
first stage, worth 18 points, is complete, and so is component 7, the full
baseline reproduction.

## Components

| # | Component | Weight | State |
|---|---|---|---|
| 1 | Repository scaffold, environment, test harness | 2 | done |
| 2 | Data pipeline: ETTh1 and ILI loading, splits, scaling, windows, loaders | 5 | done |
| 3 | Configurable PatchTST with the nine design fields, RevIN, attention capture | 5 | done |
| 4 | DLinear baseline | 2 | done |
| 5 | Trainer: Adam, official schedules, early stopping, exact full-split metrics, run records | 3 | done |
| 6 | Preliminary reproduction at the shortest horizon, three seeds | 1 | done |
| 7 | Full reproduction of PatchTST/42 and DLinear at all horizons on both benchmarks | 6 | done |
| 8 | Random search baseline over the design space | 5 | open |
| 9 | Regularized evolution baseline | 5 | open |
| 10 | Architect: Qwen2.5-Coder-1.5B with LoRA, JSON config generation, GRPO training | 18 | open |
| 11 | League pool of frozen past architects added to every group | 6 | open |
| 12 | Learned referee: one-epoch features, XGBoost or MLP, retraining loop | 10 | open |
| 13 | Referee audit: periodic full training of random candidates, Kendall tau on random versus architect proposals | 3 | open |
| 14 | Final training of the winning design at all horizons on both benchmarks | 8 | open |
| 15 | Attention analysis against PatchTST (RQ3) | 8 | open |
| 16 | Analysis of which design choices matter (RQ3) | 5 | open |
| 17 | Report and presentation | 8 | open |

Done: 24 of 100.

## What the first stage covers

* `leaguetst/data`: both benchmarks with the official split borders,
  training-set standardisation, and sliding windows. Validation and test
  loaders keep every window.
* `leaguetst/models`: `ForecasterConfig` holds the nine design choices and
  validates them. `PatchTST` reads the config, supports channel-independent
  and channel-mixing input, and can keep attention maps for the later
  analysis. `DLinear` matches the authors' model with a shared linear map.
* `leaguetst/training`: Adam with the official `constant`, `type1` and
  `type3` schedules, the optimiser state the official PatchTST trainer
  leaves behind (first beta 0.95, first epoch at 1/25 of the rate), early
  stopping on validation MSE, best-checkpoint restore, and
  exact MSE and MAE over the whole split next to the official-style values
  that drop the last partial batch.
* `leaguetst/experiment.py`: one run from spec to JSON record, with the
  best weights saved under `checkpoints/` for the later attention analysis.
* `leaguetst/search/space.py`: the 30,720-point grid, enumeration and
  uniform sampling. The search methods themselves are not started.
* `scripts/`: data download, single run, baseline sweep, summary table.
* `tests/`: 90 tests (36 at the end of the first stage, 54 added by the
  hardening rounds in `docs/validation_log.md`) on synthetic copies of both benchmark files plus shape
  checks on the real files when present.

## Baseline results

Components 6 and 7 reproduce PatchTST/42 and DLinear at every horizon of
both benchmarks with three seeds (2021, 2022, 2023). The exact column is
the project protocol, the mean over every test window. The drop-last
column keeps only full test batches, which is what both official loaders
did when the published numbers were made, so it is the like-for-like
comparison. Standard deviations are over the three seeds. The full table
is `results/baselines/summary.md`, built from `results/preliminary` and
`results/baselines` together; bootstrap intervals, error by step and by
variable, naive baselines and the paired comparison are in
`results/revalidation/report_round3.md`.

| model | dataset | T | params | test MSE exact | test MSE drop-last | published MSE | test MAE drop-last | published MAE |
|---|---|---|---|---|---|---|---|---|
| PatchTST/42 | ETTh1 | 96 | 81,728 | 0.3791 ± 0.0006 | 0.3754 ± 0.0005 | 0.375 | 0.3998 ± 0.0004 | 0.399 |
| PatchTST/42 | ETTh1 | 192 | 146,336 | 0.4130 ± 0.0013 | 0.4129 ± 0.0013 | 0.414 | 0.4205 ± 0.0007 | 0.421 |
| PatchTST/42 | ETTh1 | 336 | 243,248 | 0.4395 ± 0.0054 | 0.4293 ± 0.0050 | 0.431 | 0.4347 ± 0.0042 | 0.436 |
| PatchTST/42 | ETTh1 | 720 | 501,680 | 0.4551 ± 0.0052 | 0.4483 ± 0.0052 | 0.449 | 0.4653 ± 0.0037 | 0.466 |
| PatchTST/42 | ILI | 24 | 33,400 | 1.9289 ± 0.0309 | 1.4867 ± 0.0614 | 1.522 | 0.8106 ± 0.0292 | 0.814 |
| PatchTST/42 | ILI | 36 | 41,476 | 1.8258 ± 0.0827 | 1.4086 ± 0.0310 | 1.430 | 0.8004 ± 0.0266 | 0.834 |
| PatchTST/42 | ILI | 48 | 49,552 | 1.7699 ± 0.1011 | 1.7090 ± 0.1100 | 1.673 | 0.8598 ± 0.0452 | 0.854 |
| PatchTST/42 | ILI | 60 | 57,628 | 1.7981 ± 0.1911 | 1.6239 ± 0.2221 | 1.529 | 0.8558 ± 0.0512 | 0.862 |
| DLinear | ETTh1 | 96 | 64,704 | 0.3811 ± 0.0133 | 0.3810 ± 0.0133 | 0.375 | 0.4050 ± 0.0159 | 0.399 |
| DLinear | ETTh1 | 192 | 129,408 | 0.4417 ± 0.0494 | 0.4417 ± 0.0494 | 0.405 | 0.4475 ± 0.0427 | 0.416 |
| DLinear | ETTh1 | 336 | 226,464 | 0.4422 ± 0.0025 | 0.4422 ± 0.0025 | 0.439 | 0.4438 ± 0.0033 | 0.443 |
| DLinear | ETTh1 | 720 | 485,280 | 0.5311 ± 0.0503 | 0.5301 ± 0.0502 | 0.472 | 0.5294 ± 0.0353 | 0.490 |
| DLinear | ILI | 24 | 5,040 | 2.2702 ± 0.0523 | 2.0054 ± 0.0347 | 2.215 | 1.0020 ± 0.0285 | 1.081 |
| DLinear | ILI | 36 | 7,560 | 2.2993 ± 0.0426 | 2.1312 ± 0.0461 | 1.963 | 1.0212 ± 0.0174 | 0.963 |
| DLinear | ILI | 48 | 10,080 | 2.3077 ± 0.0481 | 2.1368 ± 0.0103 | 2.130 | 1.0321 ± 0.0105 | 1.024 |
| DLinear | ILI | 60 | 12,600 | 2.4677 ± 0.0720 | 2.3877 ± 0.0759 | 2.368 | 1.1037 ± 0.0196 | 1.096 |

What the table shows:

* ETTh1, PatchTST/42: the drop-last means lie within 0.0017 of the
  published MSE and within 0.0013 of the published MAE at all four
  horizons, with a seed spread of at most 0.0054. The parameter counts,
  81,728 to 501,680, are the budgets later designs must respect at each
  horizon. At T = 336 and 720 the best validation epoch is between 5 and
  17 of 100, so most of those runs is spent past the minimum. A run takes
  6 to 9 minutes on the Apple GPU with the CPU idle.
* ILI, PatchTST/42: the drop-last means are 1.487, 1.409, 1.709 and 1.624
  against 1.522, 1.430, 1.673 and 1.529, and every published value lies
  within one standard deviation. The two settings corrected during the
  hardening rounds, the constant learning rate and the optimiser state
  left by the official scheduler, moved every horizon closer (drop-last
  means before: 1.563, 1.494, 1.751, 1.675). The windows the official
  loader drops are the most recent ones, with MSE 5.5 to 9.0 against 1.4
  to 1.7 for the rest, which is why the exact column is so much higher.
* ETTh1, DLinear: close at T = 96 and 336, but 9 and 12 percent above the
  published MSE at T = 192 and 720. In both settings one seed keeps its
  first-epoch weights (T = 192 seed 2023 at 0.498, T = 720 seed 2022 at
  0.579), while seed 2021 is within 3.4 and 1.2 percent. Early stopping
  with patience 3 on a noisy validation curve decides most of the spread.
  Ten seeds (`results/seed_study/summary_10_seeds.md`) reduce the gaps to
  5.6 and 8.5 percent but do not close them; the rest is unexplained.
* ILI, DLinear: the drop-last means are below the published value at
  T = 24, above it at T = 36, and within 0.02 at T = 48 and T = 60.
* PatchTST/42 against DLinear, paired over test windows: PatchTST has the
  lower mean error in all eight settings. The 95% block-bootstrap interval
  excludes zero at ETTh1 T = 192 and 720 and ILI T = 48 and 60; at ETTh1
  T = 96 and 336 the two models cannot be told apart.

Every record was scored again from its checkpoint by an evaluator that
shares no code with the training loop: 48 of 48 agree within 1.8e-8. See
`docs/validation_log.md` for that check and the test rounds.

## Next

Components 8 and 9: random search and regularized evolution over the
design space, reusing `run_experiment` with configs drawn from
`leaguetst.search`, held to the parameter budgets above and scored on the
same exact and drop-last protocols.
