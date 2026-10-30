# Progress ledger

The proposal plans twelve weeks of work. This ledger assigns each component a
weight so that progress can be stated as one number. Weights sum to 100. The
first stage, worth 18 points, is complete.

## Components

| # | Component | Weight | State |
|---|---|---|---|
| 1 | Repository scaffold, environment, test harness | 2 | done |
| 2 | Data pipeline: ETTh1 and ILI loading, splits, scaling, windows, loaders | 5 | done |
| 3 | Configurable PatchTST with the nine design fields, RevIN, attention capture | 5 | done |
| 4 | DLinear baseline | 2 | done |
| 5 | Trainer: Adam, official schedules, early stopping, exact full-split metrics, run records | 3 | done |
| 6 | Preliminary reproduction at the shortest horizon, three seeds | 1 | done |
| 7 | Full reproduction of PatchTST/42 and DLinear at all horizons on both benchmarks | 6 | open |
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

Done: 18 of 100.

## What the first stage covers

* `leaguetst/data`: both benchmarks with the official split borders,
  training-set standardisation, and sliding windows. Validation and test
  loaders keep every window.
* `leaguetst/models`: `ForecasterConfig` holds the nine design choices and
  validates them. `PatchTST` reads the config, supports channel-independent
  and channel-mixing input, and can keep attention maps for the later
  analysis. `DLinear` matches the authors' model with a shared linear map.
* `leaguetst/training`: Adam with the official `type1` and `type3`
  schedules, early stopping on validation MSE, best-checkpoint restore, and
  exact MSE and MAE over the whole split next to the official-style values
  that drop the last partial batch.
* `leaguetst/experiment.py`: one run from spec to JSON record, with the
  best weights saved under `checkpoints/` for the later attention analysis.
* `leaguetst/search/space.py`: the 30,720-point grid, enumeration and
  uniform sampling. The search methods themselves are not started.
* `scripts/`: data download, single run, baseline sweep, summary table.
* `tests/`: 35 tests on synthetic copies of both benchmark files plus shape
  checks on the real files when present.

## Preliminary results

See `results/preliminary/summary.md`. The PatchTST/42 parameter count at
ETTh1, T = 96 is 81,728, which is the hand count of the reference
architecture and the budget later designs must respect at that horizon.

## Next

Component 7 is next: the full baseline sweep at all four ETTh1 horizons,
which needs about 13 minutes per run on an Apple GPU. After that, random
search and regularized evolution (components 8 and 9) reuse `run_experiment`
with configs drawn from `leaguetst.search`.
