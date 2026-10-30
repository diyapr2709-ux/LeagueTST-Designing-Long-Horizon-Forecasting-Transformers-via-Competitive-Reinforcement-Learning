# Results

Each training run writes one JSON record named
`<model>_<dataset>_h<horizon>_s<seed>[_<tag>].json` with the configuration,
parameter count, per-epoch history, the exact validation and test MSE and
MAE over every window, the same metrics with the last partial batch dropped
(`mse_drop_last`, `mae_drop_last`, the official PatchTST protocol), and the
path of the saved weights under `checkpoints/`.
`scripts/summarize_results.py` aggregates the records of a folder into
`summary.md` and `summary.json` with mean and sample standard deviation
over seeds.

* `preliminary/`: the baseline reproduction at the shortest horizon of
  ETTh1 and at all four ILI horizons, three seeds each.
