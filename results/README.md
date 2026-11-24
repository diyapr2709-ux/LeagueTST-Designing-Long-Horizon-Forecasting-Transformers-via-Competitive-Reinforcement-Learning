# Results

`ledger.jsonl` is the append-only record of every search-era run (train and
validation only) and the source of every reported number (scripts/report.py).
`runs/` holds the raw records, checkpoints and per-window validation errors
downloaded from Modal; `search/`, `picks/`, `e3b/`, `e5/`, `e6/`, `fidelity/`
and `anova/` hold the analyses.

The folders below are from the October baseline reproduction (stage 1). Its
records also hold test MSE, scored before the test split was locked; the
code that wrote and summarized them is at git tag `stage1-baselines`.

* `preliminary/`: the baseline reproduction at the shortest horizon of
  ETTh1 and at all four ILI horizons, three seeds each.
