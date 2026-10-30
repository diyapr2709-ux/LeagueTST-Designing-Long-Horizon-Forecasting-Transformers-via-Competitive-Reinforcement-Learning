# LeagueTST

Designing long-horizon forecasting transformers via competitive reinforcement learning.

Given the previous `L` steps of a multivariate series, predict the next `T` steps.
The forecaster is a PatchTST-style transformer whose nine design choices
(patch length, stride, depth, heads, width, feed-forward ratio, dropout,
instance normalisation, channel handling) are configuration fields rather
than hand-tuned constants. Later stages of the project train a language-model
architect with group-relative rewards and a league of past checkpoints to
choose those fields, screen candidates with a learned referee after one
epoch, and compare the result with random search and regularized evolution
at equal compute.

This repository is a solo project. The proposal that defines the research
questions, benchmarks and verification protocol is summarised in
[docs/progress.md](docs/progress.md), which also tracks what is built.

## Status

The first stage is complete: data pipeline, configurable forecaster, DLinear
baseline, training loop, design space, tests, and a preliminary reproduction
of the published baselines at the shortest horizon of each benchmark. See
[results/preliminary/summary.md](results/preliminary/summary.md) for the
numbers and [docs/progress.md](docs/progress.md) for the ledger.

## Layout

```
configs/      YAML files for the baselines and the search space
data/         benchmark CSVs (downloaded, not committed)
docs/         progress ledger and validation log
leaguetst/    the package
  data/       loading, splitting, scaling, windowing
  models/     ForecasterConfig, PatchTST, DLinear, RevIN
  training/   Trainer, schedules, exact metrics
  search/     the 30,720-point design space
  utils/      seeding, device choice, YAML and JSON helpers
  experiment.py  one run from spec to saved JSON record
results/      run records and summary tables
checkpoints/  best weights of every run (not committed)
scripts/      download_data, train, reproduce_baselines, summarize_results
tests/        pytest suite on synthetic copies of both benchmarks
```

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_data.py
.venv/bin/python -m pytest
```

Python 3.10 or newer. Training uses CUDA, then Apple MPS, then CPU, whichever
is available first; pass `--device cpu` to force the CPU.

## Usage

Train one model and write its record to `results/preliminary/`:

```
.venv/bin/python scripts/train.py --config configs/patchtst42_etth1.yaml --horizon 96 --seed 2021
```

Reproduce the baselines over horizons and seeds, then build the table:

```
.venv/bin/python scripts/reproduce_baselines.py \
    --configs configs/patchtst42_etth1.yaml configs/dlinear_etth1.yaml \
    --horizons 96 --seeds 2021 2022 2023
.venv/bin/python scripts/summarize_results.py --results-dir results/preliminary
```

Build a forecaster from a design-space point in code:

```python
import random
from leaguetst.search import sample_config
from leaguetst.models import PatchTST

config = sample_config(random.Random(0))
model = PatchTST(config, n_vars=7, input_len=336, horizon=96)
```

## Protocol

* ETTh1: 336 input steps, horizons 96, 192, 336, 720; split 12/4/4 months.
* ILI: 104 input steps, horizons 24, 36, 48, 60; split 70/10/20 by rows.
* Variables are standardised with training-set statistics only.
* Validation and test errors are exact means over every window of the split;
  no final batch is dropped. Each record also carries the official-style
  values that drop the last partial test batch, so the gap between the two
  protocols is visible next to the published numbers.
* Only validation error guides any search; the test split is read once per
  final model.
* Every number is reported as mean and standard deviation over three seeds.

## References

* Nie et al., A Time Series is Worth 64 Words: Long-term Forecasting with
  Transformers, ICLR 2023 (PatchTST).
* Zeng et al., Are Transformers Effective for Time Series Forecasting?,
  AAAI 2023 (DLinear).
* Kim et al., Reversible Instance Normalization for Accurate Time-Series
  Forecasting against Distribution Shift, ICLR 2022 (RevIN).
