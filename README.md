# LeagueTST

Configuring PatchTST-style long-horizon forecasters with an LLM architect
trained by reinforcement learning.

Given the previous `L` steps of a multivariate series, predict the next `T` steps.
The forecaster is a PatchTST-style transformer whose design choices
(patch length, stride, depth, heads, width, feed-forward ratio, dropout,
instance normalisation, channel handling) are configuration fields rather
than hand-tuned constants. Search runs only over the configs that fit the
PatchTST/42 parameter cap, which keeps every candidate channel-independent
and at width 16 or 32 (amendment A10). An LLM architect (Qwen2.5-1.5B with
LoRA) is trained with GRPO and a league of rivals to choose those fields,
with lookups in a precomputed table of validation results as its reward. It
is compared with random search, regularized evolution, TPE, ASHA, a
policy-gradient controller and the same LLM without RL. Every method gets
the same total number of training epochs; nothing else about cost is held
equal.

This repository is a solo project. The research questions, the build plan
and every amendment are in [docs/PLAN.md](docs/PLAN.md), the current state
in [docs/STATE.md](docs/STATE.md), and the working rules in
[CLAUDE.md](CLAUDE.md).

## Status

Stage 1 built the data pipeline, the forecaster and the training loop, and
reproduced PatchTST/42 and DLinear on the test split. The stage-1 baseline
code that produced the progress-report numbers lives at tag `stage1-baselines`.
Its records stay in `results/` and its history in
[docs/progress.md](docs/progress.md) and
[docs/validation_log.md](docs/validation_log.md).

The current pipeline, items E0 to E8 of docs/PLAN.md, trains only on
`data/trainval/` and selects on exact validation MSE. The test split is
evaluated once, by the owner, from a committed prereg file. Progress on each
item is in docs/PLAN.md and docs/STATE.md.

## Layout

```
configs/      PatchTST/42 recipes; every candidate trains with its dataset's recipe
data/         benchmark files (not committed); trainval/ holds the train and
              validation rows and MANIFEST.json
docs/         plan, state, issues, claims, verdicts, report draft, validation log
prereg/       the E0 lock and the pre-registration files for the test evaluation
leaguetst/    the package
  data/       loading, checking, splitting, scaling, windowing, the trainval
              copy, the GPU-resident loader, and the logged door to the test rows
  models/     ForecasterConfig, PatchTST, RevIN
  training/   Trainer, schedules, exact metrics, divergence flag
  search/     the design grid, the 2,368 configs that fit the PatchTST/42
              parameter cap at all eight settings (74 architectures, widths
              16 and 32, channel-independent only), and replay of every
              search method on a table
  architect/  the LLM architect: prompts, constrained decoding, episodes
  eval/       NumPy re-scorer, statistics, and the E0 lock checks
  utils/      seeding, type checks, atomic JSON and checkpoint writes
  table.py    one table entry: train a config on trainval rows, write its record
  ledger.py   the append-only ledger, results/ledger.jsonl
results/
  ledger.jsonl  every trained run; every reported number comes from it
  search/, fidelity/, e3b/, e6/, picks/, anova/   outputs of the pipeline
  preliminary/, baselines/, first_pass/, seed_study/, revalidation/
                stage-1 records, kept as history
scripts/      the pipeline (see Usage) and download_data
tests/        pytest suite on synthetic copies of both benchmarks
```

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/python -m pytest
```

`requirements-lock.txt` pins the exact versions behind the stored results;
`requirements.txt` gives the looser lower bounds. Python 3.10 or newer.

Three further requirement files serve specific jobs and are never installed
into `.venv`: `requirements-llm.txt` (the LLM architect runs, pinned by
amendment A25), `requirements-modal.txt` (the Modal client) and
`requirements-ablations.txt` (the AB-X external-HPO baselines, which need
numpy below 2 and so run in their own `.venv-abx`, as its header says).

The owner provisions the data once: `scripts/download_data.py` fetches both
benchmark files and checks each against a pinned SHA-256 digest, and
`scripts/make_trainval.py --src <folder with the full files>` writes the
train and validation rows to `data/trainval/`. The full files then move to
an encrypted vault outside the repository. Everything else reads
`data/trainval/` only.

## Usage

Each item of docs/PLAN.md runs through these scripts; every script's
docstring has a full example.

* Tables (E1, E3b, E6): `scripts/modal_app.py` trains configs on Modal GPUs,
  reading only the trainval volume. ETTh1 jobs run with
  `MODAL_PROFILE=diya-027`, ILI jobs with `MODAL_PROFILE=diyapr`.
  `scripts/collect_runs.py` downloads the records and appends them to the
  ledger.
* Fidelity and table checks (E2, E3b): `scripts/fidelity.py`,
  `scripts/e3b.py`, `scripts/table_stats.py`.
* Search (E3, E4): `scripts/replay.py` replays the CPU methods on a table;
  `scripts/modal_llm.py` runs the prompted and GRPO-trained LLM architect.
  `scripts/picks.py` scores every method's picks on fresh seeds.
* RQ3 (E8): `scripts/anova.py`.
* Checks and report: `scripts/gates.py` runs gates G1 to G6, and
  `scripts/report.py` regenerates docs/REPORT_DRAFT.md from the ledger.
* Test evaluation (owner only): `scripts/build_prereg_spec.py` and
  `scripts/prereg.py` register the final checkpoints, then
  `scripts/eval_test.py` scores them once.

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
* Selection uses exact, unshuffled validation MSE over every window; no
  batch is dropped in evaluation.
* Every candidate trains with its dataset's PatchTST/42 recipe (Adam first
  beta 0.95, epoch 1 at a twenty-fifth of the rate, the official schedule),
  capped at 30 epochs with patience 10. The recipe is never tuned per
  candidate.
* Budgets count training epochs, and every method gets the same total.
  Every distinct config a method looks up is charged its epochs once.
* Every record carries its git hash, its config, the SHA-256 digests of its
  checkpoint and per-window errors, and a divergence flag.
* The test split is read once, by the owner, for the models registered in a
  committed prereg file.

## References

* Nie et al., A Time Series is Worth 64 Words: Long-term Forecasting with
  Transformers, ICLR 2023 (PatchTST).
* Zeng et al., Are Transformers Effective for Time Series Forecasting?,
  AAAI 2023 (DLinear).
* Kim et al., Reversible Instance Normalization for Accurate Time-Series
  Forecasting against Distribution Shift, ICLR 2022 (RevIN).
