# Validation log

Each pass runs the test suite and an end-to-end run, then fixes whatever
surfaced. Three passes were run for the first stage.

## Pass 1: test suite on synthetic data

35 tests, 3 failures.

* Channel-mixing input padded a 4-D tensor with replicate mode, which
  PyTorch does not support. Fixed by patching the 3-D (batch, series, time)
  tensor first and stacking variables afterwards; both channel modes now
  share one `patchify` step.
* `test_error_accumulator_matches_global_mean` skipped one sample and
  counted another twice, so the expected mean did not match. Fixed the
  slice bounds in the test.
* The training loop converted a loss that still carried a gradient to a
  float, which PyTorch warns about. Switched to `loss.item()`.

Rerun: 35 passed.

## Pass 2: end-to-end run on the real ETTh1 file

Two epochs of PatchTST/42 at T = 96 on CPU and on MPS.

* Parameter count 81,728 on both devices, equal to the hand count.
* Validation and test errors agree between devices to the fourth decimal
  after two epochs, so the MPS path is numerically sound for this model.
* Per-epoch time: 19 s on CPU, 7.6 s on MPS after warm-up.
* No code change was needed.

## Pass 3: full preliminary runs and review

The three ETTh1 seeds of PatchTST/42 at T = 96 ran for 100 epochs on MPS,
the ILI and DLinear runs on CPU. A review of the first records led to two
additions and exposed one bug.

* The official PatchTST test loader drops its last partial batch, which on
  ILI removes the ten most recent windows. Records now carry both the exact
  mean over every window (the project protocol) and the drop-last mean, so
  the two can be compared with the published numbers.
* The best weights of every run are now saved under `checkpoints/` for the
  attention analysis later in the project.
* The first version of the per-sample accumulator cast errors to float64 on
  the device, which MPS does not support, so the GPU rerun failed at its
  first evaluation. Fixed by summing on the device and casting on the CPU.
  A test now evaluates on MPS when the device exists and checks that the
  values agree with the CPU to four digits.

Rerun: 36 passed. The ETTh1 and ILI runs were then repeated with the final
code; the first-pass ETTh1 records serve as a reproducibility check and are
listed in the results section below.

## Results of pass 3

The reruns with the final code reproduce the first-pass records. Over the
ten runs present in both sets, the largest difference in test MSE or MAE
is 2.7e-8. The first-pass records are kept in `results/first_pass/`, the
final ones in `results/preliminary/`. The PatchTST records of this pass
were later superseded by the corrected reruns of passes 4 to 6.

| run | first-pass MSE | rerun MSE | first-pass MAE | rerun MAE | epochs |
|---|---|---|---|---|---|
| dlinear_ETTh1_h96_s2021 | 0.3716 | 0.3716 | 0.3934 | 0.3934 | 10 |
| dlinear_ETTh1_h96_s2022 | 0.3755 | 0.3755 | 0.3986 | 0.3986 | 9 |
| dlinear_ETTh1_h96_s2023 | 0.3963 | 0.3963 | 0.4231 | 0.4231 | 7 |
| patchtst_ETTh1_h96_s2021 | 0.3776 | 0.3776 | 0.3993 | 0.3993 | 100 |
| patchtst_ETTh1_h96_s2022 | 0.3778 | 0.3778 | 0.3997 | 0.3997 | 100 |
| patchtst_ETTh1_h96_s2023 | 0.3770 | 0.3770 | 0.3997 | 0.3997 | 100 |
| patchtst_ILI_h24_s2021 | 2.0568 | 2.0568 | 0.9144 | 0.9144 | 100 |
| patchtst_ILI_h24_s2022 | 2.0357 | 2.0357 | 0.9198 | 0.9198 | 100 |
| patchtst_ILI_h24_s2023 | 1.9319 | 1.9319 | 0.8679 | 0.8679 | 100 |
| patchtst_ILI_h36_s2021 | 1.8029 | 1.8029 | 0.8576 | 0.8576 | 100 |

The seed-2023 ETTh1 PatchTST run had to be repeated on 2026-10-30 because
the session that started it ended before it ran; it matched its first-pass
value as well. The three ETTh1 seeds give 0.3775 ± 0.0004 exact and
0.3737 ± 0.0004 drop-last against the published 0.375, at the reference
parameter count 81,728.

The drop-last question from pass 3 is answered. At ILI T = 24 the exact
PatchTST/42 mean is 2.008 ± 0.067 and the drop-last mean 1.563 ± 0.109
against the published 1.522. The ten windows the official loader drops
are the most recent ones and have a mean squared error of about 8.6, so
the evaluation protocol accounts for most of the gap. At every ILI horizon
the published PatchTST/42 value lies within one standard deviation of the
drop-last mean. Full table: `results/preliminary/summary.md`.

Final check on 2026-10-30: 36 tests pass on the checkout that produced
the records.

## Passes 4 to 6: revalidation, system and penetration rounds (2026-10-30)

Three rounds of three kinds of check, with fixes between rounds. Every
check can be rerun:

* Revalidation: `scripts/revalidate.py` reloads every checkpoint, scores
  it again on the CPU with a NumPy evaluator that shares no code with the
  training loop, and compares the result with the record.
  `scripts/validation/data_integrity.py` checks the benchmark files, the
  splits, leakage and parameter counts. The configs were also compared
  line by line with the official scripts and training loops.
* System test: `scripts/validation/system_test.sh` copies the source tree
  without data, results or environment, downloads the data, runs the test
  suite, retrains stored runs from the command line, exercises the error
  paths, and builds and imports a wheel.
* Penetration test: `scripts/validation/pentest.py` replays attacks and
  malformed inputs against the code. A probe is VULNERABLE when the attack
  works or bad input is accepted silently.

### Round 1 (pass 4): the code as it stood

Revalidation.

* All 43 records then present (31 current, 12 ILI PatchTST) agree with
  their checkpoints. The largest relative gap is 2e-8; checkpoints trained
  on the Apple GPU reproduce on the CPU within 4e-9.
* Both data files are byte-identical to a fresh download from the
  original sources.
* 34 of 34 data checks pass after one correction to the check itself: row
  counts, exact columns, time order at a constant step, finite values,
  split borders, window counts at every horizon, and parameter counts equal
  to the hand count at every horizon for both models. The first leakage
  check compared scaler statistics bit for bit and failed because pandas
  hands over a column-major array whose sums differ from a row-major copy
  by 1e-16. It was replaced by a stronger test: every row after the
  training block is multiplied by 1000 in a copy of the file, and the
  scaler and every training window come out identical.
* The config audit found one deviation. The official ILI script passes
  `--lradj 'constant'`; our ILI PatchTST config decayed the rate with
  `type3`. The twelve ILI PatchTST records were moved to
  `results/superseded/patchtst_ili_type3/` and rerun.
* The audit also settled which published protocol applies to DLinear.
  LTSF-Linear switched its test loader from dropping the last batch to
  keeping it in March 2023 (commit 6fe4c28), after the published numbers.
  The drop-last column is therefore the comparison for both models.

System test: 11 of 11 after one harness fix (the copy step excluded every
folder named `data`, which also removed the `leaguetst/data` package).
Three DLinear runs retrained from the command line reproduce their stored
records to 4e-10.

Penetration test: 18 probes, 15 vulnerable, 3 safe.

| probe | round 1 | fix |
|---|---|---|
| A1 run tag `x/../../../escaped` | wrote the record and checkpoint two levels above the output directory | tags must match `[A-Za-z0-9][A-Za-z0-9._-]{0,63}` |
| A2 YAML `!!python/object/apply:os.system` | safe: `yaml.safe_load` refuses it | top-level mapping now required as well |
| A3 pickle payload in a checkpoint | safe: `torch.load` defaults to weights only | `load_state_dict` helper also checks the digest and the tensor types |
| A4a `instance_norm: "false"` | accepted; a truthy string switched RevIN on | booleans must be booleans |
| A4b `patch_len: true`, `16.5`, `n_layers: 2.0` | accepted | integers must be integers, not booleans or floats |
| A4c `n_heads: 0`, negative sizes | ZeroDivisionError, or accepted | every size must lie in [1, limit] |
| A4d `d_model: 2**20` | accepted; the model would need about 8.8e12 parameters | caps far above the grid: d_model 1024, n_layers 12, n_heads 64, patch 512, ff_ratio 16 |
| A5 learning rate -1, NaN, inf | accepted | finite and positive required |
| A6 horizon 0 or -5, seed -1 or 2**40, input 0 | accepted by the spec | positive horizon and input, seed in [0, 2**32) |
| A7 learning rate 1e30 | NaN losses recorded as a normal run with the untrained weights' test error | training stops on a non-finite loss and the record carries `fit.diverged`; summaries refuse it |
| A8 data with a NaN, shuffled rows, no OT column, an extra column | all trained silently | the loader checks exact columns, constant time step and finite values |
| A9a one run in two pooled directories | counted as two seeds | duplicate runs are an error naming both files |
| A9b two configs under one setting | averaged into one row | every row's seeds must share config, training and parameter count |
| A9c record without fields | bare KeyError | error names the file and the missing fields |
| A10 `--skip-existing` after a config change | kept the stale record silently | the stored record's settings must match the spec, otherwise an error |
| A11 interrupt during a JSON write | truncated record left on disk | records and checkpoints are written to a temporary file and renamed |
| A12 forged file served as ETTh1 | accepted (only the first bytes were checked) | https only, size cap, SHA-256 pinned per file; present files are checked too |
| A13 horizon or input of 10**7 | safe: clear error | |

### Round 2 (pass 5): after the round 1 fixes

* Tests: 36 grew to 88 with a regression test per finding and tests of the
  new evaluator.
* Revalidation: 44 of 44 records agree, then 56 of 56 in the DLinear seed
  study.
* System test: 17 of 17. The full suite now runs after the download, so
  the real-file tests are included, none skipped. Retraining with the
  hardened code reproduces the stored DLinear records to 4e-10, so the
  fixes changed no number.
* Dependency audit: pip-audit 2.10.1 found no known vulnerability in the
  21 installed packages. The exact versions are pinned in
  `requirements-lock.txt`.
* Penetration test: 23 probes, 21 safe, 2 vulnerable, plus one found by
  review.
  * A16: a record holding a NaN metric without the divergence flag was
    printed as a result. Summaries now refuse non-finite metrics.
  * A17: six threads writing one record produced 17 torn reads, because
    threads of one process shared a temporary file name. Temporary names
    now carry a random suffix.
  * Review: NaN compares false with any tolerance, so `revalidate.py`
    would have passed a NaN record. Non-finite values now fail.
* A second config deviation surfaced while checking the ILI gap at
  T = 60. The official PatchTST trainer builds a OneCycleLR scheduler
  before training for every schedule. Building it sets Adam's first beta
  to 0.95, which nothing resets, and the first epoch's rate to
  `learning_rate / 25`. PyTorch 2.15 shows the same effect. Both PatchTST
  configs now set `adam_beta1: 0.95` and `first_epoch_lr_factor: 0.04`; a
  test builds the official scheduler and checks the two values match. The
  LTSF-Linear trainer builds no such scheduler, so DLinear keeps the
  defaults and its stored records still match. The 21 PatchTST records
  made before the fix were moved to
  `results/superseded/patchtst_adam_beta1_0.9/` and every PatchTST run was
  repeated.

### Round 3 (pass 6): the final code

* Tests: 90 passed.
* Penetration test: 24 of 24 probes safe, no crash. A5 now also covers the
  two new optimiser settings.
* System test: 17 of 17.
* Data checks: 34 of 34.
* Every archived record still agrees with its checkpoint: 12 of 12 for the
  ILI `type3` set and 21 of 21 for the Adam 0.9 set.

### Final records (2026-10-30, after round 3)

The PatchTST reruns with both corrections finished at 20:26: twelve ILI
runs on the CPU and twelve ETTh1 runs on the Apple GPU.

* Revalidation round 3 over `results/preliminary` and `results/baselines`:
  48 of 48 runs agree with their records, largest relative gap 1.8e-8.
  Report: `results/revalidation/report_round3.md`.
* Tests: 90 passed on the checkout that produced the records.
* The ETTh1 T = 96 reruns pick the same best epochs as the superseded
  runs (68, 75, 81), which looked like a stale record. It is not: the new
  records carry `adam_beta1` 0.95 and `first_epoch_lr_factor` 0.04, the
  log shows epoch 1 at 4e-6, and the test MSE differs by about 0.0015.
  Each seed fixes the batch order and dropout masks, and near the minimum
  the rate is about 1e-7, so the epoch-to-epoch changes in validation MSE
  of old and new runs correlate at 0.998 or more and the minimum falls on
  the same epoch.

Drop-last test MSE of PatchTST/42 before and after the corrections, mean
over three seeds:

| dataset | T | before | after | published |
|---|---|---|---|---|
| ETTh1 | 96 | 0.3737 | 0.3754 | 0.375 |
| ILI | 24 | 1.5631 | 1.4867 | 1.522 |
| ILI | 36 | 1.4939 | 1.4086 | 1.430 |
| ILI | 48 | 1.7513 | 1.7090 | 1.673 |
| ILI | 60 | 1.6753 | 1.6239 | 1.529 |

At ETTh1 T = 192, 336 and 720 the corrected drop-last means are 0.4129,
0.4293 and 0.4483 against 0.414, 0.431 and 0.449. The full table and its
reading are in `docs/progress.md`.
