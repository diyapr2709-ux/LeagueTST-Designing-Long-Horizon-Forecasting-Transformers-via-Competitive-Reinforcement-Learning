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
