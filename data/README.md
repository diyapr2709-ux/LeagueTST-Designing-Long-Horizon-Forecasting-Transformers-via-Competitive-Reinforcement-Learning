# Benchmark data

Run `python scripts/download_data.py` from the repository root. It writes:

* `ETT-small/ETTh1.csv`: 17,420 hourly rows, 7 variables, from the
  ETDataset repository (github.com/zhouhaoyi/ETDataset).
* `illness/national_illness.csv`: 966 weekly rows, 7 variables, the
  Autoformer copy of the CDC influenza-like-illness series, mirrored on the
  Hugging Face hub under `thuml/Time-Series-Library`.

The script pins both files to these SHA-256 digests, the copies behind
every stored result, and checks files already present against them:

* `ETTh1.csv`: `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`
* `national_illness.csv`: `93601f64d2566dc796ca4305adad8b8560c2db1a1ff04543c3bd813a7263570a`

If a mirror is down, download the Autoformer dataset folder linked from the
official PatchTST README and copy the two files to the paths above, then
rerun the script to check their digests. Both files start with a `date`
column and end with the `OT` target column; the loader also checks the
exact columns, the time order and that every value is finite.

The CSV files are ignored by git; only this note is committed.
