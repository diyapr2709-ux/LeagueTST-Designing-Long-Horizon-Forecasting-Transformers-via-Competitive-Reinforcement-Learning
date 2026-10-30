# Benchmark data

Run `python scripts/download_data.py` from the repository root. It writes:

* `ETT-small/ETTh1.csv`: 17,420 hourly rows, 7 variables, from the
  ETDataset repository (github.com/zhouhaoyi/ETDataset).
* `illness/national_illness.csv`: 966 weekly rows, 7 variables, the
  Autoformer copy of the CDC influenza-like-illness series, mirrored on the
  Hugging Face hub under `thuml/Time-Series-Library`.

If a mirror is down, download the Autoformer dataset folder linked from the
official PatchTST README and copy the two files to the paths above. Both
files start with a `date` column and end with the `OT` target column.

The CSV files are ignored by git; only this note is committed.
