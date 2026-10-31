#!/bin/bash
# System test: fresh source copy, test suite, CLI reproduction of stored records,
# error paths, wheel build. The work directory is wiped first.
# Usage: scripts/validation/system_test.sh "$PWD" /tmp/leaguetst_system
R="$1"; W="$2"; PY="$R/.venv/bin/python"
rm -rf "$W"; mkdir -p "$W"
pass=0; fail=0
check() { if [ "$1" = 0 ]; then echo "PASS  $2"; pass=$((pass+1)); else echo "FAIL  $2"; fail=$((fail+1)); fi; }

rsync -a --exclude /.venv --exclude /.git --exclude /data --exclude /results --exclude /checkpoints --exclude .pytest_cache --exclude __pycache__ --exclude /build --exclude "*.egg-info" "$R/" "$W/src/"
mkdir -p "$W/src/data" && cp "$R/data/README.md" "$W/src/data/"
cd "$W/src"


"$PY" scripts/download_data.py --data-dir "$W/src/data" > "$W/dl.txt" 2>&1
cmp -s data/ETT-small/ETTh1.csv "$R/data/ETT-small/ETTh1.csv" && cmp -s data/illness/national_illness.csv "$R/data/illness/national_illness.csv"
check $? "S2 download script fetches both benchmarks, byte-identical to the files used for the records"

OMP_NUM_THREADS=2 "$PY" -m pytest -q -p no:cacheprovider -rs > "$W/pytest.txt" 2>&1; rc=$?; grep -q "SKIPPED" "$W/pytest.txt" && rc=1
check $rc "S1 full test suite on a fresh copy, real-data tests included, none skipped ($(tail -1 "$W/pytest.txt"))"

for run in "dlinear_ilI:configs/dlinear_ili.yaml:24:2021:dlinear_ILI_h24_s2021" "dlinear_etth1:configs/dlinear_etth1.yaml:96:2021:dlinear_ETTh1_h96_s2021" "dlinear_etth1_720:configs/dlinear_etth1.yaml:720:2023:dlinear_ETTh1_h720_s2023"; do
  IFS=: read -r _ cfg h s name <<< "$run"
  OMP_NUM_THREADS=2 "$PY" scripts/train.py --config "$cfg" --horizon "$h" --seed "$s" --device cpu --out-dir "$W/out" --checkpoint-dir "$W/ckpt" > "$W/train_$name.txt" 2>&1
  stored="$R/results/preliminary/$name.json"; [ -f "$stored" ] || stored="$R/results/baselines/$name.json"
  "$PY" -I -c "
import json,sys
a=json.load(open(sys.argv[1]))['test']; b=json.load(open(sys.argv[2]))['test']
d=max(abs(a[k]-b[k]) for k in ('mse','mae','mse_drop_last','mae_drop_last'))
print(f'      max |diff| vs stored record {d:.2e}')
sys.exit(0 if d < 1e-6 else 1)" "$W/out/$name.json" "$stored"
  check $? "S3 CLI retrain of $name on a fresh copy reproduces the stored record"
done

OMP_NUM_THREADS=2 "$PY" scripts/reproduce_baselines.py --configs configs/dlinear_ili.yaml --horizons 24 --seeds 2021 --device cpu --out-dir "$W/out" --checkpoint-dir "" --skip-existing > "$W/skip.txt" 2>&1
grep -q "skip" "$W/skip.txt"; check $? "S4 reproduce_baselines skips a finished run"

"$PY" scripts/summarize_results.py --results-dir "$W/out" --out "$W/summary.md" > /dev/null 2>&1 && [ "$(grep -c '^| dlinear' "$W/summary.md")" = 3 ]
check $? "S5 summary over the CLI outputs has one row per setting"

"$PY" scripts/train.py --config configs/missing.yaml --horizon 24 > "$W/e1.txt" 2>&1; [ $? -ne 0 ] && grep -q "missing.yaml" "$W/e1.txt"
check $? "S6 missing config fails with the file name in the error"
"$PY" scripts/train.py --config configs/dlinear_ili.yaml --horizon 24 --data-dir "$W/nowhere" --device cpu > "$W/e2.txt" 2>&1; [ $? -ne 0 ] && grep -q "download_data.py" "$W/e2.txt"
check $? "S7 missing data fails with a pointer to the download script"
"$PY" scripts/train.py --config configs/dlinear_ili.yaml --horizon 5000 --device cpu > "$W/e3.txt" 2>&1; [ $? -ne 0 ] && grep -q "cannot hold one window" "$W/e3.txt"
check $? "S8 impossible horizon fails with a clear message"

"$PY" -m pip wheel . --no-deps --no-build-isolation -w "$W/wheel" -q > "$W/wheel.txt" 2>&1 && mkdir -p "$W/site" && "$PY" -m pip install --no-deps --target "$W/site" "$W"/wheel/*.whl -q >> "$W/wheel.txt" 2>&1
(cd / && "$PY" -c "import sys; sys.path.insert(0, '$W/site'); import leaguetst, leaguetst.data, leaguetst.models, leaguetst.training, leaguetst.search, leaguetst.utils, leaguetst.experiment; assert leaguetst.__file__.startswith('$W/site'); print('      imported from', leaguetst.__file__)")
check $? "S9 wheel builds and every subpackage imports from the installed copy"

"$PY" scripts/download_data.py --data-dir "$W/src/data" > "$W/dl2.txt" 2>&1 && [ "$(grep -c 'digest matches' "$W/dl2.txt")" = 2 ]
check $? "S10 present files are re-verified against the pinned digests"

OMP_NUM_THREADS=2 "$PY" scripts/revalidate.py --results-dir "$W/out" --checkpoint-dir "$W/ckpt" --out-dir "$W/reval" --resamples 200 > "$W/reval.txt" 2>&1
check $? "S11 revalidation of the CLI outputs ($(tail -1 "$W/reval.txt"))"

"$PY" -I -c "
import json, glob, hashlib, sys
recs = [json.load(open(p)) for p in glob.glob(sys.argv[1] + '/*.json')]
ok = all(r['checkpoint_sha256'] == hashlib.sha256(open(r['checkpoint'], 'rb').read()).hexdigest() and r['fit']['diverged'] is False for r in recs)
sys.exit(0 if ok and recs else 1)" "$W/out"
check $? "S12 new records carry the checkpoint digest and a divergence flag"

sed 's/lr_schedule: type1/lr_schedule: constant/' configs/dlinear_ili.yaml > "$W/changed.yaml"
"$PY" scripts/reproduce_baselines.py --configs "$W/changed.yaml" --horizons 24 --seeds 2021 --device cpu --out-dir "$W/out" --checkpoint-dir "" --skip-existing > "$W/stale.txt" 2>&1
[ $? -ne 0 ] && grep -q "different training" "$W/stale.txt"
check $? "S13 skip-existing refuses a stored run made with another config"

mkdir -p "$W/dup" && cp "$W/out/dlinear_ILI_h24_s2021.json" "$W/dup/"
"$PY" scripts/summarize_results.py --results-dir "$W/out" "$W/dup" --out "$W/dup.md" > "$W/dup.txt" 2>&1
[ $? -eq 1 ] && grep -q "appears twice" "$W/dup.txt"
check $? "S14 summary exits 1 and names both files when a run is pooled twice"

OMP_NUM_THREADS=2 "$PY" scripts/train.py --config configs/patchtst42_ili.yaml --horizon 24 --seed 7 --epochs 2 --device cpu --out-dir "$W/short" --checkpoint-dir "$W/short_ckpt" > "$W/short.txt" 2>&1 && \
OMP_NUM_THREADS=2 "$PY" scripts/revalidate.py --results-dir "$W/short" --out-dir "$W/short_reval" --resamples 100 > "$W/short_reval.txt" 2>&1 && \
grep -q '"lr_schedule": "constant"' "$W/short/patchtst_ILI_h24_s7.json"
check $? "S15 short PatchTST run with the corrected ILI config trains, saves and revalidates"

echo "SYSTEM TEST: $pass passed, $fail failed"
