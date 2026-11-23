"""The one test evaluation, run by the project owner from a committed prereg.

Needs LEAGUETST_VAULT pointing at the decrypted folder with the full CSVs.
Before the first test row is read it checks, and refuses on any problem:
the prereg file is committed unchanged and passes scripts/prereg.py's
checks; gate G3 finds nothing (lock, append-only files, test log); the test
split was not opened for this id before; and every checkpoint, re-scored on
the validation windows with the locked NumPy model, reproduces the
validation MSE registered for it within 1e-6 (which catches a wrong config).

Each run's per-window test errors and metrics are saved as soon as they are
scored, under results/final/test_<id>/. Statistics are computed from those
files, so ``--analyze-only`` can recompute them without opening the test
split again, and ``--resume`` scores only runs that were not saved.

The primary endpoint is the mean over horizons of exact test MSE. A
seed-level permutation test gives each comparison's p value (Holm over
comparisons); a two-level bootstrap (seeds, then window blocks of length
L + T) gives a supporting interval. Rank checks report the Spearman
correlation between registered validation MSE and test MSE.

Example:
    LEAGUETST_VAULT=~/leaguetst_vault python scripts/eval_test.py --prereg prereg/final_v1.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import prereg as prereg_checks  # noqa: E402
from leaguetst.data.test_split import ACCESS_LOG, load_test_split  # noqa: E402
from leaguetst.eval.integrity import g3_problems  # noqa: E402
from leaguetst.eval.metrics import exact_and_drop_last  # noqa: E402
from leaguetst.eval.numpy_model import numpy_patchtst_forward  # noqa: E402
from leaguetst.eval.rescore import load_numpy_state, validation_windows  # noqa: E402
from leaguetst.eval.stats import endpoint_bootstrap, holm, permutation_test  # noqa: E402

INPUT_LEN = {"ETTh1": 336, "ILI": 104}
BATCH_SIZE = {"ETTh1": 128, "ILI": 16}  # the PatchTST/42 recipes, for the drop-last column only
VAL_TOLERANCE = 1e-6


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def committed_unchanged(path: Path, root: Path) -> bool:
    rel = str(path.resolve().relative_to(root))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=root, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=root).returncode == 0
    return tracked and clean


def already_opened(prereg_id: str) -> bool:
    if not ACCESS_LOG.exists():
        return False
    return any(json.loads(line).get("prereg") == prereg_id for line in ACCESS_LOG.read_text().splitlines() if line.strip())


def numpy_forecast(state: dict, config: dict, inputs: np.ndarray, horizon: int, chunk: int = 256) -> np.ndarray:
    return np.concatenate([numpy_patchtst_forward(state, config, inputs[i:i + chunk], horizon) for i in range(0, len(inputs), chunk)])


def preflight(prereg: dict, root: Path, trainval: Path, check_opened: bool = True) -> list[str]:
    """Every problem that must stop the evaluation before a test row is read."""
    problems = prereg_checks.validate(prereg, root=root, access_log=ACCESS_LOG if check_opened else root / "none.log")
    g3, _ = g3_problems(root)
    problems += [f"G3: {p}" for p in g3]
    for arm in prereg["arms"]:
        dataset = arm["dataset"]
        for run in arm["runs"]:
            horizon = int(run["horizon"])
            try:
                state = load_numpy_state(root / run["checkpoint"], run["sha256"])
                inputs, targets = validation_windows(trainval, dataset, INPUT_LEN[dataset], horizon)
                preds = numpy_forecast(state, arm["model_config"], inputs, horizon)
                mse = float(((preds - targets) ** 2).mean())
            except Exception as error:  # any failure here must stop the evaluation
                problems.append(f"{arm['name']} T={horizon} seed {run['seed']}: cannot re-score validation ({error!r})")
                continue
            if abs(mse - float(run["val_mse"])) > VAL_TOLERANCE:
                problems.append(f"{arm['name']} T={horizon} seed {run['seed']}: validation MSE {mse:.8f} does not "
                                f"reproduce the registered {float(run['val_mse']):.8f}")
    return problems


def run_key(arm: str, horizon: int, seed: int) -> str:
    return f"{arm}_h{horizon}_s{seed}"


def score_runs(prereg: dict, vault: Path, trainval: Path, root: Path, save_dir: Path) -> None:
    """Score every run not saved yet; save its per-window errors and metrics at once."""
    for arm in prereg["arms"]:
        dataset = arm["dataset"]
        for run in arm["runs"]:
            horizon, seed = int(run["horizon"]), int(run["seed"])
            key = run_key(arm["name"], horizon, seed)
            if (save_dir / f"{key}.json").exists():
                continue
            test, _ = load_test_split(vault, trainval, dataset, INPUT_LEN[dataset], horizon, prereg["id"])
            values = test.values.double().numpy()
            starts = np.arange(len(test))
            inputs = values[starts[:, None] + np.arange(INPUT_LEN[dataset])[None, :]]
            targets = values[starts[:, None] + INPUT_LEN[dataset] + np.arange(horizon)[None, :]]
            state = load_numpy_state(root / run["checkpoint"], run["sha256"])
            preds = numpy_forecast(state, arm["model_config"], inputs, horizon)
            metrics = exact_and_drop_last(preds, targets, int(run.get("batch_size") or BATCH_SIZE[dataset]))
            per_window = ((preds - targets) ** 2).reshape(len(preds), -1).mean(axis=1)
            save_dir.mkdir(parents=True, exist_ok=True)
            np.save(save_dir / f"{key}.npy", per_window)
            write_json(save_dir / f"{key}.json", {"arm": arm["name"], "dataset": dataset, "horizon": horizon, "seed": seed,
                                                  "val_mse": float(run["val_mse"]), **metrics})
            print(f"{arm['name']:28s} T={horizon:3d} seed {seed}: test MSE {metrics['mse']:.4f} "
                  f"(drop-last {metrics['mse_drop_last']:.4f})", flush=True)


def analyze(prereg: dict, save_dir: Path) -> dict:
    """Comparisons and rank checks from the saved per-run files only."""
    per_window: dict[str, dict[int, dict[int, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
    runs_out = []
    for arm in prereg["arms"]:
        for run in arm["runs"]:
            horizon, seed = int(run["horizon"]), int(run["seed"])
            key = run_key(arm["name"], horizon, seed)
            runs_out.append(json.loads((save_dir / f"{key}.json").read_text()))
            per_window[arm["name"]][horizon][seed] = np.load(save_dir / f"{key}.npy")
    boot = prereg.get("bootstrap", {})
    datasets = {arm["name"]: arm["dataset"] for arm in prereg["arms"]}
    comparisons = {}
    for comparison in prereg.get("comparisons", []):
        a, b = comparison["a"], comparison["b"]
        horizons = sorted(per_window[a])
        if sorted(per_window[b]) != horizons:
            raise ValueError(f"{comparison['name']}: arms cover different horizons")
        seeds_a = sorted(set.intersection(*(set(per_window[a][h]) for h in horizons)))
        seeds_b = sorted(set.intersection(*(set(per_window[b][h]) for h in horizons)))
        registered = {name: {int(r["seed"]) for arm in prereg["arms"] if arm["name"] == name for r in arm["runs"]} for name in (a, b)}
        if set(seeds_a) != registered[a] or set(seeds_b) != registered[b]:
            raise ValueError(f"{comparison['name']}: scored seeds differ from the prereg")
        endpoint_a = [float(np.mean([per_window[a][h][s].mean() for h in horizons])) for s in seeds_a]
        endpoint_b = [float(np.mean([per_window[b][h][s].mean() for h in horizons])) for s in seeds_b]
        primary = permutation_test(np.array(endpoint_a), np.array(endpoint_b), seed=int(boot.get("seed", 0)))
        input_len = INPUT_LEN[datasets[a]]
        support = endpoint_bootstrap(
            {h: np.stack([per_window[a][h][s] for s in seeds_a]) for h in horizons},
            {h: np.stack([per_window[b][h][s] for s in seeds_b]) for h in horizons},
            {h: input_len + h for h in horizons},
            n_resamples=int(boot.get("resamples", 4000)),
            seed=int(boot.get("seed", 0)),
        )
        per_horizon = {h: {"a": float(np.mean([per_window[a][h][s].mean() for s in seeds_a])),
                           "b": float(np.mean([per_window[b][h][s].mean() for s in seeds_b]))} for h in horizons}
        comparisons[comparison["name"]] = {"a": a, "b": b, "seeds_a": seeds_a, "seeds_b": seeds_b,
                                           "endpoint_a": endpoint_a, "endpoint_b": endpoint_b,
                                           "endpoint_diff": primary["diff"], "endpoint_p": primary["p"],
                                           "bootstrap_supporting": support, "per_horizon": per_horizon}
    rejected = holm({k: v["endpoint_p"] for k, v in comparisons.items()}, float(prereg.get("alpha", 0.05))) if comparisons else {}
    for name in comparisons:
        comparisons[name]["significant_after_holm"] = rejected[name]

    from scipy.stats import spearmanr

    rank_checks = {}
    for group in prereg.get("rank_checks", []):
        pairs = [(r["val_mse"], r["mse"]) for r in runs_out if r["arm"] in group["arms"]]
        if len(pairs) >= 3:
            rho, p = spearmanr([v for v, _ in pairs], [t for _, t in pairs])
            rank_checks[group["name"]] = {"n": len(pairs), "spearman": float(rho), "p": float(p)}
    return {"prereg": prereg["id"], "runs": runs_out, "comparisons": comparisons, "rank_checks": rank_checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prereg", required=True)
    parser.add_argument("--trainval", default=str(ROOT / "data" / "trainval"))
    parser.add_argument("--resume", action="store_true", help="score only the runs not saved yet after an interruption")
    parser.add_argument("--analyze-only", action="store_true", help="recompute statistics from the saved runs")
    args = parser.parse_args()
    path = Path(args.prereg).resolve()
    prereg = json.loads(path.read_text(encoding="utf-8"))
    save_dir = ROOT / "results" / "final" / f"test_{prereg['id']}"
    if not args.analyze_only:
        vault = os.environ.get("LEAGUETST_VAULT")
        if not vault:
            print("set LEAGUETST_VAULT to the decrypted vault folder")
            return 1
        if not committed_unchanged(path, ROOT):
            print(f"{path} must be committed and unchanged before the test evaluation")
            return 1
        opened = already_opened(prereg["id"])
        if opened and not args.resume:
            print(f"the test split was already opened for {prereg['id']}; use --resume only to finish an interrupted run")
            return 1
        problems = preflight(prereg, ROOT, Path(args.trainval), check_opened=not args.resume)
        problems = [p for p in problems if not (args.resume and "test_access.log has uncommitted changes" in p)]
        if problems:
            print("evaluation refused before any test access:\n  " + "\n  ".join(problems))
            return 1
        score_runs(prereg, Path(vault).expanduser(), Path(args.trainval), ROOT, save_dir)
    result = analyze(prereg, save_dir)
    out = ROOT / "results" / "final" / f"test_{prereg['id']}.json"
    write_json(out, result)
    for name, comparison in result["comparisons"].items():
        print(f"{name}: endpoint difference {comparison['endpoint_diff']:+.4f}, permutation p {comparison['endpoint_p']:.4g}, "
              f"significant after Holm: {comparison['significant_after_holm']}")
    for name, check in result["rank_checks"].items():
        print(f"{name}: validation-to-test Spearman {check['spearman']:.3f} over {check['n']} runs")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
