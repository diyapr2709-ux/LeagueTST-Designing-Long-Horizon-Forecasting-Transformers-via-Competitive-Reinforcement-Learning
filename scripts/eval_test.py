"""The one test evaluation, run by the project owner from a committed prereg.

Needs LEAGUETST_VAULT pointing at the decrypted folder with the full CSVs.
Refuses to run when the prereg file is not committed unchanged, when any
checkpoint digest differs, or when the test split was already opened for
this prereg id. Every test access is logged by leaguetst.data.test_split.

For each arm, horizon and seed it scores the exact test MSE and MAE over
every window (plus the drop-last values for comparison with the
literature), then tests each pre-registered comparison on the primary
endpoint with a paired two-level bootstrap and Holm's correction.

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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leaguetst.data.test_split import load_test_split  # noqa: E402
from leaguetst.eval.metrics import exact_and_drop_last  # noqa: E402
from leaguetst.eval.stats import holm, paired_two_level_bootstrap  # noqa: E402
from leaguetst.experiment import build_model  # noqa: E402
from leaguetst.search.feasible import INPUT_LEN  # noqa: E402
from leaguetst.utils import load_state_dict, save_json  # noqa: E402


def committed_unchanged(path: Path, root: Path) -> bool:
    rel = str(path.resolve().relative_to(root))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=root, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=root).returncode == 0
    return tracked and clean


def already_opened(prereg_id: str, log_path: Path) -> bool:
    if not log_path.exists():
        return False
    return any(json.loads(line).get("prereg") == prereg_id for line in log_path.read_text().splitlines() if line.strip())


@torch.no_grad()
def forecast(model: torch.nn.Module, dataset, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    for start in range(0, len(dataset), batch_size):
        items = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        inputs = torch.stack([x for x, _ in items])
        preds.append(model(inputs).double().numpy())
        targets.append(torch.stack([y for _, y in items]).double().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def evaluate(prereg: dict, vault: Path, trainval: Path, root: Path, log_path: Path) -> dict:
    per_window: dict[str, dict[int, dict[int, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
    runs_out = []
    torch.set_num_threads(4)
    for arm in prereg["arms"]:
        dataset = arm["dataset"]
        for run in arm["runs"]:
            horizon, seed = int(run["horizon"]), int(run["seed"])
            test, _ = load_test_split(vault, trainval, dataset, INPUT_LEN[dataset], horizon, prereg["id"], log_path)
            model = build_model("patchtst", arm["model_config"], test.n_vars, INPUT_LEN[dataset], horizon)
            model.load_state_dict(load_state_dict(root / run["checkpoint"], expected_sha256=run["sha256"]))
            preds, targets = forecast(model, test)
            metrics = exact_and_drop_last(preds, targets, int(run.get("batch_size", 128)))
            per_window[arm["name"]][horizon][seed] = ((preds - targets) ** 2).reshape(len(preds), -1).mean(axis=1)
            runs_out.append({"arm": arm["name"], "dataset": dataset, "horizon": horizon, "seed": seed, **metrics})
            print(f"{arm['name']:28s} T={horizon:3d} seed {seed}: test MSE {metrics['mse']:.4f} (drop-last {metrics['mse_drop_last']:.4f})", flush=True)

    boot = prereg.get("bootstrap", {})
    comparisons = {}
    for comparison in prereg.get("comparisons", []):
        a, b = comparison["a"], comparison["b"]
        per_horizon = {}
        diffs = []
        for horizon in sorted(per_window[a]):
            seeds_a = sorted(per_window[a][horizon])
            seeds_b = sorted(per_window[b][horizon])
            arr_a = np.stack([per_window[a][horizon][s] for s in seeds_a])
            arr_b = np.stack([per_window[b][horizon][s] for s in seeds_b])
            result = paired_two_level_bootstrap(arr_a, arr_b, block=horizon,
                                                n_resamples=int(boot.get("resamples", 4000)), seed=int(boot.get("seed", 0)))
            per_horizon[horizon] = result
            diffs.append(result["diff"])
        comparisons[comparison["name"]] = {"a": a, "b": b, "per_horizon": per_horizon,
                                           "endpoint_diff": float(np.mean(diffs)),
                                           "endpoint_p": float(max(r["p"] for r in per_horizon.values()))}
    rejected = holm({k: v["endpoint_p"] for k, v in comparisons.items()}, float(prereg.get("alpha", 0.05)))
    for name in comparisons:
        comparisons[name]["significant_after_holm"] = rejected[name]
    return {"prereg": prereg["id"], "runs": runs_out, "comparisons": comparisons}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prereg", required=True)
    parser.add_argument("--trainval", default=str(ROOT / "data" / "trainval"))
    args = parser.parse_args()
    vault = os.environ.get("LEAGUETST_VAULT")
    if not vault:
        print("set LEAGUETST_VAULT to the decrypted vault folder")
        return 1
    path = Path(args.prereg).resolve()
    prereg = json.loads(path.read_text(encoding="utf-8"))
    log_path = ROOT / "results" / "test_access.log"
    if not committed_unchanged(path, ROOT):
        print(f"{path} must be committed and unchanged before the test evaluation")
        return 1
    if already_opened(prereg["id"], log_path):
        print(f"the test split was already opened for {prereg['id']}; the evaluation runs once")
        return 1
    result = evaluate(prereg, Path(vault).expanduser(), Path(args.trainval), ROOT, log_path)
    out = ROOT / "results" / "final" / f"test_{prereg['id']}.json"
    save_json(result, out)
    for name, comparison in result["comparisons"].items():
        print(f"{name}: endpoint difference {comparison['endpoint_diff']:+.4f}, "
              f"p {comparison['endpoint_p']:.4f}, significant after Holm: {comparison['significant_after_holm']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
