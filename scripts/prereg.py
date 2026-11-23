"""Pre-register the final models before the one test evaluation.

Reads a spec (JSON) that lists each arm, its checkpoints at every horizon
and seed, and the comparisons to test. Checks every checkpoint digest, at
least five seeds per compared arm, and that no test access exists for the
id. Writes prereg/<id>.json, which must be committed before
scripts/eval_test.py will run it. An existing prereg file is never
overwritten.

Spec format:
    {"id": "final_v1",
     "primary_endpoint": "mean exact test MSE over the four horizons of each dataset",
     "alpha": 0.05, "bootstrap": {"resamples": 4000, "seed": 0},
     "arms": [{"name": "winner_ETTh1", "dataset": "ETTh1", "model_config": {...},
               "runs": [{"horizon": 96, "seed": 1, "checkpoint": "results/final/...pt",
                         "sha256": "...", "batch_size": 128}, ...]}, ...],
     "comparisons": [{"name": "ETTh1 winner vs PatchTST/42", "a": "winner_ETTh1", "b": "patchtst42_ETTh1"}]}

Example:
    python scripts/prereg.py --spec results/final/prereg_spec.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leaguetst.utils import sha256_file  # noqa: E402

MIN_SEEDS = 5
ACCESS_LOG = ROOT / "results" / "test_access.log"


def validate(spec: dict, root: Path = ROOT, access_log: Path = ACCESS_LOG) -> list[str]:
    """Problems with a spec; an empty list means it can be registered."""
    problems = []
    if not spec.get("id") or not str(spec["id"]).replace("_", "").replace("-", "").isalnum():
        problems.append("id must be letters, digits, '-' or '_'")
    arms = {arm["name"]: arm for arm in spec.get("arms", [])}
    if not arms:
        problems.append("no arms")
    for arm in arms.values():
        for run in arm.get("runs", []):
            path = root / run["checkpoint"]
            if not path.exists():
                problems.append(f"{arm['name']}: missing {run['checkpoint']}")
            elif sha256_file(path) != run["sha256"]:
                problems.append(f"{arm['name']}: digest mismatch for {run['checkpoint']}")
    for comparison in spec.get("comparisons", []):
        for side in ("a", "b"):
            name = comparison.get(side)
            if name not in arms:
                problems.append(f"comparison {comparison.get('name')}: unknown arm {name}")
                continue
            seeds = Counter(run["horizon"] for run in arms[name]["runs"])
            if not seeds or min(seeds.values()) < MIN_SEEDS:
                problems.append(f"arm {name} has fewer than {MIN_SEEDS} seeds at some horizon")
        if comparison.get("a") in arms and comparison.get("b") in arms:
            ha = {r["horizon"] for r in arms[comparison["a"]]["runs"]}
            hb = {r["horizon"] for r in arms[comparison["b"]]["runs"]}
            if ha != hb or arms[comparison["a"]]["dataset"] != arms[comparison["b"]]["dataset"]:
                problems.append(f"comparison {comparison.get('name')}: arms differ in dataset or horizons")
    if access_log.exists() and any(json.loads(l).get("prereg") == spec.get("id") for l in access_log.read_text().splitlines() if l.strip()):
        problems.append(f"the test split was already opened for {spec.get('id')}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    problems = validate(spec)
    if problems:
        print("prereg refused:\n  " + "\n  ".join(problems))
        return 1
    out = ROOT / "prereg" / f"{spec['id']}.json"
    if out.exists():
        print(f"{out} exists; a prereg is written once")
        return 1
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    registered = {
        **spec,
        "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_hash": head,
        "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(registered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}; commit it, then the owner runs scripts/eval_test.py --prereg {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
