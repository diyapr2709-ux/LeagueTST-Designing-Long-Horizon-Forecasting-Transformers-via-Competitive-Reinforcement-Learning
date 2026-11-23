"""Gates G1 to G6. A result enters the report only when every gate passes.

    G1  pytest is green.
    G2  the NumPy re-score matches every claim-bearing ledger row and a fixed
        5% sample of table rows within 1e-6 (needs their checkpoints locally).
    G3  results/test_access.log is unchanged since the last commit and holds
        no access before the test day, and the split code and manifest still
        match the E0 lock.
    G4  every claimed win in docs/claims/<item>.json has at least 5 seeds per
        arm and a paired interval that excludes zero after Holm correction.
    G5  every method in results/search/<item>/budget.json used no more than
        the common epoch budget, and no less than it minus one lookup.
    G6  docs/verdicts/<item>.md holds the critic's verdict ACCEPT.

Examples:
    python scripts/gates.py --item E1
    python scripts/gates.py --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leaguetst.ledger import read_rows  # noqa: E402

LEDGER = ROOT / "results" / "ledger.jsonl"
LOCK = ROOT / "prereg" / "e0_lock.json"
LOCKED_FILES = ("leaguetst/data/datasets.py", "leaguetst/data/trainval.py", "leaguetst/data/test_split.py",
                "data/trainval/MANIFEST.json", "leaguetst/eval/rescore.py", "leaguetst/eval/numpy_model.py")
TOLERANCE = 1e-6
CLAIM_ITEMS = ("E6",)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def in_sample(row_id: str) -> bool:
    """A fixed 5% of rows, chosen by digest so nobody picks them."""
    return int(hashlib.sha256(row_id.encode()).hexdigest(), 16) % 20 == 0


def g1() -> tuple[bool, str]:
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"], cwd=ROOT,
                         capture_output=True, text=True)
    tail = (run.stdout + run.stderr).strip().splitlines()[-1:] or [""]
    return run.returncode == 0, tail[0]


def checkpoint_path(row: dict) -> Path:
    setting = f"{row['dataset']}_T{row['horizon']}_s{row['seed']}"
    return ROOT / "results" / "runs" / row["item"] / setting / "ckpt" / f"{row['config_id']}.pt"


def g2(item: str | None) -> tuple[bool, str]:
    from leaguetst.eval.rescore import rescore_checkpoint

    rows = [r for r in read_rows(LEDGER) if item is None or r["item"] == item]
    chosen = [r for r in rows if r["item"] in CLAIM_ITEMS or in_sample(r["row_id"])]
    if not chosen:
        return True, "no ledger rows to re-score for this item"
    missing, worst, failed = [], 0.0, []
    for row in chosen:
        path = checkpoint_path(row)
        if not path.exists():
            missing.append(row["config_id"])
            continue
        result = rescore_checkpoint(row, path, ROOT / "data" / "trainval")
        worst = max(worst, result["abs_gap_mse"])
        if result["abs_gap_mse"] > TOLERANCE:
            failed.append(f"{row['config_id']} gap {result['abs_gap_mse']:.1e}")
    if missing:
        return False, f"{len(missing)} of {len(chosen)} checkpoints not downloaded (collect_runs.py --with-checkpoints)"
    if failed:
        return False, f"{len(failed)} rows off: " + "; ".join(failed[:5])
    return True, f"{len(chosen)} rows re-scored, worst gap {worst:.1e}"


def g3() -> tuple[bool, str]:
    log = ROOT / "results" / "test_access.log"
    if log.exists():
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(log.relative_to(ROOT))], cwd=ROOT,
                                 capture_output=True).returncode == 0
        clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(log.relative_to(ROOT))], cwd=ROOT).returncode == 0
        if not (tracked and clean):
            return False, "results/test_access.log changed since the last commit"
        lines = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        preregs = {l["prereg"] for l in lines}
        unknown = [p for p in preregs if not (ROOT / "prereg" / f"{p}.json").exists()]
        if unknown:
            return False, f"test access without a prereg file: {unknown}"
    if not LOCK.exists():
        return False, "prereg/e0_lock.json is missing"
    lock = json.loads(LOCK.read_text())
    changed = [f for f in LOCKED_FILES if sha(ROOT / f) != lock["files"].get(f)]
    if changed:
        return False, "changed since the E0 lock: " + ", ".join(changed)
    return True, "no test access outside a prereg; split code and manifest match the E0 lock"


def g4(item: str) -> tuple[bool, str]:
    from leaguetst.eval.stats import holm

    path = ROOT / "docs" / "claims" / f"{item}.json"
    if not path.exists():
        return True, "no win claimed for this item"
    claims = json.loads(path.read_text())["claims"]
    problems = [c["name"] for c in claims if min(c["seeds_a"], c["seeds_b"]) < 5]
    rejected = holm({c["name"]: c["p"] for c in claims})
    for claim in claims:
        if claim.get("claims_win") and not (rejected[claim["name"]] and (claim["high"] < 0 or claim["low"] > 0)):
            problems.append(f"{claim['name']} interval or Holm")
    return (not problems), ("all claimed wins hold" if not problems else "; ".join(problems))


def g5(item: str) -> tuple[bool, str]:
    path = ROOT / "results" / "search" / item / "budget.json"
    if not path.exists():
        return True, "no search budget log for this item"
    log = json.loads(path.read_text())
    budget = log["budget_epochs"]
    costs = {r["config_id"]: r["epochs_charged"] for r in read_rows(LEDGER) if r["item"] == log["table_item"]
             and r["dataset"] == log["dataset"] and int(r["horizon"]) == int(log["horizon"]) and int(r["seed"]) == int(log["table_seed"])}
    largest = max(costs.values()) if costs else 0
    problems = []
    for run in log["runs"]:
        charged = sum(costs[c] for c in set(run["looked_up"]))
        if charged != run["charged"]:
            problems.append(f"{run['method']} seed {run['seed']}: logged {run['charged']} vs recount {charged}")
        elif charged > budget or charged < budget - largest:
            problems.append(f"{run['method']} seed {run['seed']}: {charged} epochs against budget {budget}")
    return (not problems), (f"{len(log['runs'])} runs within budget {budget}" if not problems else "; ".join(problems[:5]))


def g6(item: str) -> tuple[bool, str]:
    path = ROOT / "docs" / "verdicts" / f"{item}.md"
    if not path.exists():
        return False, f"docs/verdicts/{item}.md is missing"
    text = path.read_text()
    return ("Verdict: ACCEPT" in text), ("critic accepted" if "Verdict: ACCEPT" in text else "critic did not accept")


def run_gates(item: str) -> bool:
    results = {"G1": g1(), "G2": g2(item), "G3": g3(), "G4": g4(item), "G5": g5(item), "G6": g6(item)}
    for gate, (ok, detail) in results.items():
        print(f"{item} {gate} {'PASS' if ok else 'FAIL'}  {detail}", flush=True)
    return all(ok for ok, _ in results.values())


def write_lock() -> int:
    """Record the digests of the split code and manifest, once, at the E0 commit."""
    if LOCK.exists():
        print(f"{LOCK} exists; the E0 lock is written once")
        return 1
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps({"files": {f: sha(ROOT / f) for f in LOCKED_FILES}}, indent=2, sort_keys=True) + "\n")
    print(f"wrote {LOCK}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--item")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--write-e0-lock", action="store_true", help="record the E0 digests (once)")
    args = parser.parse_args()
    if args.write_e0_lock:
        return write_lock()
    if args.all:
        items = sorted(p.stem for p in (ROOT / "docs" / "verdicts").glob("*.md")) if (ROOT / "docs" / "verdicts").exists() else []
        if not items:
            print("no items with verdicts yet")
            return 1
    elif args.item:
        items = [args.item]
    else:
        parser.error("give --item or --all")
    ok = all([run_gates(item) for item in items])
    print("ALL GATES PASS" if ok else "GATES FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
