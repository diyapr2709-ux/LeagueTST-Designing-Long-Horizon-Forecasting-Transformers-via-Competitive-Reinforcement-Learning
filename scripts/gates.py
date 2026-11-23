"""Gates G1 to G6. A result enters the report only when every gate passes.

    G1  pytest is green.
    G2  the NumPy re-score matches, within 1e-6, every claim-bearing ledger
        row, the 20 best rows of every table setting, and a fixed 5% sample
        of all rows (on both val.mse and best_val_mse). Items that must have
        rows fail when they have none.
    G3  every test access in results/test_access.log names a committed
        prereg file and the log is unchanged since the last commit; the
        ledger only grew since the last commit; the E0 lock is committed in
        exactly one commit and every locked file still matches it.
    G4  for items that make claims, docs/claims/<item>.json exists, every
        claim has at least 5 values per arm, and every claimed win is
        recomputed here (seed-level permutation test, Holm over the file's
        claims) and holds in the stated direction.
    G5  for search items, results/search/<item>/budget.json exists and every
        run's charges recount from the ledger: a full lookup costs the epochs
        that seed's run took, a screening at most that, each config is
        charged once, and the total stays within the budget.
    G6  docs/verdicts/<item>.md has a line that is exactly "Verdict: ACCEPT".

Examples:
    python scripts/gates.py --item E1
    python scripts/gates.py --all
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leaguetst.ledger import read_rows  # noqa: E402

LEDGER = ROOT / "results" / "ledger.jsonl"
TOLERANCE = 1e-6
TOP_ROWS = 20
ROW_ITEMS = ("E1", "E3b", "E6", "PICK")
CLAIM_ITEMS = ("E3", "E4")
SEARCH_ITEMS = ("E3", "E4")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def in_sample(row_id: str) -> bool:
    """A fixed 5% of rows, chosen by digest so nobody picks them."""
    return int(hashlib.sha256(row_id.encode()).hexdigest(), 16) % 20 == 0


def g1() -> tuple[bool, str]:
    run = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-x"], cwd=ROOT, capture_output=True, text=True)
    summary = [l for l in (run.stdout + run.stderr).splitlines() if re.search(r"\d+ (passed|failed)", l)]
    return run.returncode == 0, (summary[-1].strip("= ") if summary else f"pytest exit {run.returncode}")


def checkpoint_path(row: dict) -> Path:
    setting = f"{row['dataset']}_T{row['horizon']}_s{row['seed']}"
    return ROOT / "results" / "runs" / row["item"] / setting / "ckpt" / f"{row['config_id']}.pt"


def g2(item: str) -> tuple[bool, str]:
    from leaguetst.eval.rescore import rescore_checkpoint

    rows = [r for r in read_rows(LEDGER) if r["item"] == item]
    if not rows:
        return (item not in ROW_ITEMS), (f"no ledger rows for {item}" if item in ROW_ITEMS else "item has no ledger rows to re-score")
    by_setting: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        by_setting[(row["dataset"], row["horizon"], row["seed"])].append(row)
    chosen = {r["row_id"]: r for r in rows if item in CLAIM_ITEMS or in_sample(r["row_id"])}
    for group in by_setting.values():
        for row in sorted(group, key=lambda r: r["best_val_mse"])[:TOP_ROWS]:
            chosen[row["row_id"]] = row
    missing, failed, worst = [], [], 0.0
    for row in chosen.values():
        path = checkpoint_path(row)
        if not path.exists():
            missing.append(row["config_id"])
            continue
        result = rescore_checkpoint(row, path, ROOT / "data" / "trainval")
        gap = max(result["abs_gap_mse"], abs(result["mse"] - float(row["best_val_mse"])))
        worst = max(worst, gap)
        if gap > TOLERANCE:
            failed.append(f"{row['config_id']} s{row['seed']} gap {gap:.1e}")
    if missing:
        return False, f"{len(missing)} of {len(chosen)} checkpoints not downloaded (collect_runs.py --with-checkpoints)"
    if failed:
        return False, f"{len(failed)} rows off: " + "; ".join(failed[:5])
    return True, f"{len(chosen)} rows re-scored, worst gap {worst:.1e}"


def g3() -> tuple[bool, str]:
    from leaguetst.eval.integrity import g3_problems

    problems, note = g3_problems(ROOT)
    detail = "; ".join(problems) if problems else "test access only through committed preregs; ledger and log append-only; locked files unchanged"
    return (not problems), f"{detail} ({note})" if note else detail


def g4(item: str) -> tuple[bool, str]:
    from leaguetst.eval.stats import holm, permutation_test
    from leaguetst.search.replay import Table, claim_values

    path = ROOT / "docs" / "claims" / f"{item}.json"
    if not path.exists():
        return (item not in CLAIM_ITEMS), (f"docs/claims/{item}.json is missing" if item in CLAIM_ITEMS else "item makes no claims")
    claims = json.loads(path.read_text())["claims"]
    if not claims:
        return False, "claims file holds no comparisons"
    problems, p_values, tables = [], {}, {}
    for claim in claims:
        source = claim.get("source")
        if not source:
            problems.append(f"{claim['name']}: no source to recompute from")
            continue
        key = (source["table_item"], source["dataset"], int(source["horizon"]))
        if key not in tables:
            tables[key] = Table([r for r in read_rows(LEDGER) if r["item"] == key[0] and r["dataset"] == key[1]
                                 and int(r["horizon"]) == key[2]])
        try:
            a, b = claim_values(tables[key], ROOT / source["runs_dir"], source["method"], source["baseline"], int(source["budget"]))
        except (KeyError, ValueError, FileNotFoundError) as error:
            problems.append(f"{claim['name']}: cannot recompute ({error})")
            continue
        if len(a) != len(claim["a_values"]) or len(b) != len(claim["b_values"]) or \
                not (np.allclose(a, claim["a_values"]) and np.allclose(b, claim["b_values"])):
            problems.append(f"{claim['name']}: values differ from the recomputation")
        if min(len(a), len(b)) < 5:
            problems.append(f"{claim['name']}: fewer than 5 values per arm")
            continue
        p_values[claim["name"]] = permutation_test(a, b)["p"]
        claim["_diff"] = float(np.mean(a) - np.mean(b))
    rejected = holm(p_values) if p_values else {}
    for claim in claims:
        if claim.get("claims_win") and claim["name"] in p_values:
            better = claim["_diff"] < 0 if claim.get("lower_is_better", True) else claim["_diff"] > 0
            if not (rejected[claim["name"]] and better):
                problems.append(f"{claim['name']}: win not supported (diff {claim['_diff']:+.4g}, p {p_values[claim['name']]:.3g})")
    wins = sum(bool(c.get("claims_win")) for c in claims)
    return (not problems), ("; ".join(problems[:5]) if problems else f"{len(claims)} comparisons recomputed from source; {wins} claimed wins hold after Holm")


def g5(item: str) -> tuple[bool, str]:
    paths = sorted((ROOT / "results" / "search" / item).glob("*/budget.json"))
    if not paths:
        return (item not in SEARCH_ITEMS), (f"no budget log under results/search/{item}/" if item in SEARCH_ITEMS else "item runs no search")
    results = [g5_one(path) for path in paths]
    return all(ok for ok, _ in results), "; ".join(detail for _, detail in results)


def g5_one(path: Path) -> tuple[bool, str]:
    log = json.loads(path.read_text())
    if not log.get("runs"):
        return False, f"{path.parent.name}: budget log has no runs"
    runs_by_key = {(r["config_id"], int(r["seed"])): int(r["epochs_charged"]) for r in read_rows(LEDGER)
                   if r["item"] == log["table_item"] and r["dataset"] == log["dataset"] and int(r["horizon"]) == int(log["horizon"])}
    problems = []
    for run in log["runs"]:
        totals: dict[tuple, int] = defaultdict(int)
        label = f"{run['method']} seed {run['seed']}"
        if not run["charges"]:
            problems.append(f"{label}: charged nothing")
        for cid, seed, epochs, kind, read_epoch in run["charges"]:
            key = (cid, int(seed))
            totals[key] += int(epochs)
            if key not in runs_by_key:
                problems.append(f"{label}: {key} is not in the table")
                continue
            ran = runs_by_key[key]
            expected = ran if kind == "full" else min(int(read_epoch), ran)
            if totals[key] != expected:
                problems.append(f"{label}: {key} charged {totals[key]} by its {kind} lookup, expected {expected}")
        if len({cid for cid, _ in totals}) != len(totals):
            problems.append(f"{label}: a config was charged under two seeds")
        spent = sum(totals.values())
        if spent != run["spent"] or spent > log["budget_epochs"]:
            problems.append(f"{label}: recount {spent}, logged {run['spent']}, budget {log['budget_epochs']}")
    where = f"{log['dataset']} T{log['horizon']}"
    return (not problems), (f"{where}: " + "; ".join(problems[:5]) if problems else f"{where}: {len(log['runs'])} runs recounted within {log['budget_epochs']} epochs")


def g6(item: str) -> tuple[bool, str]:
    path = ROOT / "docs" / "verdicts" / f"{item}.md"
    if not path.exists():
        return False, f"docs/verdicts/{item}.md is missing"
    verdicts = [l.strip() for l in path.read_text().splitlines() if l.strip().startswith("Verdict:")]
    ok = bool(verdicts) and verdicts[-1] == "Verdict: ACCEPT"
    return ok, ("critic accepted" if ok else f"last verdict line: {verdicts[-1] if verdicts else 'none'}")


def run_gates(item: str) -> bool:
    results = {"G1": g1(), "G2": g2(item), "G3": g3(), "G4": g4(item), "G5": g5(item), "G6": g6(item)}
    for gate, (ok, detail) in results.items():
        print(f"{item} {gate} {'PASS' if ok else 'FAIL'}  {detail}", flush=True)
    return all(ok for ok, _ in results.values())


def write_lock() -> int:
    from leaguetst.eval.integrity import write_lock as write

    try:
        path = write(ROOT)
    except FileExistsError as error:
        print(error)
        return 1
    print(f"wrote {path}; commit it now")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--item")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--write-e0-lock", action="store_true", help="record the locked files' digests (once)")
    args = parser.parse_args()
    if args.write_e0_lock:
        return write_lock()
    if args.all:
        verdicts = ROOT / "docs" / "verdicts"
        items = sorted(p.stem for p in verdicts.glob("*.md")) if verdicts.exists() else []
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
