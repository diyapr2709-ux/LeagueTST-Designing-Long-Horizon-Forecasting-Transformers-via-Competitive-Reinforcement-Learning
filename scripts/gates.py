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
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leaguetst.ledger import read_rows  # noqa: E402

LEDGER = ROOT / "results" / "ledger.jsonl"
LOCK = ROOT / "prereg" / "e0_lock.json"
LOCKED_FILES = (
    "leaguetst/data/datasets.py",
    "leaguetst/data/trainval.py",
    "leaguetst/data/test_split.py",
    "data/trainval/MANIFEST.json",
    "leaguetst/eval/metrics.py",
    "leaguetst/eval/numpy_model.py",
    "leaguetst/eval/rescore.py",
    "leaguetst/eval/stats.py",
    "scripts/eval_test.py",
    "scripts/prereg.py",
)
TOLERANCE = 1e-6
TOP_ROWS = 20
ROW_ITEMS = ("E1", "E3b", "E6", "PICK")
CLAIM_ITEMS = ("E3", "E4", "E6")
SEARCH_ITEMS = ("E3", "E4")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def committed_clean(rel: str) -> bool:
    return git("ls-files", "--error-unmatch", rel).returncode == 0 and git("diff", "--quiet", "HEAD", "--", rel).returncode == 0


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
    problems = []
    log = ROOT / "results" / "test_access.log"
    if log.exists():
        rel = str(log.relative_to(ROOT))
        if not committed_clean(rel):
            problems.append("results/test_access.log changed since the last commit")
        for line in log.read_text().splitlines():
            if line.strip():
                prereg = json.loads(line).get("prereg", "")
                if not committed_clean(f"prereg/{prereg}.json"):
                    problems.append(f"test access for {prereg!r} without a committed prereg file")
    head_ledger = git("show", "HEAD:results/ledger.jsonl")
    if head_ledger.returncode == 0 and LEDGER.exists():
        if not LEDGER.read_text().startswith(head_ledger.stdout):
            problems.append("results/ledger.jsonl lost or changed rows since the last commit")
    rel_lock = str(LOCK.relative_to(ROOT))
    if not LOCK.exists() or not committed_clean(rel_lock):
        problems.append("prereg/e0_lock.json is missing or not committed unchanged")
    else:
        history = [h for h in git("log", "--format=%H", "--", rel_lock).stdout.split() if h]
        if len(history) != 1:
            problems.append(f"prereg/e0_lock.json was changed after it was written ({len(history)} commits)")
        lock = json.loads(LOCK.read_text())["files"]
        changed = [f for f in LOCKED_FILES if f not in lock or sha(ROOT / f) != lock[f]]
        if changed:
            problems.append("changed since the E0 lock: " + ", ".join(changed))
    return (not problems), ("; ".join(problems) if problems else "test access only through committed preregs; ledger append-only; locked files unchanged")


def g4(item: str) -> tuple[bool, str]:
    from leaguetst.eval.stats import holm, permutation_test

    path = ROOT / "docs" / "claims" / f"{item}.json"
    if not path.exists():
        return (item not in CLAIM_ITEMS), (f"docs/claims/{item}.json is missing" if item in CLAIM_ITEMS else "item makes no claims")
    claims = json.loads(path.read_text())["claims"]
    if not claims:
        return True, "claims file lists no claimed win"
    problems, p_values = [], {}
    for claim in claims:
        a, b = claim["a_values"], claim["b_values"]
        if min(len(a), len(b)) < 5:
            problems.append(f"{claim['name']}: fewer than 5 values per arm")
        p_values[claim["name"]] = permutation_test(a, b)["p"]
    rejected = holm(p_values)
    for claim in claims:
        if claim.get("claims_win"):
            diff = sum(claim["a_values"]) / len(claim["a_values"]) - sum(claim["b_values"]) / len(claim["b_values"])
            if not rejected[claim["name"]] or diff >= 0:
                problems.append(f"{claim['name']}: win not supported (diff {diff:+.4g}, p {p_values[claim['name']]:.3g})")
    return (not problems), ("; ".join(problems) if problems else f"{len(claims)} claims recomputed; every claimed win holds after Holm")


def g5(item: str) -> tuple[bool, str]:
    path = ROOT / "results" / "search" / item / "budget.json"
    if not path.exists():
        return (item not in SEARCH_ITEMS), (f"results/search/{item}/budget.json is missing" if item in SEARCH_ITEMS else "item runs no search")
    log = json.loads(path.read_text())
    runs_by_key = {(r["config_id"], int(r["seed"])): int(r["epochs_charged"]) for r in read_rows(LEDGER)
                   if r["item"] == log["table_item"] and r["dataset"] == log["dataset"] and int(r["horizon"]) == int(log["horizon"])}
    problems = []
    for run in log["runs"]:
        totals: dict[tuple, int] = defaultdict(int)
        kinds: dict[tuple, list] = defaultdict(list)
        for cid, seed, epochs, kind in run["charges"]:
            totals[(cid, int(seed))] += int(epochs)
            kinds[(cid, int(seed))].append(kind)
        label = f"{run['method']} seed {run['seed']}"
        for key, total in totals.items():
            if key not in runs_by_key:
                problems.append(f"{label}: {key} is not in the table")
            elif total > runs_by_key[key] or ("full" in kinds[key] and total != runs_by_key[key]):
                problems.append(f"{label}: {key} charged {total}, table run took {runs_by_key[key]}")
        if len({cid for cid, _ in totals}) != len(totals):
            problems.append(f"{label}: a config was charged under two seeds")
        spent = sum(totals.values())
        if spent != run["spent"] or spent > log["budget_epochs"]:
            problems.append(f"{label}: recount {spent}, logged {run['spent']}, budget {log['budget_epochs']}")
    return (not problems), ("; ".join(problems[:5]) if problems else f"{len(log['runs'])} runs recounted within {log['budget_epochs']} epochs")


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
    """Record the digests of the locked files, once, at the E0 commit."""
    if LOCK.exists():
        print(f"{LOCK} exists; the E0 lock is written once")
        return 1
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps({"files": {f: sha(ROOT / f) for f in LOCKED_FILES}}, indent=2, sort_keys=True) + "\n")
    print(f"wrote {LOCK}; commit it now")
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
