"""Download table records from a Modal volume and append them to the ledger.

Only the small JSON records come down by default; checkpoints and
per-window errors stay on the volume until a gate or an analysis needs
them (``--with-perwindow``, ``--with-checkpoints``).

Example:
    python scripts/collect_runs.py --profile diya-027 --item E1 --dataset ETTh1 --horizon 336 --seed 2021
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leaguetst.ledger import append_rows, row_from_record  # noqa: E402

MODAL = str(ROOT / ".venv" / "bin" / "modal")


def fetch(profile: str, remote: str, local: Path) -> None:
    local.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MODAL_PROFILE": profile}
    subprocess.run([MODAL, "volume", "get", "--force", "leaguetst-runs", remote, str(local)], check=True, env=env,
                   stdout=subprocess.DEVNULL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--item", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--with-perwindow", action="store_true")
    parser.add_argument("--with-checkpoints", action="store_true")
    parser.add_argument("--no-download", action="store_true", help="use records already in results/runs")
    parser.add_argument("--ledger", default=str(ROOT / "results" / "ledger.jsonl"))
    args = parser.parse_args()

    setting = f"{args.dataset}_T{args.horizon}_s{args.seed}"
    local = ROOT / "results" / "runs" / args.item / setting
    remote = f"/{args.item}/{setting}"
    if not args.no_download:
        fetch(args.profile, f"{remote}/records", local)
        if args.with_perwindow:
            fetch(args.profile, f"{remote}/perwindow", local)
        if args.with_checkpoints:
            fetch(args.profile, f"{remote}/ckpt", local)
    records = sorted((local / "records").glob("*.json"))
    source = f"modal:{args.profile}:leaguetst-runs{remote}"
    rows = [row_from_record(json.loads(p.read_text()), source) for p in records]
    counts = append_rows(rows, args.ledger)
    print(f"{setting}: {len(records)} records, {counts['appended']} appended, {counts['skipped']} already in the ledger")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
