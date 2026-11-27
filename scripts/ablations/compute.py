"""Compute per arm (descriptive): target epochs, source epochs and recorded seconds, with the card. CPU.

Added on 26 Nov, after every result was seen (area-chair review D, NeurIPS checklist item 8; results review R13).
No test and no claim: it sums what the committed records hold.

* Search records: "spent" is the target-table epochs a search was charged (the equal-epoch budget); "seconds" is
  the wall time of one search, where the runner recorded it (the LLM arms; the CPU replays record none).
* Training records: "training_epochs" is the source-table epochs charged while training one policy; "seconds"
  its wall time. Error records hold no seconds; they are counted.
* Card: env.gpu_name, else the record's gpu field; "not recorded" otherwise.
* The RTX queue status files give each job's wall clock from its START line to its DONE or INTERRUPTED line,
  which includes model loading and anything outside a record.
* Dollars: only what a committed file records (results/logs/ab2_modal_billing.txt at the pinned RTX commit, a
  month-to-date Modal summary of Soumitra's profile); it is not split per arm.

Sources: E3 and E4 runs under results/search/, E4 training under results/runs/E4/, the RTX items (AB2, AB4) at
llm_rtx.RTX_REF through llm_rtx.RtxSource, the Mac items (AB3) under results/ablations/mps/.

    .venv/bin/python scripts/ablations/compute.py   # writes results/ablations/compute.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROOT, provenance, read_runs, write_json  # noqa: E402
from llm_rtx import PROMPTED, RTX_DIR, RTX_REF, STATUS_FILES, TRANSFER, RtxSource, item_rows  # noqa: E402

NOT_RECORDED = "not recorded"
# E4's first training attempt (23 Nov, diyapr), superseded by A25 and never deployed (results review, 26 Nov)
SUPERSEDED = {"train_ETTh1", "train_ILI"}
SUPERSEDED_LABEL = "superseded (A25), never deployed, compute only"
BILLING = "results/logs/ab2_modal_billing.txt"
E4_SPEND_COMMIT = "22111daa57f1e15c4fe8502d847760870e560b68"  # its message states the metered spend of the E4 rerun (J1 and J2)


def card(record: dict) -> str:
    return str(record.get("env", {}).get("gpu_name") or record.get("gpu") or NOT_RECORDED)


def group(records: list[dict], kind: str) -> dict:
    """Totals of one arm's search or training records."""
    seconds = [float(r["seconds"]) for r in records if r.get("seconds") is not None]
    by_card: dict[str, float] = defaultdict(float)
    for r in records:
        if r.get("seconds") is not None:
            by_card[card(r)] += float(r["seconds"])
    epochs_key = "spent" if kind == "search" else "training_epochs"
    epochs = [float(r[epochs_key]) for r in records if r.get(epochs_key) is not None]
    return {"kind": kind, "records": len(records), "cards": dict(sorted(Counter(card(r) for r in records).items())),
            "target_epochs" if kind == "search" else "source_epochs": sum(epochs) if len(epochs) == len(records) and records else None,
            "seconds": sum(seconds) if seconds else None, "records_with_seconds": len(seconds), "seconds_by_card": dict(sorted(by_card.items()))}


def by_method(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        out[r["method"]].append(r)
    return dict(sorted(out.items()))


def search_rows(label: str, records: list[dict], machine: str | None, setting: str) -> list[dict]:
    """One row per method; with ``machine`` None, "Modal" if a record names a card, else "CPU replay"."""
    return [{"arm": f"{label}: {method}", "machine": machine or ("Modal" if any(card(r) != NOT_RECORDED for r in rs) else "CPU replay"),
             "table": setting, **group(rs, "search")} for method, rs in by_method(records).items()]


def committed_rows() -> list[dict]:
    """E3 and E4 (Modal and CPU), from results/search/ and results/runs/E4/."""
    rows = []
    for item in ("E3", "E4"):
        for folder in sorted((ROOT / "results/search" / item).glob("*_T*")):
            for path in sorted(folder.glob("*_runs.jsonl")):
                rows += search_rows(f"{item} {path.name.removesuffix('_runs.jsonl')}", read_runs(path), None, folder.name)
    for folder in sorted((ROOT / "results/runs/E4").iterdir()):
        records = read_runs(folder / "training.jsonl")
        errors = read_runs(folder / "errors.jsonl")
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            groups[f"{r['job'].get('rivals')}, source {r['job'].get('dataset')}"].append(r)
        old = folder.name in SUPERSEDED
        label = f"E4 training ({folder.name}{', ' + SUPERSEDED_LABEL if old else ''})"
        for name, rs in sorted(groups.items()):
            rows.append({"arm": f"{label}: {name}", "machine": "Modal", "table": folder.name, "superseded": old,
                         **group(rs, "training"), **({"kind": "training, " + SUPERSEDED_LABEL} if old else {})})
        if errors:
            rows.append({"arm": f"{label}: error records", "machine": "Modal", "table": folder.name, "superseded": old,
                         "kind": "errors", "records": len(errors), "cards": {}, "seconds": None, "records_with_seconds": 0, "seconds_by_card": {}})
    return rows


def rtx_rows(src: RtxSource) -> list[dict]:
    rows = []
    for arm, (item, _) in PROMPTED.items():
        for setting in ("ILI_T24", "ETTh1_T336"):
            runs = src.runs(item, setting)
            if runs:
                rows.append({"arm": f"RTX {item}", "machine": "RTX (local)", "table": setting, **group(runs, "search")})
    for arm, item in TRANSFER.items():
        runs = src.runs(item, "ILI_T24")
        if runs:
            rows.append({"arm": f"RTX {item} (deployment)", "machine": "RTX (local)", "table": "ILI_T24", **group(runs, "search")})
        records = item_rows(src, item, "training.jsonl")
        if records:
            rows.append({"arm": f"RTX {item} (training)", "machine": "RTX (local)", "table": "ETTh1_T336 (source)", **group(records, "training")})
    return rows


def mac_rows() -> list[dict]:
    rows = []
    for folder in sorted((OUT / "mps").glob("*/*")):
        for path in sorted(folder.glob("*_runs.jsonl")):
            rows += search_rows(f"Mac {folder.parent.name}", read_runs(path), "Mac (local)", folder.name)
        records = read_runs(folder / "training.jsonl")
        if records:
            rows.append({"arm": f"Mac {folder.parent.name} (training)", "machine": "Mac (local)", "table": "ETTh1_T336 (source)", **group(records, "training")})
    return rows


def queue_wall_clock(lines: list[str]) -> list[dict]:
    """Each job's wall clock from its START line to the next DONE or INTERRUPTED line of the same name."""
    started: dict[str, str] = {}
    out = []
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        name = parts[2].rstrip(":")
        if parts[1] == "START":
            started[name] = parts[0]
        elif parts[1] in ("DONE", "INTERRUPTED") and name in started:
            begin = started.pop(name)
            out.append({"job": name, "start": begin, "end": parts[0], "outcome": parts[1] if parts[1] == "INTERRUPTED" else " ".join(parts[1:2] + parts[3:4]),
                        "seconds": (datetime.fromisoformat(parts[0]) - datetime.fromisoformat(begin)).total_seconds()})
    return out


def billing(src: RtxSource) -> dict | None:
    """The committed Modal billing summary, if any: its comment lines and metered cost, as recorded."""
    data = src._read(BILLING)
    if not data:
        return None
    text = data.decode()
    comments = [line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("#")]
    body = json.loads("\n".join(line for line in text.splitlines() if not line.startswith("#")))
    import re

    read = next((m.group(1) for c in comments for m in [re.search(r"on (\d{4}-\d{2}-\d{2}T[0-9:]+[+-]\d{2}:\d{2})", c)] if m), None)
    return {"path": BILLING, "commit": src.commit, "comments": comments, "metered_cost": body.get("metered_cost"),
            "billed_cost": body.get("billed_cost"), "read_at": read}


def e4_spend(src: RtxSource) -> dict | None:
    """The E4 rerun's metered spend as commit 22111da's message states it (round 2: both dollar figures from their sources)."""
    import re
    import subprocess

    try:
        message = src._run("log", "-1", "--format=%B", E4_SPEND_COMMIT).decode()
        time = src._run("log", "-1", "--format=%cI", E4_SPEND_COMMIT).decode().strip()
    except subprocess.CalledProcessError:
        return None
    line = next((x.strip() for x in message.splitlines() if re.match(r"Spend for J1 and J2 together: metered \$[0-9.]+", x.strip())), None)
    if not line:
        return None
    return {"commit": E4_SPEND_COMMIT, "time": time, "line": line, "metered": float(re.search(r"metered \$([0-9.]+)", line).group(1))}


def totals(rows: list[dict]) -> list[dict]:
    """Recorded seconds per machine, card and kind (search or training), summed record by record."""
    out: dict[tuple, float] = defaultdict(float)
    for r in rows:
        for c, seconds in r.get("seconds_by_card", {}).items():
            out[(r["machine"], c, r["kind"])] += seconds
    return [{"machine": m, "card": c, "kind": k, "seconds": s, "hours": s / 3600} for (m, c, k), s in sorted(out.items())]


def main() -> int:
    from common import code_hash

    if code_hash().endswith("-dirty"):
        raise SystemExit("scripts/ or leaguetst/ has uncommitted changes; commit before writing results")
    src = RtxSource(RTX_REF)
    rows = committed_rows() + rtx_rows(src) + mac_rows()
    queue = {rel: queue_wall_clock((src._read(rel) or b"").decode().splitlines()) for rel in STATUS_FILES}
    inputs = [str(p.relative_to(ROOT)) for p in sorted((ROOT / "results/search").glob("E[34]/*_T*/*_runs.jsonl"))]
    inputs += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "results/runs/E4").glob("*/*.jsonl"))]
    inputs += [str(p.relative_to(ROOT)) for p in sorted((OUT / "mps").glob("*/*/*.jsonl"))]
    write_json(OUT / "compute.json", {
        "descriptive": "added 26 Nov after every result was seen; sums of what the records hold, no test",
        "provenance": provenance(__file__, inputs), "rtx_source": {"commit": src.commit, "files": src.files, "dir": RTX_DIR},
        "rows": rows, "totals": totals(rows), "queue_wall_clock": queue, "dollars": billing(src), "e4_spend": e4_spend(src)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
