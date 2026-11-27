"""Compare the result of record (results/ablations/LLM-registered.json, AB4/AB6) with the independent re-scores.

Re-scores: Anirudh's scripts/ablations/llm_rtx.py (results/ablations/LLM-rtx.json, AB5), the verifier's blind
recomputation (results/ablations/verify_rtx/ab4/recompute_ab4.json) and Soumitra's scripts/ablations/rtx_tests.py
(results/ablations/rtx/RTX_tests.json). Exact fields (diff, interval, means, rho, reject) must agree to 1e-9;
permutation p values may differ by Monte Carlo error (tolerance 0.005) because the verifier draws its own permutations.
Writes results/ablations/LLM-registered_comparison.json and prints one line per field. Descriptive; no test is defined here.

    .venv/bin/python scripts/ablations/compare_registered.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "ablations"
RECORD = OUT / "LLM-registered.json"
SOURCES = {
    "llm_rtx": (OUT / "LLM-rtx.json", lambda d: {**d["registered"]["families"]["AB-P"]["tests"],
                                                   **d["registered"]["families"]["AB-G"]["tests"]}),
    "verifier": (OUT / "verify_rtx" / "ab4" / "recompute_ab4.json", lambda d: d["tests"]),
    "rtx_tests": (OUT / "rtx" / "RTX_tests.json", lambda d: {**d["AB-P"]["tests"], **d["AB-G"]["tests"]}),
}
# record name -> names used by the re-scores
ALIASES = {"anon vs ref": ["anon vs ref_local"], "repair vs ref": ["repair vs ref_local"],
           "scale trend": ["scale trend"], "KL trend": ["KL trend"], "step trend": ["steps trend", "step trend"]}
EXACT = ("diff", "low", "high", "rho", "n", "reject")
MONTE_CARLO = ("p", "p_holm")
P_TOL = 0.005


def flatten(test: dict) -> dict:
    out = {}
    for k in EXACT + MONTE_CARLO:
        if k in test and not isinstance(test[k], (list, dict)):
            out[k] = test[k]
    if isinstance(test.get("mean"), list):
        out["mean_arm"], out["mean_ref"] = test["mean"]
    for level, stats in (test.get("levels") or {}).items():
        try:
            name = f"{float(level):g}"
        except ValueError:
            name = level
        for s in ("mean", "low", "high", "n"):
            if s in stats:
                out[f"level {name} {s}"] = stats[s]
    return out


def lookup(tests: dict, record_name: str) -> dict | None:
    for name in ALIASES[record_name]:
        if name in tests:
            return tests[name]
    return None


def main() -> int:
    record = json.load(open(RECORD))["families"]
    rec_tests = {**record["AB-P"]["tests"], **record["AB-G"]["tests"]}
    rows, disagreements = [], []
    for source, (path, extract) in SOURCES.items():
        if not path.exists():
            rows.append({"source": source, "status": "missing", "path": str(path.relative_to(ROOT))})
            continue
        other = extract(json.load(open(path)))
        for name, rec in rec_tests.items():
            oth = lookup(other, name)
            if oth is None:
                rows.append({"source": source, "test": name, "status": "absent in re-score"})
                continue
            a, b = flatten(rec), flatten(oth)
            for field, va in a.items():
                if field not in b:
                    continue
                vb = b[field]
                if field in MONTE_CARLO or field.endswith(" p"):
                    ok = abs(float(va) - float(vb)) <= P_TOL
                    kind = "MC"
                elif isinstance(va, bool) or isinstance(vb, bool):
                    ok, kind = bool(va) == bool(vb), "exact"
                else:
                    ok, kind = abs(float(va) - float(vb)) <= 1e-9, "exact"
                row = {"source": source, "test": name, "field": field, "record": va, "rescore": vb, "kind": kind, "ok": ok}
                rows.append(row)
                if not ok:
                    disagreements.append(row)
                print(f"{'OK ' if ok else 'DIFF'} {source:10} {name:16} {field:22} record {va!s:>22} rescore {vb!s:>22}")
    result = {"record": str(RECORD.relative_to(ROOT)), "fields_compared": sum(1 for r in rows if "ok" in r),
              "disagreements": disagreements, "rows": rows, "p_tolerance_monte_carlo": P_TOL}
    (OUT / "LLM-registered_comparison.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"{result['fields_compared']} fields compared, {len(disagreements)} disagreements")
    return 1 if disagreements else 0


if __name__ == "__main__":
    sys.exit(main())
