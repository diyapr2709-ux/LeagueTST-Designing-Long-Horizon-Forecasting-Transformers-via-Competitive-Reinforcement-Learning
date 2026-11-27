"""Exploratory, post hoc: one Holm over every registered ablation test at once. CPU.

Added on 26 Nov, after every result was seen (area-chair review B12). The registered decisions are the per-family
Holm decisions (docs/prereg_ablations.md) and they stand; this is a sensitivity reading only. It reads the p
values already in the result files and recomputes nothing else: AB-C and AB-X on ILI (Holm over 3 each), AB-P
(3) and AB-G (2) on the RTX (results/ablations/LLM-rtx.json), and AB-L (2, Soumitra's). A registered test with
no p value counts as p = 1, as in each family.

    .venv/bin/python scripts/ablations/programme_holm.py   # writes results/ablations/programme-holm.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, apply_holm, provenance, write_json  # noqa: E402
from llm_rtx import TEST_ARMS  # noqa: E402

FAMILIES = {"AB-C": 3, "AB-X": 3, "AB-P": 3, "AB-G": 2, "AB-L": 2}  # registered Holm sizes
INPUTS = {"AB-C": "AB-C.json", "AB-X": "AB-X.json", "AB-P": "LLM-rtx.json", "AB-G": "LLM-rtx.json", "AB-L": "AB-L.json"}


def family_tests(name: str, data: dict) -> dict[str, dict]:
    """The registered tests of a family with their p, family Holm p and decision, as the result file holds them."""
    if name in ("AB-C", "AB-X"):
        return {k: {"p": v["p"], "p_holm": v["p_holm"], "reject": v["reject"]} for k, v in data["tables"]["ILI_T24"]["tests"].items()}
    if name in ("AB-P", "AB-G"):
        return {k: {"p": v["p"], "p_holm": v["p_holm"], "reject": v["reject"]} for k, v in data["registered"]["families"][name]["tests"].items()}
    return {k: {"p": data["tests"][k]["p"], "p_holm": h["p_holm"], "reject": h["reject"]} for k, h in data["holm"].items()}


def programme(families: dict[str, dict[str, dict]]) -> dict:
    """One Holm over all registered tests (size: the sum of the registered family sizes)."""
    size = sum(FAMILIES.values())
    tests = {f"{fam}: {name}": {"p": t["p"]} for fam, ts in families.items() for name, t in ts.items()}
    held = apply_holm(tests, size)
    rows = [{"family": fam, "test": name, "p": t["p"], "family_p_holm": t["p_holm"], "family_reject": t["reject"],
             "programme_p_holm": held[f"{fam}: {name}"]["p_holm"], "programme_reject": held[f"{fam}: {name}"]["reject"]}
            for fam, ts in families.items() for name, t in ts.items()]
    shared = sorted(f"{fam}: {name}" for fam in ("AB-P", "AB-G") for name in families.get(fam, {})
                    if ("prompted", "ref") in TEST_ARMS.get(name, []))
    return {"size": size, "tests_with_p": len(tests), "rows": rows, "tests_sharing_ABL_P_ref_local": shared}


def main() -> int:
    from common import code_hash

    if code_hash().endswith("-dirty"):
        raise SystemExit("scripts/ or leaguetst/ has uncommitted changes; commit before writing results")
    loaded = {name: json.loads((OUT / f).read_text()) for name, f in INPUTS.items()}
    families = {name: family_tests(name, loaded[name]) for name in FAMILIES}
    result = {"exploratory": "post hoc sensitivity, added 26 Nov after every result was seen; the registered decisions are per family",
              "provenance": provenance(__file__, sorted({f"results/ablations/{f}" for f in INPUTS.values()})),
              "family_sizes": FAMILIES, **programme(families)}
    write_json(OUT / "programme-holm.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
