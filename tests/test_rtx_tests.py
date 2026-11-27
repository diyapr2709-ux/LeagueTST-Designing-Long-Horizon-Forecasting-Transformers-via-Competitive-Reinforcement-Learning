"""scripts/ablations/rtx_tests.py on synthetic data only (no real AB2/AB4 outcome is computed here)."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ABL = ROOT / "scripts" / "ablations"
sys.path.insert(0, str(ABL))

try:
    import rtx_tests  # noqa: E402
except SystemExit as refusal:  # after the merge of origin/main, llm_tests.py is Diya's AB4 version (08c72a4c...)
    pytest.skip(f"rtx_tests.py refuses to import ({refusal}); its single run is recorded in "
                "results/ablations/rtx/RTX_tests.json (AB5)", allow_module_level=True)


def test_pinned_copies_match_5abbd20():
    for name, digest in rtx_tests.PINNED.items():
        assert hashlib.sha256((ABL / name).read_bytes()).hexdigest() == digest
    assert rtx_tests.PINNED["llm_tests.py"] == "01b70835dd128001c78eceb040590b7e79ace2a17f260a3b98022cdbcd0ee0a6"  # AB4


def test_registered_arms_and_levels():
    assert rtx_tests.P_ARMS["ref"] == ("ABL_P_ref_local", "qwen_prompted_1.5b_local")
    assert rtx_tests.P_ARMS["anon"][1] == "qwen_prompted_anon_1.5b_local"
    assert rtx_tests.P_ARMS["repair"][1] == "qwen_prompted_repair_1.5b_local"
    assert rtx_tests.G_STEPS == {30: "ABL_G_steps30_local", 60: "ABL_G_ref_local", 120: "ABL_G_steps120_local"}
    assert rtx_tests.G_BETA == {0.0: "ABL_G_beta0_local", 0.04: "ABL_G_ref_local", 0.1: "ABL_G_beta01_local"}
    assert tuple(rtx_tests.STEP_LEVELS) == (0, 30, 60, 120)


def _write(folder: Path, rows):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "x_runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_values_refuses_duplicate_seed_and_foreign_method(tmp_path, monkeypatch):
    monkeypatch.setattr(rtx_tests, "RTX", tmp_path)
    _write(tmp_path / "A" / "ILI_T24", [{"seed": 0, "method": "m"}, {"seed": 0, "method": "m"}])
    with pytest.raises(SystemExit):
        rtx_tests.values("A", "m", table=None)
    _write(tmp_path / "B" / "ILI_T24", [{"seed": 0, "method": "m"}, {"seed": 1, "method": "other"}])
    with pytest.raises(SystemExit):
        rtx_tests.values("B", "m", table=None)


def test_values_in_seed_order(tmp_path, monkeypatch):
    monkeypatch.setattr(rtx_tests, "RTX", tmp_path)
    _write(tmp_path / "A" / "ILI_T24", [{"seed": s, "method": "m"} for s in (2, 0, 1)])
    monkeypatch.setattr(rtx_tests, "by_seed", lambda rows, method, table: {int(r["seed"]): float(r["seed"]) * 10 for r in rows})
    assert rtx_tests.values("A", "m", table=None) == [0.0, 10.0, 20.0]


def test_missing_arm_is_not_run_and_family_keeps_size(monkeypatch):
    fake = {"ABL_P_ref_local": [0.0] * 30, "ABL_P_anon_local": [0.1] * 29 + [0.2],
            "ABL_P_repair_local": [0.0] * 12, "ABL_P_05b_local": [0.0] * 30, "ABL_P_3b_local": [0.0] * 30}
    monkeypatch.setattr(rtx_tests, "values", lambda item, method, table, setting="ILI_T24": fake[item])
    monkeypatch.setattr(rtx_tests, "compare", lambda a, b, fam: {"p": 0.004, "family": fam})
    monkeypatch.setattr(rtx_tests, "spearman_trend", lambda levels: {"p": 0.5, "levels": sorted(levels)})
    out = rtx_tests.ab_p(None)
    assert out["not_run"] == ["repair vs ref"]
    assert set(out["tests"]) == {"anon vs ref", "scale trend"}
    assert out["tests"]["anon vs ref"]["family"] == 3
    assert out["tests"]["anon vs ref"]["p_holm"] == pytest.approx(0.012)  # 3 x 0.004: the missing test keeps size 3
    assert out["tests"]["scale trend"]["levels"] == [0.5, 1.5, 3.0]


def test_ab_g_levels_and_caveat(monkeypatch):
    fake = {"ABL_P_ref_local": list(range(30)), "ABL_G_steps30_local": [1.0] * 10, "ABL_G_ref_local": [2.0] * 10,
            "ABL_G_steps120_local": [3.0] * 10, "ABL_G_beta0_local": [1.0] * 10, "ABL_G_beta01_local": [1.0] * 9}
    seen = []
    monkeypatch.setattr(rtx_tests, "values", lambda item, method, table, setting="ILI_T24": fake[item])
    monkeypatch.setattr(rtx_tests, "spearman_trend", lambda levels: seen.append(sorted(levels)) or {"p": 0.02})
    monkeypatch.setattr(rtx_tests, "policy_level", lambda levels: {"dose": "x"})
    out = rtx_tests.ab_g(None)
    assert seen == [[0, 30, 60, 120]]  # KL trend not run: beta 0.1 has 9 policies
    assert out["not_run"] == ["KL trend"]
    assert "dependent" in out["tests"]["steps trend"]["caveat"]
    assert out["tests"]["steps trend"]["p_holm"] == pytest.approx(0.04)
    assert out["exploratory_addendum3"]["label"].startswith("exploratory")


def test_steps_trend_uses_untrained_reference_as_level_zero(monkeypatch):
    calls = {}
    monkeypatch.setattr(rtx_tests, "values", lambda item, method, table, setting="ILI_T24": calls.setdefault((item, method), [0.0] * (30 if item == "ABL_P_ref_local" else 10)))
    monkeypatch.setattr(rtx_tests, "spearman_trend", lambda levels: {"p": 1.0})
    monkeypatch.setattr(rtx_tests, "policy_level", lambda levels: {})
    rtx_tests.ab_g(None)
    assert ("ABL_P_ref_local", "qwen_prompted_1.5b_local") in calls
    assert all(m == rtx_tests.G for (i, m) in calls if i != "ABL_P_ref_local")


def test_values_refuses_seeds_other_than_0_to_n(tmp_path, monkeypatch):
    monkeypatch.setattr(rtx_tests, "RTX", tmp_path)
    _write(tmp_path / "A" / "ILI_T24", [{"seed": s, "method": "m"} for s in (0, 1, 3)])
    with pytest.raises(SystemExit):
        rtx_tests.values("A", "m", table=None)


def test_detectable_rho_and_caveats_on_every_trend(monkeypatch):
    assert 0 < rtx_tests.detectable_rho(90, 3) < rtx_tests.detectable_rho(30, 2) < 1
    fake = {k: [0.0] * 30 for k in ("ABL_P_ref_local", "ABL_P_anon_local", "ABL_P_repair_local", "ABL_P_05b_local", "ABL_P_3b_local")}
    fake.update({k: [0.0] * 10 for k in ("ABL_G_steps30_local", "ABL_G_ref_local", "ABL_G_steps120_local", "ABL_G_beta0_local", "ABL_G_beta01_local")})
    monkeypatch.setattr(rtx_tests, "values", lambda item, method, table, setting="ILI_T24": fake[item])
    monkeypatch.setattr(rtx_tests, "compare", lambda a, b, fam: {"p": 0.5})
    monkeypatch.setattr(rtx_tests, "spearman_trend", lambda levels: {"p": 0.5})
    monkeypatch.setattr(rtx_tests, "policy_level", lambda levels: {})
    p, g = rtx_tests.ab_p(None), rtx_tests.ab_g(None)
    for t in (p["tests"]["scale trend"], g["tests"]["steps trend"], g["tests"]["KL trend"]):
        assert "dependent" in t["caveat"] and "detectable_rho_80_approx" in t
    assert p["tests"]["scale trend"]["detectable_rho_80_approx"] == pytest.approx(rtx_tests.detectable_rho(90, 3))
    assert g["tests"]["steps trend"]["detectable_rho_80_approx"] == pytest.approx(rtx_tests.detectable_rho(60, 2))


def test_main_refuses_dirty_tree(monkeypatch):
    class R:
        stdout = " M scripts/ablations/rtx_tests.py\n"
    monkeypatch.setattr(rtx_tests.subprocess, "run", lambda *a, **k: R())
    with pytest.raises(SystemExit):
        rtx_tests.main()
