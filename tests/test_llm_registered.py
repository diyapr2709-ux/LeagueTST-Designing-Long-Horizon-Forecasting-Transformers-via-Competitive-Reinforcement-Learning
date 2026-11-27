"""scripts/ablations/llm_registered.py on synthetic record files (AB4): completeness, settings, guards, families."""

from __future__ import annotations

import functools
import json
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ablations"))

import llm_registered as reg  # noqa: E402
import llm_tests  # noqa: E402

from leaguetst.architect import runner  # noqa: E402

STOP = "RuntimeError('sampled and scored log-probabilities differ by 0.061 per token on average')"


def env(revision: str, **over) -> dict:
    return {"backend": "local", "per_gpu": 1, "model_revision": revision, "git_hash": "abc123", "gpu_name": reg.CARD,
            **over}


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def fake(item: str, seed: int, shift: float = 0.0) -> float:
    return float(np.random.default_rng(zlib.crc32(f"{item} {seed}".encode())).normal(shift, 1))


def prompted(root: Path, item: str, table: str, seeds=range(30), shift: float = 0.0, **env_over) -> None:
    method, revision, _ = reg.PROMPTED[item]
    write(root / item / table / f"{method}_runs.jsonl",
          [{"seed": s, "method": method, "env": env(revision, **env_over), "fake": fake(item + table, s, shift)}
           for s in seeds])


def grpo(root: Path, item: str, steps=None, beta=None, stopped=(), deployed=range(10), shift: float = 0.0) -> None:
    reg_steps, reg_beta, _ = reg.GRPO[item]
    steps, beta = reg_steps if steps is None else steps, reg_beta if beta is None else beta
    folder = root / item / "ILI_T24"
    names = list(reg.POLICY_NAMES)
    write(folder / "training.jsonl", [{"job": {"run_name": n, "steps": steps, "beta": beta, "lr": 1e-4, "rivals": "league",
                                               "dataset": "ETTh1"}, "env": env(runner.REVISION), "log": []}
                                      for n in names if n not in stopped])
    write(folder / f"{reg.GRPO_METHOD}_runs.jsonl",
          [{"seed": s, "method": reg.GRPO_METHOD, "env": env(runner.REVISION), "fake": fake(item, s, shift)}
           for s in deployed if names[s] not in stopped])
    rows = []
    for n in stopped:  # what the runner writes for a stopped policy: the stop, then "no adapter to deploy"
        rows += [{"phase": "train", "run_names": [n], "error": STOP},
                 {"phase": "search", "error": "no adapter to deploy", "run_names": [f"{reg.GRPO_METHOD}:{reg.POLICY_NAMES[n]}"]}]
    if rows:
        write(folder / "errors.jsonl", rows)


@pytest.fixture
def rtx(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "RTX", tmp_path)
    monkeypatch.setattr(reg, "RESULT", tmp_path / "LLM-registered.json")
    monkeypatch.setattr(reg, "by_seed", lambda runs, method, table, metric="auc":
                        {int(r["seed"]): r["fake"] for r in runs if r["method"] == method})
    monkeypatch.setattr(reg, "spearman_trend", functools.partial(llm_tests.spearman_trend, n_perm=200))
    monkeypatch.setattr(reg, "commit_problem", lambda git_hash, ab4: None)
    monkeypatch.setattr(reg, "manifest_problems", lambda: [])
    return tmp_path


def everything(root: Path) -> None:
    for item in reg.PROMPTED:
        for table in reg.TABLES.values():
            prompted(root, item, table)
    for item in reg.GRPO:
        grpo(root, item)


def run_with_runner(root: Path, item: str, fail: dict[int, str], relaunch: bool) -> Path:
    """Build the item folder with the real runner.run_plan and a fake executor; fail maps training seed to an error."""
    steps, beta, _ = reg.GRPO[item]
    plan = runner.transfer_plan("ETTh1", "ILI", "league:10", runs_root=str(root / "adapters"), per_gpu=1, backend="local",
                                rows_source=[], rows_target=[], noise_source=0.0, noise_target=0.0, steps=steps,
                                beta=beta, item=item)
    out = root / item / "ILI_T24"

    def execute(kind, batches, failing=dict(fail)):
        for batch in batches:
            result = []
            for job in batch:
                if kind == "train":
                    if job["seed"] in failing:
                        result.append({"error": failing[job["seed"]], "job": job})
                        continue
                    saved = Path(job["out_dir"])
                    (saved / "adapter").mkdir(parents=True, exist_ok=True)
                    (saved / "adapter" / "adapter_config.json").write_text("{}")
                    (saved / "provenance.json").write_text(json.dumps(runner.provenance(job)))
                    result.append({"job": job, "adapter": str(saved / "adapter"), "env": env(runner.REVISION), "log": []})
                else:
                    s = job["seeds"][0]
                    result.append([{"seed": s, "method": job["method"], "env": env(runner.REVISION), "fake": fake(item, s)}])
            yield result

    runner.run_plan(plan, out, execute)
    if relaunch:  # the crash is fixed on relaunch; an on-policy stop is never retrained
        runner.run_plan(plan, out, lambda kind, batches: execute(kind, batches, failing={}))
    return out


def test_prompted_completeness(rtx):
    prompted(rtx, "ABL_P_anon_local", "ILI_T24")
    prompted(rtx, "ABL_P_repair_local", "ILI_T24", seeds=range(29))
    assert reg.prompted_arm("ABL_P_anon_local", "ILI")["complete"]
    short = reg.prompted_arm("ABL_P_repair_local", "ILI")
    assert not short["complete"] and short["seeds"] == 29
    assert not reg.prompted_arm("ABL_P_3b_local", "ILI")["complete"]  # no file at all


@pytest.mark.parametrize("over", [{"backend": "modal"}, {"per_gpu": 2}, {"model_revision": "other"},
                                  {"gpu_name": "Other GPU"}])
def test_prompted_settings_are_checked(rtx, over):
    prompted(rtx, "ABL_P_anon_local", "ILI_T24", **over)
    arm = reg.prompted_arm("ABL_P_anon_local", "ILI")
    assert not arm["complete"] and arm["problems"]


def test_commit_problems_make_an_arm_incomplete(rtx, monkeypatch):
    prompted(rtx, "ABL_P_anon_local", "ILI_T24")
    monkeypatch.setattr(reg, "commit_problem", lambda git_hash, ab4: "commit abc123 predates AB4" if ab4 else None)
    assert not reg.prompted_arm("ABL_P_anon_local", "ILI")["complete"]


def test_commit_problem_on_real_history():
    reg.commit_problem.cache_clear()
    assert reg.commit_problem("abc-dirty", False).startswith("dirty")
    assert reg.commit_problem("0" * 40, False).startswith("unknown")
    assert reg.commit_problem(reg.AB2_COMMIT, False) is None  # AB2's own commit descends from itself
    assert "predates AB4" in reg.commit_problem(reg.AB2_COMMIT, True)


def test_a_truncated_line_is_a_problem_not_a_crash(rtx):
    prompted(rtx, "ABL_P_anon_local", "ILI_T24")
    path = rtx / "ABL_P_anon_local" / "ILI_T24" / f"{reg.PROMPTED['ABL_P_anon_local'][0]}_runs.jsonl"
    path.write_text(path.read_text() + '{"seed": 3')
    arm = reg.prompted_arm("ABL_P_anon_local", "ILI")
    assert not arm["complete"] and any("does not parse" in p for p in arm["problems"])


def test_grpo_settings_are_checked(rtx):
    grpo(rtx, "ABL_G_steps30_local", steps=60)  # trained with the reference's steps by mistake
    arm = reg.grpo_arm("ABL_G_steps30_local")
    assert not arm["complete"] and any("steps" in p for p in arm["problems"])


def test_stop_written_by_the_runner_leaves_the_arm_complete(rtx):
    run_with_runner(rtx, "ABL_G_steps120_local", {103: STOP}, relaunch=True)
    arm = reg.grpo_arm("ABL_G_steps120_local")
    assert arm["complete"], arm["problems"]
    assert arm["stopped"] == ["grpo_league_ETTh1_seed103"] and arm["deployed"] == 9


def test_crash_fixed_by_a_relaunch_leaves_the_arm_complete(rtx):
    run_with_runner(rtx, "ABL_G_steps30_local", {105: "RuntimeError('CUDA out of memory')"}, relaunch=True)
    arm = reg.grpo_arm("ABL_G_steps30_local")
    assert arm["complete"], arm["problems"]
    assert arm["deployed"] == 10 and arm["stopped"] == []


def test_crash_not_relaunched_is_incomplete(rtx):
    run_with_runner(rtx, "ABL_G_steps30_local", {105: "RuntimeError('CUDA out of memory')"}, relaunch=False)
    arm = reg.grpo_arm("ABL_G_steps30_local")
    assert not arm["complete"] and arm["missing_outcome"] == ["grpo_league_ETTh1_seed105"]


def test_grpo_missing_deployment_is_incomplete(rtx):
    grpo(rtx, "ABL_G_steps30_local", deployed=range(9))
    assert reg.grpo_arm("ABL_G_steps30_local")["undeployed"] == ["grpo_league_ETTh1_seed109"]


def test_retried_policies_stay_in_the_tests(rtx):
    everything(rtx)
    (rtx / "ABL_G_beta0_local" / "ILI_T24" / "retried.txt").write_text("grpo_league_ETTh1_seed101\n")
    arm = reg.grpo_arm("ABL_G_beta0_local")
    assert arm["complete"] and arm["retried"] == ["grpo_league_ETTh1_seed101"]
    kl = reg.families(reg.arms(), None)["AB-G"]["tests"]["KL trend"]
    assert kl["n"] == 30  # kept as they finished (AB1, AB2, A29)
    assert kl["filled_policies"] == {"0.0": [1]} and {"fill_lowest", "fill_highest"} <= set(kl)


def test_unknown_retried_name_is_a_problem(rtx):
    grpo(rtx, "ABL_G_steps30_local")
    (rtx / "ABL_G_steps30_local" / "ILI_T24" / "retried.txt").write_text("grpo_league_ILI_seed101\n")
    assert not reg.grpo_arm("ABL_G_steps30_local")["complete"]


def test_a_card_mismatch_affects_only_that_arm(rtx):
    everything(rtx)
    prompted(rtx, "ABL_P_anon_local", "ETTh1_T336", gpu_name="Other GPU")
    all_arms = reg.arms()
    assert not all_arms["ABL_P_anon_local ETTh1"]["complete"]
    assert all(a["complete"] for n, a in all_arms.items() if n != "ABL_P_anon_local ETTh1")


def test_manifest_problem_voids_only_ab2_arms(rtx, monkeypatch):
    everything(rtx)
    monkeypatch.setattr(reg, "manifest_problems", lambda: ["x differs"])
    all_arms = reg.arms()
    assert not all_arms["ABL_G_ref_local"]["complete"] and not all_arms["ABL_P_ref_local ILI"]["complete"]
    assert all_arms["ABL_G_steps30_local"]["complete"] and all_arms["ABL_P_anon_local ILI"]["complete"]


def test_manifest_matches_the_committed_ab2_records():
    assert reg.manifest_problems() == []


def test_refuses_while_incomplete_and_writes_nothing(rtx):
    everything(rtx)
    (rtx / "ABL_G_steps120_local" / "ILI_T24" / "training.jsonl").unlink()
    with pytest.raises(SystemExit, match="incomplete"):
        reg.main([])
    assert not reg.RESULT.exists()


def test_final_refuses_before_the_deadline(rtx, monkeypatch):
    class Before(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 28, 22, 59, tzinfo=timezone.utc)

    monkeypatch.setattr(reg, "datetime", Before)
    with pytest.raises(SystemExit, match="only after 28 Nov 18:00 ET"):
        reg.preflight(final=True)


def test_computed_once(rtx):
    reg.RESULT.write_text("{}")
    with pytest.raises(SystemExit, match="computed once"):
        reg.preflight(final=False)


def test_registered_digests_match_the_shared_code():
    for path, digest in reg.DIGESTS.items():
        assert reg.sha256(ROOT / path) == digest, path


def test_check_prints_counts_not_values(rtx, capsys):
    everything(rtx)
    assert reg.main(["--check"]) == 0
    out = capsys.readouterr().out
    assert "30/30 seeds" in out and "auc" not in out.lower() and "rho" not in out
    assert not reg.RESULT.exists()


def test_families_with_a_missing_arm_keep_their_size(rtx):
    everything(rtx)
    (rtx / "ABL_P_repair_local" / "ILI_T24" / f"{reg.PROMPTED['ABL_P_repair_local'][0]}_runs.jsonl").unlink()
    (rtx / "ABL_G_steps30_local" / "ILI_T24" / "training.jsonl").unlink()
    result = reg.families(reg.arms(), None)
    assert set(result["AB-P"]["tests"]) == {"anon vs ref", "scale trend"}
    assert set(result["AB-G"]["tests"]) == {"KL trend"}
    assert {"AB-P repair vs ref", "AB-G step trend"} <= set(result["not_run"])
    assert result["AB-P"]["family_size"] == 3 and result["AB-G"]["family_size"] == 2
    for test in [*result["AB-P"]["tests"].values(), *result["AB-G"]["tests"].values()]:
        assert test["p_holm"] >= test["p"]


def test_a_real_effect_is_found(rtx):
    everything(rtx)
    prompted(rtx, "ABL_P_anon_local", "ILI_T24", shift=3.0)  # anon much worse than ref
    assert reg.families(reg.arms(), None)["AB-P"]["tests"]["anon vs ref"]["reject"]


def test_step_trend_with_a_stop_and_its_secondary_analyses(rtx):
    everything(rtx)
    grpo(rtx, "ABL_G_steps120_local", stopped=("grpo_league_ETTh1_seed100",))
    result = reg.families(reg.arms(), None)
    step = result["AB-G"]["tests"]["step trend"]
    assert step["n"] == 30 + 10 + 10 + 9 and {"fill_lowest", "fill_highest"} <= set(step)
    secondary = result["secondary_no_holm"]["step trend"]
    assert secondary["ii_per_policy_rho"]["policies"] == 9
    assert 0 < secondary["ii_per_policy_rho"]["p_sign_flip"] <= 1
    assert secondary["i_trained_vs_untrained"]["n"] == [10, 30]


def test_a_level_with_every_policy_stopped_is_not_tested(rtx):
    everything(rtx)
    grpo(rtx, "ABL_G_beta01_local", stopped=tuple(reg.POLICY_NAMES))
    result = reg.families(reg.arms(), None)
    assert "KL trend" not in result["AB-G"]["tests"]
    assert any(n.startswith("AB-G KL trend") for n in result["not_run"])


def test_sign_flip_is_exact():
    assert reg.sign_flip([1.0] * 10) == pytest.approx(2 / 1024)
    assert reg.sign_flip([1.0, -1.0]) == 1.0


def test_reads_no_results_search_folder_for_arms():
    source = Path(reg.__file__).read_text()
    assert 'OUT / "rtx"' in source and "results/search/ABL" not in source
