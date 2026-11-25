"""scripts/ablate_llm.py: flags reach the jobs, defaults equal the E4 recipe, item and method names (LEG2 L4)."""

import importlib.util
import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ablate_llm", ROOT / "scripts/ablate_llm.py")
ablate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ablate)
runner = ablate.runner

FAKE_ROWS = {"ILI": [{"config_id": "a"}], "ETTh1": [{"config_id": "b"}]}


@pytest.fixture(autouse=True)
def fake_tables(monkeypatch):
    monkeypatch.setattr(runner, "table_rows", lambda d: FAKE_ROWS[d])
    monkeypatch.setattr(runner, "noise_sd", lambda rows: 0.5 if rows is FAKE_ROWS["ETTh1"] else 0.0)


def plan(argv):
    return ablate.build(ablate.parser().parse_args(argv))


def test_transfer_defaults_equal_e4_recipe():
    p = plan(["transfer", "--source", "ETTh1", "--target", "ILI", "--item", "ABL_G_ref_local"])
    assert len(p["train_jobs"]) == 10 and len(p["search_jobs"]) == 10      # league:10 by default
    for job in p["train_jobs"]:
        assert job["steps"] == 60 and job["beta"] == 0.04 and job["lr"] == 1e-4
        assert job["episode_budget"] == 100 * 30 and job["rivals"] == "league"
        assert job["model"] == runner.MODEL and job["revision"] == runner.REVISION
        assert job["backend"] == "local" and job["per_gpu"] == 1
        assert job["noise_sd"] == 0.5 and job["dataset"] == "ETTh1"
        assert "/ABL_G_ref_local/adapters/" in job["out_dir"]
    assert [j["seed"] for j in p["train_jobs"]] == list(range(100, 110))
    for job in p["search_jobs"]:
        assert job["method"] == "grpo_league_from_ETTh1" and job["budget"] == 3000
        assert job["dataset"] == "ILI" and job["noise_sd"] == 0.0
    assert p["item"] == "ABL_G_ref_local" and p["name"] == "ILI_T24"


@pytest.mark.parametrize("beta", ["0", "0.1"])
def test_beta_reaches_every_training_job(beta):
    p = plan(["transfer", "--source", "ETTh1", "--target", "ILI", "--beta", beta, "--item", "ABL_G_x_local"])
    assert {j["beta"] for j in p["train_jobs"]} == {float(beta)}
    assert {j["steps"] for j in p["train_jobs"]} == {60}


def test_steps_arms_per_gpu_reach_jobs():
    p = plan(["transfer", "--source", "ILI", "--target", "ETTh1", "--steps", "30", "--arms", "none:3",
              "--per-gpu", "2", "--item", "ABL_G_y_local"])
    assert {j["steps"] for j in p["train_jobs"]} == {30}
    assert {j["rivals"] for j in p["train_jobs"]} == {"none"} and len(p["train_jobs"]) == 3
    assert p["per_gpu"] == 2 and {j["per_gpu"] for j in p["train_jobs"] + p["search_jobs"]} == {2}
    assert p["name"] == "ETTh1_T336"


@pytest.mark.parametrize("model,tag", [("Qwen/Qwen2.5-0.5B-Instruct", "0.5b"), ("Qwen/Qwen2.5-1.5B-Instruct", "1.5b"),
                                       ("Qwen/Qwen2.5-3B-Instruct", "3b")])
def test_prompted_model_revision_and_method(model, tag):
    p = plan(["prompted", "--dataset", "ILI", "--model", model, "--item", "ABL_P_x_local"])
    assert len(p["search_jobs"]) == 30 and p["train_jobs"] == []
    for s, job in enumerate(p["search_jobs"]):
        assert job["model"] == model and job["revision"] == ablate.PINNED[model]
        assert job["method"] == f"qwen_prompted_{tag}_local"
        assert job["seeds"] == [s] and job["budget"] == 3000 and job["constrained"] and not job["anonymous"]
        assert job["backend"] == "local" and job["per_gpu"] == 1
    assert p["item"] == "ABL_P_x_local" and p["name"] == "ILI_T24"


def test_prompted_defaults_are_the_e4_model():
    p = plan(["prompted", "--dataset", "ETTh1", "--item", "ABL_P_ref_local"])
    job = p["search_jobs"][0]
    assert job["model"] == runner.MODEL and job["revision"] == runner.REVISION
    assert job["method"] == "qwen_prompted_1.5b_local" and job["noise_sd"] == 0.5
    assert p["name"] == "ETTh1_T336"


def test_prompted_anonymous_and_unconstrained_flags():
    p = plan(["prompted", "--dataset", "ILI", "--seeds", "4", "--anonymous", "--no-constrained", "--item", "X"])
    assert len(p["search_jobs"]) == 4
    for job in p["search_jobs"]:
        assert job["anonymous"] and not job["constrained"]
        assert job["method"] == "qwen_prompted_anon_repair_1.5b_local"


def test_revision_must_equal_pin():
    with pytest.raises(SystemExit):
        plan(["prompted", "--dataset", "ILI", "--model", "Qwen/Qwen2.5-3B-Instruct", "--revision", "main",
              "--item", "X"])


def test_transfer_refuses_other_model():
    with pytest.raises(SystemExit):
        plan(["transfer", "--source", "ETTh1", "--target", "ILI", "--model", "Qwen/Qwen2.5-3B-Instruct",
              "--item", "X"])


def test_item_is_required():
    with pytest.raises(SystemExit):
        ablate.parser().parse_args(["prompted", "--dataset", "ILI"])


def test_reuses_frozen_executor_and_gpu_check():
    src = inspect.getsource(ablate)
    assert "llm_local.execute" in src and "llm_local.check_gpu" in src
    assert "def execute" not in src and "def check_gpu" not in src
