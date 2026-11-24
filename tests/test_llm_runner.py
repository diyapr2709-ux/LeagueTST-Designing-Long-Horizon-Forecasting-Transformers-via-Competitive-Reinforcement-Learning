"""The shared LLM driver loop: arm interleaving and the rules that make relaunches safe."""

import json
from pathlib import Path

from leaguetst.architect import runner


def plan(tmp_path, arms="league:2,none:2,uniform:1", per_gpu=2):
    rows = [{"x": 1}]
    return runner.transfer_plan("ILI", "ETTh1", arms, runs_root=str(tmp_path), per_gpu=per_gpu, backend="test",
                                rows_source=rows, rows_target=rows, noise_source=0.0, noise_target=0.0)


def fake_execute(calls, fail_train=()):
    def execute(kind, batches):
        for batch in batches:
            calls.append((kind, [j.get("run_name") or f"{j['method']}:{j['seeds'][0]}" for j in batch]))
            if kind == "train":
                results = []
                for j in batch:
                    if j["run_name"] in fail_train:
                        results.append({"error": "RuntimeError('sampled and scored log-probabilities differ by 0.25')",
                                        "job": j})
                        continue
                    saved = Path(j["out_dir"]) / "adapter"
                    saved.mkdir(parents=True, exist_ok=True)
                    (saved / "adapter_config.json").write_text("{}")
                    results.append({"adapter": str(saved), "job": j})
                yield results
            else:
                yield [[{"method": j["method"], "seed": j["seeds"][0]}] for j in batch]
    return execute


def test_arms_are_interleaved_and_jobs_stamped(tmp_path):
    p = plan(tmp_path)
    names = [j["run_name"] for j in p["train_jobs"]]
    assert names[:3] == ["grpo_league_ILI_seed100", "grpo_none_ILI_seed100", "grpo_uniform_ILI_seed100"]
    assert len(names) == 5 and all(j["revision"] == runner.REVISION and j["per_gpu"] == 2 for j in p["train_jobs"])
    assert {j["method"] for j in p["search_jobs"]} == {"grpo_league_from_ILI", "grpo_none_from_ILI", "grpo_uniform_from_ILI"}


def test_batches_follow_per_gpu(tmp_path):
    calls = []
    runner.run_plan(plan(tmp_path, per_gpu=2), tmp_path / "out", fake_execute(calls))
    assert [len(names) for kind, names in calls if kind == "train"] == [2, 2, 1]
    assert sum(len(names) for kind, names in calls if kind == "search") == 5


def test_relaunch_keeps_adapters_and_deployed_seeds(tmp_path):
    p = plan(tmp_path)
    saved = tmp_path / "E4" / "adapters" / "grpo_league_ILI_seed100" / "adapter"
    saved.mkdir(parents=True)
    (saved / "adapter_config.json").write_text("{}")
    out = tmp_path / "out"
    out.mkdir()
    (out / "grpo_none_from_ILI_runs.jsonl").write_text(json.dumps({"method": "grpo_none_from_ILI", "seed": 0}) + "\n")
    calls = []
    runner.run_plan(p, out, fake_execute(calls))
    trained = [n for kind, names in calls if kind == "train" for n in names]
    searched = [n for kind, names in calls if kind == "search" for n in names]
    assert "grpo_league_ILI_seed100" not in trained and "grpo_league_from_ILI:0" in searched
    assert "grpo_none_from_ILI:0" not in searched


def test_on_policy_stops_are_never_retrained_or_deployed(tmp_path):
    out = tmp_path / "out"
    calls = []
    runner.run_plan(plan(tmp_path), out, fake_execute(calls, fail_train={"grpo_none_ILI_seed101"}))
    assert runner.stopped_runs(out) == {"grpo_none_ILI_seed101"}
    assert "grpo_none_from_ILI:1" not in [n for kind, names in calls if kind == "search" for n in names]
    calls.clear()
    runner.run_plan(plan(tmp_path), out, fake_execute(calls))
    assert [n for kind, names in calls if kind == "train" for n in names] == []


def test_batch_errors_name_their_runs(tmp_path):
    def execute(kind, batches):
        for _ in batches:
            yield RuntimeError("container lost")
    out = tmp_path / "out"
    runner.run_plan(plan(tmp_path, arms="league:1,none:1"), out, execute)
    errors = [json.loads(line) for line in (out / "errors.jsonl").read_text().splitlines()]
    assert errors[0]["run_names"] == ["grpo_league_ILI_seed100", "grpo_none_ILI_seed100"]
