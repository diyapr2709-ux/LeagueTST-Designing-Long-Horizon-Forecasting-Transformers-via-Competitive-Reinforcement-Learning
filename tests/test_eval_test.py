"""scripts/eval_test.py end to end on synthetic data: preflight, scoring, analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from leaguetst.data.trainval import make_trainval
from leaguetst.table import patchtst42_config, recipe, run_entries

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="module")
def setup(synthetic_data_dir, tmp_path_factory):
    root = tmp_path_factory.mktemp("evalroot")
    trainval = root / "trainval"
    make_trainval(synthetic_data_dir, trainval)
    small = {**patchtst42_config("ILI"), "n_layers": 1}
    arms = {"a": patchtst42_config("ILI"), "b": small}
    prereg = {"id": "synthetic_e2e", "arms": [], "comparisons": [{"name": "a vs b", "a": "a", "b": "b"}],
              "rank_checks": [{"name": "all", "arms": ["a", "b"]}], "bootstrap": {"resamples": 200, "seed": 0}}
    for name, config in arms.items():
        runs = []
        for seed in (1, 2, 3, 4, 5):
            job = {"item": "T", "dataset": "ILI", "horizon": 24, "seed": seed, "trainval_dir": str(trainval),
                   "training": recipe("ILI", epochs=1, patience=10), "git_hash": "test"}
            out = root / "runs" / f"{name}_s{seed}"
            run_entries(job, [(name, config)], out, "cpu")
            record = json.loads((out / "records" / f"{name}.json").read_text())
            runs.append({"horizon": 24, "seed": seed, "checkpoint": str((out / "ckpt" / f"{name}.pt").relative_to(root)),
                         "sha256": record["checkpoint_sha256"], "val_mse": record["val"]["mse"]})
        prereg["arms"].append({"name": name, "dataset": "ILI", "model_config": config, "runs": runs})
    return root, trainval, prereg


@pytest.fixture
def eval_test(monkeypatch, tmp_path):
    import eval_test as module
    from leaguetst.data import test_split

    monkeypatch.setattr(module, "g3_problems", lambda root: ([], ""))
    monkeypatch.setattr(module, "ACCESS_LOG", tmp_path / "access.log")
    monkeypatch.setattr(test_split, "ACCESS_LOG", tmp_path / "access.log")
    monkeypatch.setattr(test_split, "is_registered", lambda prereg_id: prereg_id == "synthetic_e2e")
    return module


def test_preflight_passes_and_catches_a_wrong_config(setup, eval_test, tmp_path):
    root, trainval, prereg = setup
    assert eval_test.preflight(prereg, root, trainval) == []
    wrong = json.loads(json.dumps(prereg))
    wrong["arms"][0]["model_config"]["n_heads"] = 2
    problems = eval_test.preflight(wrong, root, trainval)
    assert any("does not reproduce" in p for p in problems)
    assert not (tmp_path / "access.log").exists()


def test_score_save_and_analyze(setup, eval_test, synthetic_data_dir, tmp_path):
    root, trainval, prereg = setup
    save_dir = tmp_path / "saved"
    eval_test.score_runs(prereg, synthetic_data_dir, trainval, root, save_dir)
    log_lines = (tmp_path / "access.log").read_text().splitlines()
    assert len(log_lines) == 10
    saved = sorted(p.name for p in save_dir.glob("*.npy"))
    assert len(saved) == 10
    # A second call scores nothing new and opens nothing.
    eval_test.score_runs(prereg, synthetic_data_dir, trainval, root, save_dir)
    assert len((tmp_path / "access.log").read_text().splitlines()) == 10
    result = eval_test.analyze(prereg, save_dir)
    comparison = result["comparisons"]["a vs b"]
    assert 0 < comparison["endpoint_p"] <= 1 and np.isfinite(comparison["endpoint_diff"])
    assert result["rank_checks"]["all"]["n"] == 10
    eval_test.write_json(tmp_path / "out.json", result)
    assert json.loads((tmp_path / "out.json").read_text())["prereg"] == "synthetic_e2e"


def test_analyze_refuses_missing_seeds(setup, eval_test, synthetic_data_dir, tmp_path):
    root, trainval, prereg = setup
    save_dir = tmp_path / "saved"
    eval_test.score_runs(prereg, synthetic_data_dir, trainval, root, save_dir)
    (save_dir / "b_h24_s5.json").unlink()
    with pytest.raises(FileNotFoundError):
        eval_test.analyze(prereg, save_dir)
