"""E0: test lock, trainval loader, feasible set, NumPy re-scorer, ledger, stats."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from leaguetst.data import build_dataloaders, build_splits
from leaguetst.data.device_loader import build_trainval_loaders
from leaguetst.data.test_split import load_test_split
from leaguetst.data import trainval as trainval_module
from leaguetst.data.trainval import build_trainval_splits, read_manifest, window_counts
from leaguetst.eval.numpy_model import numpy_patchtst_forward
from leaguetst.eval.rescore import rescore_checkpoint
from leaguetst.eval.stats import holm, paired_two_level_bootstrap
from leaguetst.ledger import append_rows, read_rows, row_from_record
from leaguetst.models import ForecasterConfig, PatchTST, count_parameters
from leaguetst.search import feasible_all, feasible_table, patchtst_parameters
from leaguetst.search.space import enumerate_configs
from leaguetst.table import recipe, run_entries

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_manifest_borders_come_from_the_full_file(trainval_dir):
    manifest = read_manifest(trainval_dir)["datasets"]
    assert (manifest["ETTh1"]["train_end"], manifest["ETTh1"]["val_end"]) == (8640, 11520)
    # ILI borders are ratios of the full length (966 rows), not of the copy.
    assert (manifest["ILI"]["train_end"], manifest["ILI"]["val_end"], manifest["ILI"]["source_rows"]) == (676, 773, 966)
    assert manifest["ILI"]["rows"] == 773


@pytest.mark.parametrize("name,input_len,horizon", [("ETTh1", 336, 96), ("ETTh1", 336, 720), ("ILI", 104, 24), ("ILI", 104, 60)])
def test_trainval_equals_full_pipeline(synthetic_data_dir, trainval_dir, name, input_len, horizon):
    full, full_scaler = build_splits(synthetic_data_dir, name, input_len, horizon)
    copy, copy_scaler = build_trainval_splits(trainval_dir, name, input_len, horizon)
    for split in ("train", "val"):
        assert torch.equal(full[split].values, copy[split].values)
    assert np.array_equal(full_scaler.mean, copy_scaler.mean) and np.array_equal(full_scaler.scale, copy_scaler.scale)
    counts = window_counts(read_manifest(trainval_dir)["datasets"][name], input_len, horizon)
    assert counts == {"train": len(copy["train"]), "val": len(copy["val"])}


def test_ili_validation_window_count(trainval_dir):
    counts = window_counts(read_manifest(trainval_dir)["datasets"]["ILI"], 104, 24)
    assert counts["val"] == 74


def test_test_split_is_locked(trainval_dir):
    with pytest.raises(trainval_module.TestSplitLocked):
        build_trainval_splits(trainval_dir, "ILI", 104, 24, splits=("train", "test"))


def test_tampered_trainval_file_is_refused(trainval_dir, tmp_path):
    import shutil

    copy = tmp_path / "tv"
    shutil.copytree(trainval_dir, copy)
    path = copy / "illness" / "national_illness.csv"
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:-1]))
    with pytest.raises(ValueError):
        build_trainval_splits(copy, "ILI", 104, 24)


@pytest.mark.parametrize("name,input_len,horizon,batch", [("ETTh1", 336, 96, 128), ("ILI", 104, 24, 16)])
def test_device_loader_matches_standard_batches(synthetic_data_dir, trainval_dir, name, input_len, horizon, batch):
    standard, _ = build_dataloaders(synthetic_data_dir, name, input_len, horizon, batch, seed=7)
    device, _ = build_trainval_loaders(trainval_dir, name, input_len, horizon, batch, 7, torch.device("cpu"))
    for _ in range(2):
        for split in ("train", "val"):
            a, b = list(standard[split]), list(device[split])
            assert len(a) == len(b)
            assert all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) for x, y in zip(a, b))


def test_feasible_counts_match_the_report():
    table = feasible_table()
    got = [table["per_setting"][k]["feasible"] for k in ("ETTh1_T96", "ETTh1_T192", "ETTh1_T336", "ETTh1_T720")]
    assert got == [2720, 2752, 2656, 2464]
    ili = [table["per_setting"][k] for k in ("ILI_T24", "ILI_T36", "ILI_T48", "ILI_T60")]
    assert [r["feasible"] for r in ili] == [6144, 6016, 6048, 6048]
    assert [r["independent"] for r in ili] == [4800, 4960, 5248, 5376]
    assert table["all_settings"] == len(feasible_all()) == 2368
    assert all(c.channel_mode == "independent" and c.d_model <= 32 for c in feasible_all())


def test_closed_form_parameter_count_matches_models():
    rng = random.Random(3)
    for config in rng.sample(list(enumerate_configs()), 40):
        for input_len, horizon in ((336, 96), (104, 60)):
            assert patchtst_parameters(config, input_len, horizon) == count_parameters(PatchTST(config, 7, input_len, horizon))


def _randomised_model(config: ForecasterConfig, input_len: int, horizon: int, seed: int) -> PatchTST:
    torch.manual_seed(seed)
    model = PatchTST(config, 7, input_len, horizon)
    for name, buffer in model.named_buffers():
        if name.endswith("running_mean"):
            buffer.normal_(0, 0.3)
        elif name.endswith("running_var"):
            buffer.uniform_(0.5, 2.0)
    for name, param in model.named_parameters():
        if "norm" in name:
            param.data.normal_(1.0 if name.endswith("weight") else 0.0, 0.2)
    return model.eval()


@pytest.mark.parametrize(
    "fields",
    [
        dict(patch_len=16, stride=8, n_layers=3, n_heads=4, d_model=16, ff_ratio=8, dropout=0.3),
        dict(patch_len=12, stride=12, n_layers=2, n_heads=2, d_model=32, ff_ratio=2, dropout=0.1, instance_norm=False),
        dict(patch_len=8, stride=4, n_layers=1, n_heads=8, d_model=16, ff_ratio=4, dropout=0.0, channel_mode="mixing"),
    ],
)
def test_numpy_forward_matches_torch(fields):
    config = ForecasterConfig(**fields)
    model = _randomised_model(config, 104, 24, seed=1)
    inputs = torch.randn(5, 104, 7) * 3 + 1
    with torch.no_grad():
        expected = model(inputs).double().numpy()
    state = {k: v.double().numpy() for k, v in model.state_dict().items()}
    got = numpy_patchtst_forward(state, config.to_dict(), inputs.double().numpy(), 24)
    assert np.allclose(got, expected, rtol=1e-4, atol=1e-5)


def test_table_entry_rescore_and_ledger(trainval_dir, tmp_path):
    from leaguetst.search import configs_by_id

    key, config = next(iter(configs_by_id().items()))
    job = {"item": "T", "dataset": "ILI", "horizon": 24, "seed": 3, "trainval_dir": str(trainval_dir),
           "training": recipe("ILI", epochs=2, patience=10), "git_hash": "test"}
    out = tmp_path / "run"
    assert run_entries(job, [(key, config.to_dict())], out, "cpu") == {"done": 1, "skipped": 0, "failed": 0}
    assert run_entries(job, [(key, config.to_dict())], out, "cpu")["skipped"] == 1
    record = json.loads((out / "records" / f"{key}.json").read_text())
    assert record["epochs_run"] == 2 and len(record["history"]) == 2
    errors = np.load(out / "perwindow" / f"{key}.npy")
    assert len(errors) == 74 and abs(errors.mean() - record["val"]["mse"]) < 1e-6
    rescored = rescore_checkpoint(record, out / "ckpt" / f"{key}.pt", trainval_dir)
    assert rescored["abs_gap_mse"] < 1e-6

    ledger = tmp_path / "ledger.jsonl"
    row = row_from_record(record, "test")
    assert append_rows([row, row], ledger) == {"appended": 1, "skipped": 1}
    changed = row_from_record({**record, "val": {**record["val"], "mse": record["val"]["mse"] + 1}}, "test")
    with pytest.raises(ValueError):
        append_rows([changed], ledger)
    assert len(list(read_rows(ledger))) == 1


def test_holm_and_paired_bootstrap():
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": True, "b": False, "c": False}
    assert holm({"a": 0.01, "b": 0.02}) == {"a": True, "b": True}
    rng = np.random.default_rng(0)
    base = rng.gamma(2.0, 0.2, size=(1, 400))
    same = paired_two_level_bootstrap(base + rng.normal(0, 0.01, (5, 400)), base + rng.normal(0, 0.01, (5, 400)), block=24, n_resamples=500)
    assert same["low"] < 0 < same["high"]
    lower = paired_two_level_bootstrap(base * 0.9 + rng.normal(0, 0.01, (5, 400)), base + rng.normal(0, 0.01, (5, 400)), block=24, n_resamples=500)
    assert lower["high"] < 0 and lower["p"] < 0.05


def test_test_split_logs_every_access(synthetic_data_dir, trainval_dir, tmp_path, monkeypatch):
    from leaguetst.data import test_split

    log = tmp_path / "access.log"
    monkeypatch.setattr(test_split, "ACCESS_LOG", log)
    with pytest.raises(ValueError):
        load_test_split(synthetic_data_dir, trainval_dir, "ILI", 104, 24, "")
    with pytest.raises(ValueError):
        load_test_split(synthetic_data_dir, trainval_dir, "ILI", 104, 24, "not_registered_anywhere")
    assert not log.exists()
    monkeypatch.setattr(test_split, "is_registered", lambda prereg_id: prereg_id == "unit")
    test, _ = load_test_split(synthetic_data_dir, trainval_dir, "ILI", 104, 24, "unit")
    assert len(test) == 193 - 24 + 1
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert lines[-1]["prereg"] == "unit" and lines[-1]["dataset"] == "ILI"
    full, _ = build_splits(synthetic_data_dir, "ILI", 104, 24)
    assert torch.equal(full["test"].values, test.values)


def test_test_split_refuses_a_different_file(synthetic_data_dir, trainval_dir, tmp_path, monkeypatch):
    import shutil

    from leaguetst.data import test_split

    monkeypatch.setattr(test_split, "ACCESS_LOG", tmp_path / "log")
    monkeypatch.setattr(test_split, "is_registered", lambda prereg_id: True)
    vault = tmp_path / "vault"
    shutil.copytree(synthetic_data_dir, vault)
    path = vault / "illness" / "national_illness.csv"
    text = path.read_text().splitlines()
    path.write_text("\n".join(text[:-1]) + "\n")
    with pytest.raises(ValueError):
        load_test_split(vault, trainval_dir, "ILI", 104, 24, "unit")


def test_prereg_refuses_bad_specs(tmp_path):
    import prereg

    import hashlib

    runs = []
    for s in range(5):
        (tmp_path / f"a{s}.pt").write_bytes(f"weights {s}".encode())
        runs.append({"horizon": 24, "seed": s, "checkpoint": f"a{s}.pt", "sha256": hashlib.sha256(f"weights {s}".encode()).hexdigest(), "val_mse": 0.5})
    spec = {"id": "unit_1", "arms": [{"name": "x", "dataset": "ILI", "runs": runs}, {"name": "y", "dataset": "ILI", "runs": runs[:3]}],
            "comparisons": [{"name": "x vs y", "a": "x", "b": "y"}]}
    problems = prereg.validate(spec, root=tmp_path, access_log=tmp_path / "log")
    assert any("fewer than 5 distinct seeds" in p for p in problems)
    spec["arms"][1]["runs"] = runs
    assert prereg.validate(spec, root=tmp_path, access_log=tmp_path / "log") == []
    (tmp_path / "log").write_text(json.dumps({"prereg": "unit_1"}) + "\n")
    assert any("already opened" in p for p in prereg.validate(spec, root=tmp_path, access_log=tmp_path / "log"))
    spec["arms"][0]["runs"] = [dict(runs[0], seed=s) for s in range(5)]
    assert any("repeats a checkpoint" in p for p in prereg.validate(spec, root=tmp_path, access_log=tmp_path / "log2"))
    spec["arms"][0]["runs"] = [dict(runs[0], sha256="0" * 64)] + runs[1:]
    assert any("digest" in p for p in prereg.validate(spec, root=tmp_path, access_log=tmp_path / "log2"))
    other = [dict(r, horizon=36, seed=r["seed"] + 5) for r in runs]
    spec["arms"][0]["runs"] = runs + other
    assert any("same seeds at every horizon" in p for p in prereg.validate(spec, root=tmp_path, access_log=tmp_path / "log2"))


def test_build_splits_refuses_a_trainval_copy(trainval_dir):
    with pytest.raises(ValueError):
        build_splits(trainval_dir, "ILI", 104, 24)


def test_permutation_test_and_endpoint_bootstrap():
    from leaguetst.eval.stats import endpoint_bootstrap, permutation_test

    rng = np.random.default_rng(1)
    same = permutation_test(rng.normal(0, 1, 10), rng.normal(0, 1, 10))
    assert 0 < same["p"] <= 1
    apart = permutation_test(rng.normal(-3, 1, 10), rng.normal(0, 1, 10))
    assert apart["p"] < 0.001
    # Opposite effects at different horizons cancel in the endpoint.
    base = {h: rng.gamma(2.0, 0.2, (1, 200)) for h in (1, 2)}
    a = {1: base[1] - 0.05 + rng.normal(0, 1e-3, (5, 200)), 2: base[2] + 0.05 + rng.normal(0, 1e-3, (5, 200))}
    b = {h: base[h] + rng.normal(0, 1e-3, (5, 200)) for h in base}
    result = endpoint_bootstrap(a, b, {1: 20, 2: 20}, n_resamples=300)
    assert result["low"] < 0 < result["high"] and result["p"] > 0.05


def test_refused_ledger_batch_writes_nothing(trainval_dir, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    base = {"item": "T", "dataset": "ILI", "horizon": 24, "input_len": 104, "seed": 1, "model_config": {}, "n_params": 1,
            "training": {}, "best_val_mse": 0.5, "best_epoch": 1, "epochs_run": 1, "val": {"mse": 0.5}, "history": []}
    first = row_from_record({**base, "config_id": "c1"}, "t")
    append_rows([first], ledger)
    fresh = row_from_record({**base, "config_id": "c2"}, "t")
    clash = row_from_record({**base, "config_id": "c1", "best_val_mse": 0.4, "val": {"mse": 0.4}}, "t")
    with pytest.raises(ValueError):
        append_rows([fresh, clash], ledger)
    assert [r["config_id"] for r in read_rows(ledger)] == ["c1"]


def test_statistics_refuse_empty_or_non_finite_input():
    from leaguetst.eval.stats import endpoint_bootstrap, permutation_test

    with pytest.raises(ValueError):
        permutation_test([], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        permutation_test([1.0, float("nan"), 2.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        permutation_test([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        endpoint_bootstrap({}, {}, {})
    with pytest.raises(ValueError):
        endpoint_bootstrap({1: np.ones((1, 10))}, {1: np.ones((3, 10))}, {1: 2})


def _git_repo(path):
    import subprocess

    def git(*args):
        subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "unit@example.com")
    git("config", "user.name", "unit")
    return git


@pytest.mark.parametrize("attack", ["delete_uncommitted", "truncate_committed", "delete_committed", "edit_uncommitted"])
def test_append_only_files_resist_truncation_and_deletion(tmp_path, attack):
    from leaguetst.eval.integrity import append_only_problems

    git = _git_repo(tmp_path)
    rel = "results/ledger.jsonl"
    path = tmp_path / rel
    path.parent.mkdir(parents=True)
    path.write_text("a\n")
    git("add", rel)
    git("commit", "-q", "-m", "one")
    path.write_text("a\nb\n")
    git("commit", "-q", "-am", "two")
    assert append_only_problems(rel, tmp_path) == []
    if attack == "delete_uncommitted":
        path.unlink()
    elif attack == "truncate_committed":
        path.write_text("a\n")
        git("commit", "-q", "-am", "truncate")
    elif attack == "delete_committed":
        git("rm", "-q", rel)
        git("commit", "-q", "-m", "delete")
    else:
        path.write_text("a\nX\n")
    assert append_only_problems(rel, tmp_path)
