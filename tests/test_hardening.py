"""Regression tests for the findings of the penetration rounds.

Each test replays one attack or malformed input from docs/validation_log.md
and checks that it is now refused or flagged.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from leaguetst.data import load_benchmark_frame
from leaguetst.experiment import RunSpec, run_experiment
from leaguetst.models import ForecasterConfig
from leaguetst.training import TrainingConfig
from leaguetst.utils import load_state_dict, load_yaml, save_json, save_state_dict, sha256_file

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quick_spec(data_dir, out_dir, **overrides) -> RunSpec:
    values = dict(
        model="dlinear", dataset="ILI", horizon=24, seed=1,
        training=TrainingConfig(epochs=1, batch_size=32, learning_rate=1e-3, patience=5),
        device="cpu", data_dir=str(data_dir), out_dir=str(out_dir), checkpoint_dir=str(Path(out_dir) / "ckpt"),
    )
    values.update(overrides)
    return RunSpec(**values)


@pytest.mark.parametrize("tag", ["x/../../escaped", "../up", ".hidden", "a\\b", "with space", "x" * 65])
def test_tags_that_could_leave_the_output_directory_are_refused(tag, tmp_path):
    with pytest.raises(ValueError):
        quick_spec(tmp_path, tmp_path, tag=tag)


def test_plain_tags_still_work(tmp_path):
    assert quick_spec(tmp_path, tmp_path, tag="lr-1e3_v2").run_name.endswith("_lr-1e3_v2")


@pytest.mark.parametrize(
    "values",
    [
        {"instance_norm": "false"},
        {"instance_norm": 1},
        {"patch_len": True, "stride": True},
        {"patch_len": 16.5},
        {"n_layers": 2.0},
        {"n_heads": 0},
        {"n_heads": -4, "d_model": -16},
        {"d_model": 2**20},
        {"n_layers": 13},
        {"dropout": float("nan")},
        {"dropout": "0.1"},
        {"dropout": True},
        {"channel_mode": ["independent"]},
    ],
)
def test_config_rejects_wrong_types_and_runaway_sizes(values):
    with pytest.raises(ValueError):
        ForecasterConfig(**values)


def test_config_normalises_numpy_integers():
    config = ForecasterConfig(patch_len=np.int64(16), stride=np.int64(8))
    assert type(config.patch_len) is int and type(config.stride) is int
    json.dumps(config.to_dict())


@pytest.mark.parametrize("lr", [-1.0, 0.0, float("nan"), float("inf"), "0.001"])
def test_training_config_rejects_bad_learning_rates(lr):
    with pytest.raises(ValueError):
        TrainingConfig(learning_rate=lr)


@pytest.mark.parametrize("values", [{"horizon": 0}, {"horizon": -5}, {"seed": -1}, {"seed": 2**40}, {"input_len": 0}, {"horizon": 24.0}])
def test_run_spec_rejects_out_of_range_values(values, tmp_path):
    with pytest.raises(ValueError):
        quick_spec(tmp_path, tmp_path, **values)


def test_divergence_is_flagged_and_refused_by_the_summary(synthetic_data_dir, tmp_path):
    spec = quick_spec(
        synthetic_data_dir, tmp_path, tag="diverge",
        training=TrainingConfig(epochs=3, batch_size=32, learning_rate=1e30, patience=5),
    )
    record = run_experiment(spec)
    assert record["fit"]["diverged"] is True
    assert record["fit"]["epochs_run"] == 1
    summarize = load_script("summarize_results")
    with pytest.raises(summarize.RecordError, match="diverged"):
        summarize.load_records([tmp_path])


def write_tampered(source: Path, target: Path, edit) -> Path:
    shutil.copytree(source, target)
    path = target / "illness" / "national_illness.csv"
    edit(pd.read_csv(path)).to_csv(path, index=False)
    return target


@pytest.mark.parametrize(
    "edit,message",
    [
        (lambda f: f.assign(**{"AGE 0-4": f["AGE 0-4"].where(f.index != 100)}), "missing or infinite"),
        (lambda f: f.sample(frac=1.0, random_state=0), "time order"),
        (lambda f: f.drop(columns=["OT"]), "columns"),
        (lambda f: f.assign(EXTRA=1.0), "columns"),
        (lambda f: f.drop(index=[50]), "time order"),
        (lambda f: f.assign(OT=f["OT"].astype(str).where(f.index != 7, "abc")), "non-numeric"),
    ],
    ids=["nan", "shuffled", "no-target", "extra-column", "missing-row", "text-value"],
)
def test_malformed_benchmark_files_are_refused(synthetic_data_dir, tmp_path, edit, message):
    root = write_tampered(synthetic_data_dir, tmp_path / "data", edit)
    with pytest.raises(ValueError, match=message):
        load_benchmark_frame(root, "ILI")


def test_summary_refuses_duplicates_mixed_configs_and_broken_records(synthetic_data_dir, tmp_path):
    summarize = load_script("summarize_results")
    first, second = tmp_path / "one", tmp_path / "two"
    run_experiment(quick_spec(synthetic_data_dir, first, seed=1))
    second.mkdir()
    shutil.copy(next(first.glob("*.json")), second)
    with pytest.raises(summarize.RecordError, match="appears twice"):
        summarize.load_records([first, second])

    run_experiment(quick_spec(synthetic_data_dir, first, seed=2, model_config={"kernel_size": 5}))
    with pytest.raises(summarize.RecordError, match="model_config differs"):
        summarize.summarize(summarize.load_records([first]))

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "bad.json").write_text(json.dumps({"model": "dlinear"}))
    with pytest.raises(summarize.RecordError, match="bad.json"):
        summarize.load_records([broken])


def test_stale_record_is_detected(synthetic_data_dir, tmp_path):
    spec = quick_spec(synthetic_data_dir, tmp_path)
    record = run_experiment(spec)
    assert spec.stale_fields(json.loads(json.dumps(record))) == []
    changed = quick_spec(
        synthetic_data_dir, tmp_path,
        training=TrainingConfig(epochs=1, batch_size=32, learning_rate=1e-3, patience=5, lr_schedule="constant"),
    )
    assert changed.stale_fields(record) == ["training"]


def test_interrupted_json_write_leaves_no_partial_file(tmp_path, monkeypatch):
    import leaguetst.utils.io as io_module

    target = tmp_path / "record.json"
    save_json({"version": 1}, target)

    def exploding_dump(obj, handle, **kwargs):
        handle.write('{"version": 2, "tru')
        raise KeyboardInterrupt

    monkeypatch.setattr(io_module.json, "dump", exploding_dump)
    with pytest.raises(KeyboardInterrupt):
        save_json({"version": 2}, target)
    assert json.loads(target.read_text()) == {"version": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["record.json"]


def test_checkpoint_loader_refuses_pickled_code_and_wrong_digests(tmp_path):
    marker = tmp_path / "ran"

    class Payload:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    evil = tmp_path / "evil.pt"
    torch.save({"weight": Payload()}, evil)
    with pytest.raises(Exception):
        load_state_dict(evil)
    assert not marker.exists()

    good = tmp_path / "good.pt"
    digest = save_state_dict({"w": torch.ones(2)}, good)
    assert digest == sha256_file(good)
    assert torch.equal(load_state_dict(good, expected_sha256=digest)["w"], torch.ones(2))
    with pytest.raises(ValueError, match="does not match"):
        load_state_dict(good, expected_sha256="0" * 64)


def test_records_carry_the_checkpoint_digest(synthetic_data_dir, tmp_path):
    record = run_experiment(quick_spec(synthetic_data_dir, tmp_path))
    assert record["checkpoint_sha256"] == sha256_file(record["checkpoint"])


def test_yaml_loader_builds_no_objects_and_wants_a_mapping(tmp_path):
    marker = tmp_path / "ran"
    evil = tmp_path / "evil.yaml"
    evil.write_text(f'!!python/object/apply:os.system ["touch {marker}"]\n')
    with pytest.raises(Exception):
        load_yaml(evil)
    assert not marker.exists()
    listing = tmp_path / "list.yaml"
    listing.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(listing)


def test_download_refuses_non_https_and_unpinned_content(tmp_path, monkeypatch):
    download = load_script("download_data")
    forged = tmp_path / "forged.csv"
    forged.write_text("date,HUFL,OT\n2016-07-01 00:00:00,1,2\n")
    expected = download.FILES["ETT-small/ETTh1.csv"][0]
    destination = tmp_path / "out" / "ETTh1.csv"
    assert not download.fetch(forged.as_uri(), destination, expected)
    assert not download.fetch("http://example.com/ETTh1.csv", destination, expected)

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def geturl(self):
            return "https://mirror.example/ETTh1.csv"

        def read(self, limit):
            return self.payload[:limit]

    monkeypatch.setattr(download.urllib.request, "urlopen", lambda url, timeout: FakeResponse(forged.read_bytes()))
    assert not download.fetch("https://mirror.example/ETTh1.csv", destination, expected)
    assert not destination.exists()


def test_concurrent_writers_never_expose_a_torn_record(tmp_path):
    import threading

    target = tmp_path / "shared.json"
    save_json({"writer": -1, "pad": "x" * 20000}, target)
    stop, torn, failures = threading.Event(), [], []

    def reader():
        while not stop.is_set():
            try:
                json.loads(target.read_text())
            except (json.JSONDecodeError, FileNotFoundError) as error:
                torn.append(error)

    def writer(index):
        try:
            for _ in range(40):
                save_json({"writer": index, "pad": "x" * 20000}, target)
        except Exception as error:  # noqa: BLE001 - collected for the assertion
            failures.append(error)

    threads = [threading.Thread(target=reader)] + [threading.Thread(target=writer, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads[1:]:
        thread.join()
    stop.set()
    threads[0].join()
    assert not torn and not failures
    assert [p.name for p in tmp_path.iterdir()] == ["shared.json"]


def test_summary_refuses_non_finite_metrics(synthetic_data_dir, tmp_path):
    summarize = load_script("summarize_results")
    run_experiment(quick_spec(synthetic_data_dir, tmp_path, checkpoint_dir=""))
    path = next(tmp_path.glob("*.json"))
    record = json.loads(path.read_text())
    record["test"]["mse"] = float("nan")
    path.write_text(json.dumps(record))
    with pytest.raises(summarize.RecordError, match="non-finite test.mse"):
        summarize.load_records([tmp_path])


def test_records_written_before_new_training_fields_still_match_defaults(synthetic_data_dir, tmp_path):
    spec = quick_spec(synthetic_data_dir, tmp_path)
    record = json.loads(json.dumps(run_experiment(spec)))
    for field in ("adam_beta1", "first_epoch_lr_factor"):
        record["training"].pop(field)
    assert spec.stale_fields(record) == []
    patchtst_like = quick_spec(
        synthetic_data_dir, tmp_path,
        training=TrainingConfig(epochs=1, batch_size=32, learning_rate=1e-3, patience=5, adam_beta1=0.95),
    )
    assert patchtst_like.stale_fields(record) == ["training"]
