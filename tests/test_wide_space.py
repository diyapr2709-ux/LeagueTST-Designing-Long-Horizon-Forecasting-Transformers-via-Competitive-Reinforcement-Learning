"""Wide forecaster, its parameter counts, the wide feasible sets and the cost model."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from leaguetst.data.device_loader import build_trainval_loaders
from leaguetst.data.trainval import make_trainval, read_manifest
from leaguetst.models import ForecasterConfig, PatchTST, count_parameters
from leaguetst.models.wide import HEAD_TYPES, SharedLinearHead, WideConfig, WideTST
from leaguetst.search import cost_model as cm
from leaguetst.search.feasible import (
    BUDGETS,
    INPUT_LEN,
    SETTINGS,
    config_id,
    feasible_all,
    feasible_table,
    patchtst_parameters,
)
from leaguetst.search.space import enumerate_configs
from leaguetst.search.wide_space import (
    HEAD_CHOICES,
    HORIZONS,
    LOOKBACKS,
    feasible_mask,
    grid_parameters,
    iter_feasible_configs,
    parameter_parts,
    sample_wide_config,
    shape_grid,
    shared_mask,
    summarize,
    wide_config_id,
    wide_feasible_table,
    wide_fits,
    wide_parameters,
    wide_space_size,
)

PARITY_CONFIGS = [
    (dict(patch_len=16, stride=8, n_layers=3, n_heads=4, d_model=16, ff_ratio=8, dropout=0.3), 336, 96),
    (dict(patch_len=24, stride=2, n_layers=3, n_heads=4, d_model=16, ff_ratio=8, dropout=0.3), 104, 24),
    (dict(patch_len=12, stride=6, n_layers=2, n_heads=2, d_model=32, ff_ratio=2, dropout=0.1,
          instance_norm=False, channel_mode="mixing"), 104, 36),
    (dict(patch_len=8, stride=8, n_layers=1, n_heads=8, d_model=64, ff_ratio=1, dropout=0.2,
          channel_mode="mixing"), 192, 48),
]


def _shape_config(dataset: str, index: int, n_heads: int = 4) -> WideConfig:
    g = shape_grid(dataset)
    head_type, head_rank = HEAD_CHOICES[int(g["head"][index])]
    return WideConfig(
        patch_len=int(g["patch_len"][index]), stride=int(g["stride"][index]), n_layers=int(g["n_layers"][index]),
        n_heads=n_heads, d_model=int(g["d_model"][index]), ff_ratio=int(g["ff_ratio"][index]),
        channel_mode="mixing" if g["mixing"][index] else "independent", input_len=int(g["input_len"][index]),
        head_type=head_type, head_rank=head_rank,
    )


# Model


@pytest.mark.parametrize("fields,input_len,horizon", PARITY_CONFIGS)
def test_flatten_head_reproduces_patchtst(fields, input_len, horizon):
    base = ForecasterConfig(**fields)
    torch.manual_seed(11)
    reference = PatchTST(base, 7, input_len, horizon)
    torch.manual_seed(11)
    wide = WideTST(WideConfig.from_forecaster(base, input_len), 7, horizon)
    assert count_parameters(wide) == count_parameters(reference)
    ref_state, wide_state = reference.state_dict(), wide.state_dict()
    assert list(ref_state) == list(wide_state)
    assert all(torch.equal(ref_state[k], wide_state[k]) for k in ref_state)

    x = torch.randn(5, input_len, 7, generator=torch.Generator().manual_seed(3))
    torch.manual_seed(5)
    train_ref = reference(x)
    torch.manual_seed(5)
    train_wide = wide(x)
    assert torch.equal(train_ref, train_wide)
    assert torch.equal(reference.eval()(x), wide.eval()(x))


@pytest.mark.parametrize("head_type,head_rank", HEAD_CHOICES)
@pytest.mark.parametrize("channel_mode", ["independent", "mixing"])
def test_every_head_trains_and_has_the_right_shape(head_type, head_rank, channel_mode):
    config = WideConfig(patch_len=12, stride=6, n_layers=2, n_heads=4, d_model=32, ff_ratio=2, dropout=0.1,
                        channel_mode=channel_mode, input_len=156, head_type=head_type, head_rank=head_rank)
    torch.manual_seed(0)
    model = WideTST(config, 7, 36)
    model.store_attention = True
    x = torch.randn(4, 156, 7)
    y = model(x)
    assert y.shape == (4, 36, 7) and torch.isfinite(y).all()
    assert len(model.attention_maps) == 2
    y.pow(2).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_shared_linear_head_starts_as_the_pooled_head():
    head = SharedLinearHead(n_patches=9, d_model=16, out_features=24)
    x = torch.randn(3, 7, 9 * 16)
    pooled = head.per_patch(x.unflatten(-1, (9, 16)).mean(dim=-2))
    assert torch.allclose(head(x), pooled, atol=1e-6)


def test_wide_config_validation():
    with pytest.raises(ValueError):
        WideConfig(head_type="flatten", head_rank=4)
    with pytest.raises(ValueError):
        WideConfig(head_type="lowrank", head_rank=0)
    with pytest.raises(ValueError):
        WideConfig(head_type="attention")
    with pytest.raises(ValueError):
        WideConfig(patch_len=24, stride=12, input_len=12)
    with pytest.raises(ValueError):
        WideConfig(d_model=24, n_heads=16)
    with pytest.raises(ValueError):
        WideConfig.from_dict({"head": "pooled"})
    config = WideConfig(input_len=np.int64(512), head_type="lowrank", head_rank=np.int64(8))
    assert type(config.input_len) is int and type(config.head_rank) is int
    assert WideConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    assert config.backbone() == ForecasterConfig()
    assert config.n_patches == ForecasterConfig().n_patches(512)


# Closed-form parameter counts


def test_flatten_count_equals_patchtst_count_on_the_whole_grid():
    configs = list(enumerate_configs())
    columns = {name: np.array([getattr(c, name) for c in configs]) for name in
               ("patch_len", "stride", "n_layers", "d_model", "ff_ratio")}
    mixing = np.array([c.channel_mode == "mixing" for c in configs])
    for dataset, horizon in SETTINGS:
        length = INPUT_LEN[dataset]
        backbone, head = parameter_parts(
            columns["patch_len"], columns["stride"], columns["n_layers"], columns["d_model"], columns["ff_ratio"],
            mixing, 0, 0, length, horizon,
        )
        expected = np.array([patchtst_parameters(c, length, horizon) for c in configs])
        assert np.array_equal(backbone + head, expected)


def test_closed_form_matches_meta_models_for_every_head():
    rng = random.Random(2026)
    checked = {choice: 0 for choice in HEAD_CHOICES}
    for dataset in LOOKBACKS:
        for _ in range(300):
            config = sample_wide_config(dataset, rng)
            horizon = rng.choice(HORIZONS[dataset])
            with torch.device("meta"):
                model = WideTST(config, 7, horizon)
            assert wide_parameters(config, horizon) == count_parameters(model), wide_config_id(config)
            checked[(config.head_type, config.head_rank)] += 1
    # Every head and both channel modes at the largest lookback, width and horizon.
    for (head_type, head_rank), mode in [(h, m) for h in HEAD_CHOICES for m in ("independent", "mixing")]:
        config = WideConfig(patch_len=8, stride=4, n_layers=4, n_heads=16, d_model=256, ff_ratio=4,
                            channel_mode=mode, input_len=512, head_type=head_type, head_rank=head_rank)
        with torch.device("meta"):
            model = WideTST(config, 7, 720)
        assert wide_parameters(config, 720) == count_parameters(model)
    assert min(checked.values()) >= 50


def test_grid_counts_match_scalar_counts_and_fits():
    rng = np.random.default_rng(4)
    for dataset in LOOKBACKS:
        n_shapes = len(shape_grid(dataset)["d_model"])
        rows = rng.choice(n_shapes, size=400, replace=False)
        masks = {h: feasible_mask(dataset, (h,)) for h in HORIZONS[dataset]}
        all_four = feasible_mask(dataset)
        for horizon in HORIZONS[dataset]:
            counts = grid_parameters(dataset, horizon)
            for i in rows:
                config = _shape_config(dataset, int(i))
                assert counts[i] == wide_parameters(config, horizon)
                assert masks[horizon][i] == wide_fits(config, dataset, horizon)
        for i in rows:
            config = _shape_config(dataset, int(i))
            assert all_four[i] == all(wide_fits(config, dataset, h) for h in HORIZONS[dataset])


# Feasible sets


def test_flatten_at_default_lookback_reproduces_todays_feasible_set():
    table = wide_feasible_table()
    today = feasible_table()
    for key, row in today["per_setting"].items():
        assert table["flatten_default"][key] == row["feasible"]
    assert table["flatten_default"]["all_settings"] == today["all_settings"] == 2368
    g = shape_grid("ETTh1")
    flatten = shared_mask() & (g["head_code"] == 0)
    assert flatten.sum() == 74
    found = {config_id(c.backbone()) for c in iter_feasible_configs("ETTh1", flatten)}
    assert found == {config_id(c) for c in feasible_all()}


def test_wide_space_size_and_what_the_cap_allows():
    assert wide_space_size("ETTh1") == 30720 * 6 * 4 == 737_280
    assert wide_space_size("ILI") == 30720 * 6 * 3 == 552_960
    table = wide_feasible_table()
    for dataset in LOOKBACKS:
        row = table["per_dataset"][dataset]
        assert 10_000 <= row["configs"] <= 1_000_000
        assert row["configs"] == 32 * row["architectures"]
        for part in ("by_head", "by_width", "by_channel", "by_lookback"):
            assert sum(row[part].values()) == row["configs"]
        assert row["by_width"]["64"] > 0 and row["by_channel"]["mixing"] > 0
        # One encoder layer at width 128 alone exceeds the smallest cap of each dataset.
        assert row["by_width"]["128"] == row["by_width"]["256"] == 0
        assert all(row["by_head"][label] > 0 for label in row["by_head"])
    assert table["per_setting"]["ETTh1_T336"]["by_width"]["128"] > 0
    assert table["per_setting"]["ETTh1_T720"]["by_width"]["256"] > 0
    shared = table["shared"]
    assert shared["by_head"]["flatten"] == 2368 and shared["configs"] > 10 * 2368


def test_feasible_iteration_stays_inside_the_set():
    g = shape_grid("ILI")
    mask = feasible_mask("ILI") & (g["d_model"] == 64)
    configs = list(iter_feasible_configs("ILI", mask))
    assert len(configs) == summarize("ILI", mask)["configs"] == len(set(configs))
    assert all(wide_fits(c, "ILI", h) for c in configs for h in HORIZONS["ILI"])


def test_wide_config_ids_are_unique():
    rng = random.Random(1)
    configs = {sample_wide_config(dataset, rng) for dataset in LOOKBACKS for _ in range(3000)}
    assert len({wide_config_id(c) for c in configs}) == len(configs)
    assert wide_config_id(WideConfig(head_type="lowrank", head_rank=16)) == "p16s8l3h4d16f8r3n1iL336R16"


# Cost model


def test_matmul_flops_match_the_torch_counter():
    rng = random.Random(5)
    configs = [sample_wide_config(d, rng) for d in LOOKBACKS for _ in range(30)]
    configs += [WideConfig(head_type=h, head_rank=r, channel_mode=m) for h, r in HEAD_CHOICES for m in ("independent", "mixing")]
    for config in configs:
        horizon = 36
        with torch.device("meta"):
            model = WideTST(config, 7, horizon).eval()
            with FlopCounterMode(display=False) as counter:
                model(torch.empty(3, config.input_len, 7))
        flops, elements = cm.config_work(config, horizon)
        assert counter.get_total_flops() == 3 * flops
        assert elements > 0


@pytest.fixture(scope="module")
def trainval_dir(synthetic_data_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("trainval_wide")
    make_trainval(synthetic_data_dir, out)
    return out


@pytest.mark.parametrize("dataset,input_len,horizon,batch", [("ETTh1", 512, 96, 128), ("ILI", 52, 60, 16), ("ILI", 104, 24, 16)])
def test_epoch_steps_match_the_loader(trainval_dir, dataset, input_len, horizon, batch):
    n_train, n_val = cm.epoch_windows(dataset, input_len, horizon, read_manifest(trainval_dir))
    assert (n_train, n_val) == cm.epoch_windows(dataset, input_len, horizon)
    loaders, _ = build_trainval_loaders(trainval_dir, dataset, input_len, horizon, batch, 1, torch.device("cpu"))
    assert len(loaders["train"]) == n_train // batch
    assert len(loaders["val"]) == -(-n_val // batch)
    assert len(loaders["val"].dataset) == n_val


def test_fit_recovers_known_coefficients():
    rng = random.Random(3)
    true = np.array([2e-3, 3e-3, 1e-2, 5e-2])
    x = np.stack([cm.config_features(sample_wide_config("ILI", rng), "ILI", 24, 16) for _ in range(200)])
    model = cm.fit_cost_model(x, x @ true)
    assert np.allclose(model.coefficients, true, rtol=1e-6)
    quality = cm.fit_quality(model.predict(x), x @ true)
    assert quality["r2"] > 1 - 1e-9 and quality["median_rel_error"] < 1e-9
    with pytest.raises(ValueError):
        cm.fit_cost_model(x[:2], (x @ true)[:2])


def _ledger_row(config: ForecasterConfig, seconds: list[float], wall: float, gpu: str, seed: int) -> dict:
    return {
        "item": "E1", "dataset": "ILI", "horizon": 24, "input_len": 104, "seed": seed, "config_id": config_id(config),
        "model_config": config.to_dict(), "training": {"batch_size": 16}, "epoch_seconds": seconds,
        "epochs_run": len(seconds), "wall_seconds": wall, "gpu": gpu,
    }


def test_ledger_timings_cross_validation_and_pricing(tmp_path: Path):
    rng = random.Random(8)
    true = np.array([2e-3, 3e-3, 0.0, 5e-2])
    configs = rng.sample(list(feasible_all()), 60)
    path = tmp_path / "ledger.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for seed in (1, 2):
            for config in configs:
                steady = float(cm.config_features(WideConfig.from_forecaster(config, 104), "ILI", 24, 16) @ true)
                seconds = [steady + 0.5] + [steady] * 9
                handle.write(json.dumps(_ledger_row(config, seconds, sum(seconds) + 0.1, cm.A10G, seed)) + "\n")
        handle.write(json.dumps(_ledger_row(configs[0], [1.0, 1.0], 2.0, "NVIDIA A10", 1)) + "\n")
    timings = cm.ledger_timings(path, gpus=[cm.A10G])
    assert len(timings) == 120
    assert all(abs(t.overhead_seconds - 0.6) < 1e-9 for t in timings)
    held_out = cm.held_out_predictions(timings, n_folds=4)
    measured = np.array([t.epoch_seconds for t in timings])
    assert cm.fit_quality(held_out, measured)["median_rel_error"] < 1e-6
    assert cm.container_seconds(timings) == pytest.approx(sum(t.wall_seconds for t in timings) / cm.WORKERS_PER_GPU)

    model = cm.CostModel(coefficients=true)
    g = shape_grid("ILI")
    mask = feasible_mask("ILI") & (g["d_model"] == 64) & (g["head_code"] == 1)
    price = cm.price_enumeration(model, "ILI", 24, mask, 16, epochs=20, overhead_seconds=0.5)
    configs = list(iter_feasible_configs("ILI", mask))
    expected = sum(20 * float(cm.config_features(c, "ILI", 24, 16) @ true) + 0.5 for c in configs) / cm.WORKERS_PER_GPU
    assert price["configs"] == summarize("ILI", mask)["configs"] == len(configs)
    assert price["gpu_hours"] == pytest.approx(expected / 3600)
    assert price["dollars"] == pytest.approx(price["gpu_hours"] * cm.DOLLARS_PER_GPU_HOUR)
    assert 0 < price["floor_gpu_hours"] < price["gpu_hours"]
    scaled = cm.price_enumeration(model, "ILI", 24, mask, 16, epochs=20, overhead_seconds=0.0, scale=2.0)
    plain = cm.price_enumeration(model, "ILI", 24, mask, 16, epochs=20, overhead_seconds=0.0)
    assert scaled["gpu_hours"] == pytest.approx(2 * plain["gpu_hours"])
    assert scaled["floor_gpu_hours"] == pytest.approx(plain["floor_gpu_hours"])
    heavy = cm.price_enumeration(model, "ILI", 24, mask, 16, epochs=20, overhead_seconds=0.5, memory_limit_bytes=0)
    assert heavy["configs_over_memory"] == price["configs"] and price["configs_over_memory"] == 0
    assert heavy["gpu_hours"] == pytest.approx(price["gpu_hours"])


def test_budgets_are_the_feasible_module_caps():
    assert set(BUDGETS) == {(d, h) for d in LOOKBACKS for h in HORIZONS[d]}
    assert set(HEAD_TYPES) == {h for h, _ in HEAD_CHOICES}
