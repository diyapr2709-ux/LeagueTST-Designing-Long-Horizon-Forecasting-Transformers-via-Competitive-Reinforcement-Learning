"""Shapes, parameter counts and config validation of the forecaster."""

from __future__ import annotations

import pytest
import torch

from leaguetst.models import ForecasterConfig, PatchTST, RevIN, count_parameters

PATCHTST_42 = ForecasterConfig(
    patch_len=16, stride=8, n_layers=3, n_heads=4, d_model=16, ff_ratio=8,
    dropout=0.3, instance_norm=True, channel_mode="independent",
)


def hand_count_patchtst42(horizon: int, n_patches: int = 42, d_model: int = 16, d_ff: int = 128) -> int:
    """Parameter count of PatchTST/42 worked out layer by layer."""
    embedding = 16 * d_model + d_model
    positional = n_patches * d_model
    attention = 4 * (d_model * d_model + d_model)
    feed_forward = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    batch_norms = 2 * (2 * d_model)
    layers = 3 * (attention + feed_forward + batch_norms)
    head = d_model * n_patches * horizon + horizon
    return embedding + positional + layers + head


def test_patch_count_matches_patchtst_42():
    assert PATCHTST_42.n_patches(336) == 42
    ili = ForecasterConfig(patch_len=24, stride=2)
    assert ili.n_patches(104) == 42


@pytest.mark.parametrize("horizon", [96, 336, 720])
def test_patchtst42_parameter_count(horizon):
    model = PatchTST(PATCHTST_42, n_vars=7, input_len=336, horizon=horizon)
    assert count_parameters(model) == hand_count_patchtst42(horizon)


@pytest.mark.parametrize("channel_mode", ["independent", "mixing"])
@pytest.mark.parametrize("instance_norm", [True, False])
def test_patchtst_output_shape(channel_mode, instance_norm):
    config = ForecasterConfig(
        patch_len=12, stride=6, n_layers=2, n_heads=2, d_model=32, ff_ratio=2,
        dropout=0.1, instance_norm=instance_norm, channel_mode=channel_mode,
    )
    model = PatchTST(config, n_vars=7, input_len=104, horizon=24)
    x = torch.randn(5, 104, 7)
    y = model(x)
    assert y.shape == (5, 24, 7)
    assert torch.isfinite(y).all()


def test_patchtst_stores_attention_maps_on_request():
    model = PatchTST(PATCHTST_42, n_vars=7, input_len=336, horizon=96).eval()
    model.store_attention = True
    model(torch.randn(2, 336, 7))
    assert len(model.attention_maps) == 3
    assert model.attention_maps[0].shape == (2 * 7, 4, 42, 42)
    rows = model.attention_maps[0].sum(dim=-1)
    assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5)


def test_patchtst_is_trainable():
    model = PatchTST(PATCHTST_42, n_vars=7, input_len=336, horizon=96)
    x = torch.randn(4, 336, 7)
    loss = (model(x) - torch.randn(4, 96, 7)).pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_revin_round_trip():
    revin = RevIN()
    x = torch.randn(3, 50, 7) * 4 + 2
    scaled, mean, std = revin.normalize(x)
    assert torch.allclose(scaled.mean(dim=1), torch.zeros(3, 7), atol=1e-5)
    assert torch.allclose(RevIN.denormalize(scaled, mean, std), x, atol=1e-4)


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        ForecasterConfig(d_model=16, n_heads=3)
    with pytest.raises(ValueError):
        ForecasterConfig(patch_len=8, stride=16)
    with pytest.raises(ValueError):
        ForecasterConfig(channel_mode="shared")
    with pytest.raises(ValueError):
        ForecasterConfig.from_dict({"d_model": 16, "width": 32})


def test_config_round_trips_through_dict():
    config = ForecasterConfig(patch_len=24, stride=12, instance_norm=False)
    assert ForecasterConfig.from_dict(config.to_dict()) == config
    assert config.d_ff == config.ff_ratio * config.d_model
