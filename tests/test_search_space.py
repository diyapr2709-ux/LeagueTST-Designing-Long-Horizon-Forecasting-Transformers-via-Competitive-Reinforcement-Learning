"""Size and validity of the design space."""

from __future__ import annotations

import random

from leaguetst.search import SEARCH_SPACE, enumerate_configs, sample_config, space_size
from leaguetst.search.space import resolve_choice


def test_space_has_30720_configs():
    assert space_size() == 30720
    assert len(SEARCH_SPACE) == 9


def test_every_grid_point_builds_a_valid_config():
    seen = set()
    for config in enumerate_configs():
        assert config.d_model % config.n_heads == 0
        assert config.stride in (config.patch_len // 2, config.patch_len)
        seen.add(config)
    assert len(seen) == 30720


def test_sampling_is_deterministic_and_inside_the_grid():
    first = [sample_config(random.Random(7)) for _ in range(3)]
    second = [sample_config(random.Random(7)) for _ in range(3)]
    assert first == second
    grid = set(enumerate_configs())
    assert all(config in grid for config in first)


def test_stride_words_resolve_to_integers():
    config = resolve_choice({"patch_len": 24, "stride": "half"})
    assert config.stride == 12
    config = resolve_choice({"patch_len": 24, "stride": "full"})
    assert config.stride == 24
    config = resolve_choice({"patch_len": 24, "stride": 2})
    assert config.stride == 2
