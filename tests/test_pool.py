"""Freedoom-first training map pool."""

import numpy as np
import pytest

from training.mapgen.tiers import STAGES
from training.pool import (
    GENERATED_FRAC,
    TRAIN_MAPS,
    VAL_MAPS,
    MapSpec,
    load_manifest,
    make_training_env,
    sample_spec,
)


def test_manifest_lists_map01():
    data = load_manifest()
    by_name = {e["map"]: e for e in data["maps"]}
    assert "MAP01" in by_name
    assert by_name["MAP01"]["load_ok"] is True


def test_sample_all_freedoom_train():
    rng = np.random.default_rng(0)
    for _ in range(40):
        spec = sample_spec(rng, generated_frac=0.0, split="train")
        assert spec.source == "freedoom"
        assert spec.map in TRAIN_MAPS
        assert spec.episode_ticks == 4200


def test_sample_all_generated():
    rng = np.random.default_rng(1)
    for _ in range(40):
        spec = sample_spec(rng, generated_frac=1.0, split="train")
        assert spec.source == "generated"
        assert spec.stage in STAGES
        assert spec.map == "MAP01"
        assert 0 <= spec.seed < 10_000


def test_sample_val_holdout():
    rng = np.random.default_rng(2)
    for _ in range(20):
        spec = sample_spec(rng, generated_frac=0.0, split="val")
        assert spec.map in VAL_MAPS
    spec = sample_spec(rng, generated_frac=1.0, split="val")
    assert spec.seed >= 10_000


def test_default_mix_is_freedoom_only():
    rng = np.random.default_rng(3)
    n = 200
    specs = [sample_spec(rng, generated_frac=GENERATED_FRAC, split="train") for _ in range(n)]
    assert all(s.source == "freedoom" for s in specs)
    assert {s.map for s in specs} <= set(TRAIN_MAPS)


def test_make_training_env_freedoom_map01():
    spec = MapSpec(wad="", map="MAP01", source="freedoom", episode_ticks=100)
    env = make_training_env(spec, episode_ticks=100, render_buffers=False)
    try:
        obs, _ = env.reset(seed=0)
        assert "vitals" in obs
    finally:
        env.close()


@pytest.mark.parametrize("map_name", ["MAP01", "MAP08", "MAP32"])
def test_freedoom_short_rollout_kill_does_not_swap_map(map_name):
    spec = MapSpec(wad="", map=map_name, source="freedoom", episode_ticks=400)
    env = make_training_env(spec, episode_ticks=400, render_buffers=False)
    rng = np.random.default_rng(0)
    try:
        env.reset(seed=0)
        assert env.map_name == map_name
        for _ in range(20):
            action = rng.integers(0, 2, size=9, dtype=np.int8)
            _obs, _r, term, trunc, info = env.step(action)
            st = env.privileged_state()
            if st["kills"] > 0 and not st["dead"]:
                assert not term
                assert not info["exit_reached"]
            if term or trunc:
                if st["kills"] > 0:
                    assert st["dead"] or info["exit_reached"] or trunc
                break
            assert env.map_name == map_name
    finally:
        env.close()


def test_make_training_env_generated_s0(tmp_path):
    spec = MapSpec(
        wad="",
        map="MAP01",
        source="generated",
        episode_ticks=2100,
        stage="S0",
        seed=0,
    )
    env = make_training_env(spec, tmp_dir=tmp_path, episode_ticks=100, render_buffers=False)
    try:
        env.reset(seed=0)
        assert (tmp_path / "S0_0.wad").is_file()
    finally:
        env.close()
