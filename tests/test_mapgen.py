"""Tests for training.mapgen: primitives plus S0–S3 layout."""

import hashlib

import numpy as np
import pytest

from doom_env import make_env
from doom_env import spaces as S
from doom_env.sensors import RAY_DOOR, RAY_LOCKED

from training.mapgen.layout import build, point_in_poly
from training.mapgen.preview import render_preview
from training.mapgen.tiers import STAGES
from training.mapgen.wadbuild import build_spike_map

FWD = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)
USE = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0], np.int8)
NOOP = np.zeros(9, np.int8)


def _ray0(obs):
    return int(obs["rays"]["hit_type"][0])


def _walk_to_door(env, obs, kinds=(RAY_DOOR, RAY_LOCKED)):
    for _ in range(120):
        obs, *_ = env.step(FWD)
        if float(obs["rays"]["distance"][0]) * S.RAY_RANGE < 40 and _ray0(obs) in kinds:
            return obs
    return obs


def _use_once_and_wait(env, obs):
    obs, *_ = env.step(USE)
    for _ in range(35):
        obs, *_ = env.step(NOOP)
    return obs


def _walk_east(env, obs, steps=200):
    term = trunc = False
    for _ in range(steps):
        obs, _r, term, trunc, _info = env.step(FWD)
        if term or trunc:
            break
    return obs, term, trunc


def test_udmf_locked_door_special():
    text = build_spike_map(lock=1, with_key=True).udmf()
    assert 'namespace = "zdoom"' in text
    assert "special = 13" in text
    assert "arg3 = 1" in text


def test_plain_door_opens_and_exit(tmp_path):
    wad = tmp_path / "plain.wad"
    build_spike_map(lock=0, with_key=False).save(wad)
    env = make_env(str(wad), "MAP01", episode_ticks=2100, render_buffers=False)
    obs, _ = env.reset(seed=0)
    obs = _walk_to_door(env, obs, kinds=(RAY_DOOR,))
    assert _ray0(obs) == RAY_DOOR
    obs = _use_once_and_wait(env, obs)
    assert _ray0(obs) != RAY_DOOR
    _obs, term, trunc = _walk_east(env, obs)
    st = env.privileged_state()
    env.close()
    assert st["exit_reached"]
    assert term and not trunc


def test_locked_door_with_red_card(tmp_path):
    wad = tmp_path / "locked_key.wad"
    build_spike_map(lock=1, with_key=True).save(wad)
    env = make_env(str(wad), "MAP01", episode_ticks=2100, render_buffers=False)
    obs, _ = env.reset(seed=0)
    assert obs["vitals"][S.VITALS_INDEX["key_red_card"]] == 0
    picked = False
    for _ in range(120):
        obs, *_ = env.step(FWD)
        picked = picked or obs["vitals"][S.VITALS_INDEX["key_red_card"]] == 1
        if float(obs["rays"]["distance"][0]) * S.RAY_RANGE < 40 and _ray0(obs) == RAY_LOCKED:
            break
    assert picked
    assert _ray0(obs) == RAY_LOCKED
    obs = _use_once_and_wait(env, obs)
    assert _ray0(obs) != RAY_LOCKED
    _obs, term, trunc = _walk_east(env, obs)
    st = env.privileged_state()
    env.close()
    assert st["exit_reached"]
    assert term and not trunc


def test_locked_door_without_key_stays_shut(tmp_path):
    wad = tmp_path / "locked_nokey.wad"
    build_spike_map(lock=1, with_key=False).save(wad)
    env = make_env(str(wad), "MAP01", episode_ticks=2100, render_buffers=False)
    obs, _ = env.reset(seed=0)
    obs = _walk_to_door(env, obs, kinds=(RAY_LOCKED,))
    assert _ray0(obs) == RAY_LOCKED
    obs = _use_once_and_wait(env, obs)
    assert _ray0(obs) == RAY_LOCKED
    _obs, term, trunc = _walk_east(env, obs)
    st = env.privileged_state()
    env.close()
    assert not st["exit_reached"]
    assert not term


def _edge_owner_counts(m):
    vid = {}

    def V(p):
        if p not in vid:
            vid[p] = len(vid)
        return vid[p]

    counts = {}
    for s in m.sectors:
        poly = s["poly"]
        for i in range(len(poly)):
            a, b = V(poly[i]), V(poly[(i + 1) % len(poly)])
            key = frozenset((a, b))
            counts[key] = counts.get(key, 0) + 1
    return counts


@pytest.mark.parametrize("stage", list(STAGES))
@pytest.mark.parametrize("seed", range(8))
def test_layout_geometry_and_load(tmp_path, stage, seed):
    built = build(stage, seed)
    assert built.sector_count < 200
    assert built.sector_count == len(built.map.sectors)
    if len(built.rooms) >= 2:
        assert built.start_room != built.exit_room
    start_poly = built.map.sectors[built.start_room]["poly"]
    assert point_in_poly(*built.start_xy, start_poly)
    counts = _edge_owner_counts(built.map)
    assert counts
    assert all(c in (1, 2) for c in counts.values())
    assert 2 in counts.values()
    text = built.map.udmf()
    assert 'namespace = "zdoom"' in text
    if stage == "S2":
        assert "special = 12" in text
    wad = tmp_path / f"{stage}_{seed}.wad"
    built.map.save(wad)
    env = make_env(str(wad), "MAP01", episode_ticks=100, render_buffers=False)
    env.reset(seed=0)
    env.close()


def test_layout_determinism():
    a = build("S3", 7).map.udmf()
    b = build("S3", 7).map.udmf()
    c = build("S3", 8).map.udmf()
    ha = hashlib.sha1(a.encode()).hexdigest()
    hb = hashlib.sha1(b.encode()).hexdigest()
    hc = hashlib.sha1(c.encode()).hexdigest()
    assert ha == hb
    assert ha != hc


def test_s0_walk_to_exit(tmp_path):
    built = build("S0", 0)
    assert built.map.things[0][2] == 0  # facing east
    wad = tmp_path / "s0.wad"
    built.map.save(wad)
    env = make_env(str(wad), "MAP01", episode_ticks=2100, render_buffers=False)
    obs, _ = env.reset(seed=0)
    _obs, term, trunc = _walk_east(env, obs, steps=400)
    st = env.privileged_state()
    env.close()
    assert st["exit_reached"]
    assert term and not trunc


def test_s2_udmf_has_plain_doors():
    text = build("S2", 0).map.udmf()
    assert "special = 12" in text
    assert "special = 13" not in text


def test_preview_png(tmp_path):
    path = render_preview(build("S1", 0), tmp_path / "s1.png")
    assert path.is_file()
    assert path.stat().st_size > 0
