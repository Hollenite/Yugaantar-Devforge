"""Tests for the agent's brain (memory, action mapping) and the training map generator."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]

from brain.memory import EgoMap  # noqa: E402
from brain.model import buttons_to_comps, comps_to_buttons  # noqa: E402


def _probe(heading_deg, ray):
    m = EgoMap(1)
    m.pose[0, 2] = math.radians(heading_deg)
    d = torch.ones(1, 64)
    h = torch.zeros(1, 64, dtype=torch.long)
    d[0, ray], h[0, ray] = 0.4, 1
    m.write(d, h, torch.zeros(1, 32, 5), torch.zeros(1, 32, dtype=torch.long), torch.zeros(1, 32, dtype=torch.long))
    c = m.read()[0, 1]
    return (c == c.max()).nonzero()[0].tolist()


@pytest.mark.parametrize("heading", [0, 90, -135])
def test_egomap_crop_is_heading_aligned(heading):
    ahead, left, right, behind = (_probe(heading, r) for r in (0, 16, 48, 32))
    assert ahead[0] < 10 and 13 <= ahead[1] <= 18
    assert left[1] < 10 and 13 <= left[0] <= 18
    assert right[1] > 22 and 13 <= right[0] <= 18
    assert behind[0] > 22


def test_turn_odometry_matches_doom():
    m = EgoMap(1)
    b = torch.zeros(1, 9, dtype=torch.long)
    b[0, 4] = 1  # turn left
    seen = []
    for _ in range(3):
        before = m.pose[0, 2].item()
        m.move(b, torch.zeros(1, 2))
        seen.append(math.degrees(m.pose[0, 2].item() - before))
    assert seen == pytest.approx([7.03, 12.30, 14.06], abs=0.01)


def test_action_mapping_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        comps = torch.tensor([[rng.integers(3), rng.integers(3), rng.integers(3), rng.integers(2), rng.integers(2), rng.integers(2)]])
        b = comps_to_buttons(comps)
        assert set(b.flatten().tolist()) <= {0, 1}
        assert torch.equal(buttons_to_comps(b), comps)


@pytest.mark.parametrize("tier", [0, 1, 2, 3, 4])
def test_generated_levels_load(tmp_path, tier):
    from doom_env import make_env
    from mapgen import generate, write_level

    lv = generate(tier, 3)
    env = make_env(str(write_level(lv, tmp_path / "m.wad")), render_buffers=False)
    obs, _ = env.reset(seed=0)
    ps = env.privileged_state()
    assert (int(ps["y"] // 32), int(ps["x"] // 32)) == lv.start
    locked = sum(1 for v in env.mapinfo.door_sectors.values() if v)
    assert locked == len(lv.lock_order)
    env.close()
