"""Evaluate the submission agent (exactly as the scorer would call it) on held-out maps.

Reports scorer-like metrics per tier: exit rate, keys, doors opened, kills, unique 64x64 areas
visited, deaths. Held-out maps use generator seeds from a range training never samples from
(training draws seeds at random; evaluation uses 10**12 + i).

    uv run python training/evaluate.py --episodes 5                 # submission/agent.py
    uv run python training/evaluate.py --policy teacher             # privileged teacher (upper bound)
    uv run python training/evaluate.py --policy random              # template baseline
    uv run python training/evaluate.py --wads maps/two_rooms.wad    # extra WADs
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "env")]
os.environ.setdefault("ALSOFT_DRIVERS", "null")

HELD_OUT_SEED = 10 ** 12


def load_agent():
    spec = importlib.util.spec_from_file_location("submission_agent", ROOT / "submission/agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from doom_env.spaces import action_space, observation_space

    return mod.DoomAgent(observation_space(), action_space(), "cpu")


def run_episode(env, policy, nav=None, seed=0, max_ticks=None):
    import torch

    from denv import Teacher

    obs, _ = env.reset(seed=seed)
    ps = env.privileged_state()
    teacher = Teacher(nav, np.random.default_rng(seed)) if policy == "teacher" else None
    rng = np.random.default_rng(seed)
    if policy not in ("teacher", "random"):
        policy.reset()
    visited, doors = set(), set()
    k0, d0 = ps["kills"], ps["damage_taken"]
    while True:
        if policy == "teacher":
            from brain.model import comps_to_buttons

            a = comps_to_buttons(torch.tensor(teacher.act(ps, obs)[None]))[0].numpy()
        elif policy == "random":
            a = rng.integers(0, 2, 9)
        else:
            a = policy.act(obs)
        obs, _, term, trunc, _ = env.step(a)
        ps = env.privileged_state()
        visited.add((int(ps["x"] // 64), int(ps["y"] // 64)))
        if ps["door_lock"] is not None:
            for sid in np.nonzero((ps["door_lock"] >= 0) & (ps["sector_ceiling"] - ps["sector_floor"] > 56))[0]:
                doors.add(int(sid))
        if term or trunc:
            break
    return dict(exit=int(ps["exit_reached"]), death=int(ps["dead"]), keys=int(ps["keys"].sum()), doors=len(doors),
                kills=ps["kills"] - k0, damage=ps["damage_taken"] - d0, areas=len(visited), tics=ps["tic"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="agent", choices=["agent", "teacher", "random"])
    ap.add_argument("--tiers", default="0,1,2,3,4")
    ap.add_argument("--episodes", type=int, default=5, help="maps per tier")
    ap.add_argument("--ticks", type=int, default=0, help="episode tick cap (0 = 2100 for tier 0-1, 4200 otherwise)")
    ap.add_argument("--wads", nargs="*", default=[])
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT / "submission"))
    from denv import LevelNav, load_wad_pool
    from doom_env import make_env
    from levelgen import generate, write_level

    policy = load_agent() if a.policy == "agent" else a.policy
    rows = defaultdict(list)
    tmp = Path(f"/tmp/doom_eval_{os.getpid()}")
    for tier in [int(t) for t in a.tiers.split(",") if t != ""]:
        for i in range(a.episodes):
            lv = generate(tier, HELD_OUT_SEED + i)
            ticks = a.ticks or (2100 if tier <= 1 else 4200)
            env = make_env(str(write_level(lv, tmp / "e.wad")), render_buffers=False, episode_ticks=ticks)
            nav = LevelNav.build(lv, set(env.mapinfo.door_sectors))
            r = run_episode(env, policy, nav, seed=i)
            env.close()
            rows[f"tier{tier}"].append(r)
            print(f"tier {tier} map {i}: {r}", flush=True)
    for wad, mp in load_wad_pool(a.wads):
        if a.policy == "teacher":
            continue
        env = make_env(wad, mp, render_buffers=False, episode_ticks=a.ticks or 4200)
        r = run_episode(env, policy, None, seed=0)
        env.close()
        rows[f"{Path(wad).name}:{mp}"].append(r)
        print(f"{wad}:{mp}: {r}", flush=True)

    print(f"\n=== {a.policy} ===")
    keys = ("exit", "death", "keys", "doors", "kills", "areas", "damage")
    print(f"{'set':<22}" + "".join(f"{k:>9}" for k in keys))
    for name, rs in rows.items():
        print(f"{name:<22}" + "".join(f"{np.mean([r[k] for r in rs]):9.2f}" for k in keys))


if __name__ == "__main__":
    main()
