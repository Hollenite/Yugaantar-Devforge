"""Pick the best policy checkpoint by playing them all ALONE on the same unseen maps.

Every checkpoint plays exactly the same episodes (procedural maps from a held-out seed range that
training never samples, plus real Freedoom 2 maps), with the same sampling seed, through
submission/agent.py exactly as the evaluation harness calls it. Each episode gets a scorer-like
score (exit dominates; keys, doors, kills and explored area give partial credit; death and damage
cost), weighted by tier like the hidden suite. The winner can be installed as
submission/weights/policy.pt.

    uv run python training/select_best.py --ckpts "training/runs/*/ckpt_*.pt" --install
    uv run python training/select_best.py --ckpts a.pt b.pt --episodes 4 --workers 4

Used by ppo.py (--eval-every) to keep training/runs/<run>/best_eval_policy.pt up to date.
"""

from __future__ import annotations

import argparse
import csv
import glob
import multiprocessing as mp
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "env"), str(ROOT / "submission")]
os.environ.setdefault("ALSOFT_DRIVERS", "null")

SELECT_SEED = 10 ** 12 + 10_000  # disjoint from training seeds and from evaluate.py's report maps
TIER_W = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5}
WAD_W = 1.5


def episode_score(r: dict) -> float:
    return (10.0 * r["exit"] + 1.5 * r["keys"] + 0.5 * r["doors"] + 0.3 * r["kills"] + 0.03 * r["areas"]
            - 2.0 * r["death"] - 0.01 * r["damage"])


def freedoom_iwad() -> str:
    import vizdoom

    return os.path.join(os.path.dirname(vizdoom.__file__), "freedoom2.wad")


_AGENTS: dict = {}


def _agent(ckpt: str):
    if ckpt not in _AGENTS:
        os.environ["DOOM_AGENT_POLICY"] = ckpt
        from evaluate import load_agent

        _AGENTS.clear()  # one checkpoint at a time per worker keeps memory flat
        _AGENTS[ckpt] = load_agent()
    return _AGENTS[ckpt]


def run_task(task):
    import torch

    torch.set_num_threads(1)
    from denv import LevelNav
    from doom_env import make_env
    from evaluate import run_episode
    from levelgen import generate, write_level

    ckpt, kind, a, b = task
    agent = _agent(ckpt)
    agent.gen.manual_seed(1234 + (b if kind == "gen" else 0))
    if kind == "gen":
        tier, i = a, b
        lv = generate(tier, SELECT_SEED + i)
        tmp = Path(f"/tmp/doom_select_{os.getpid()}")
        env = make_env(str(write_level(lv, tmp / "s.wad")), render_buffers=False,
                       episode_ticks=2100 if tier <= 1 else 4200)
        nav = LevelNav.build(lv, set(env.mapinfo.door_sectors))
        r = run_episode(env, agent, nav, seed=i)
        key = f"tier{tier}"
    else:
        env = make_env(freedoom_iwad(), a, render_buffers=False, episode_ticks=4200)
        r = run_episode(env, agent, None, seed=0)
        key = f"fd:{a}"
    env.close()
    return ckpt, key, r


def evaluate(ckpts, episodes=8, tiers=(0, 1, 2, 3, 4), maps=("MAP01", "MAP02", "MAP03"), workers=4):
    tasks = []
    for c in ckpts:
        tasks += [(c, "gen", t, i) for t in tiers for i in range(episodes)]
        tasks += [(c, "wad", m, 0) for m in maps]
    res = defaultdict(lambda: defaultdict(list))
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=200) as pool:
        for n, (c, key, r) in enumerate(pool.imap_unordered(run_task, tasks, chunksize=1), 1):
            res[c][key].append(r)
            if n % 25 == 0 or n == len(tasks):
                print(f"  {n}/{len(tasks)} episodes  ({time.time() - t0:.0f}s)", flush=True)
    out = {}
    for c in ckpts:
        per = {}
        tot, wsum = 0.0, 0.0
        for key, rs in res[c].items():
            s = sum(episode_score(r) for r in rs) / len(rs)
            w = TIER_W[int(key[4:])] if key.startswith("tier") else WAD_W
            per[key] = dict(score=s, exit=sum(r["exit"] for r in rs) / len(rs),
                            death=sum(r["death"] for r in rs) / len(rs))
            tot += w * s
            wsum += w
        out[c] = dict(score=tot / max(wsum, 1e-9), per=per)
    return out


def export_policy(src: str, dst: Path):
    """Write the inference file DoomAgent loads ({'model', 'odonet'}) from any checkpoint format."""
    import torch

    ck = torch.load(src, map_location="cpu", weights_only=False)
    od = ck.get("odonet")
    if od is None:
        od = torch.load(ROOT / "submission/weights/odonet.pt", map_location="cpu")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    torch.save({"model": {k: v.detach().cpu() for k, v in ck["model"].items()}, "odonet": od}, tmp)
    os.replace(tmp, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="+", required=True, help="files or quoted glob patterns")
    ap.add_argument("--episodes", type=int, default=8, help="procedural maps per tier")
    ap.add_argument("--maps", default="MAP01,MAP02,MAP03", help="Freedoom 2 maps ('' for none)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--install", action="store_true", help="copy the winner to submission/weights/policy.pt")
    ap.add_argument("--log", default="", help="append results to this CSV")
    ap.add_argument("--keep-best", default="", help="export the winner here if it beats every score in --log")
    ap.add_argument("--winner-file", default="", help="write the winning checkpoint's path here")
    a = ap.parse_args()

    ckpts = []
    for p in a.ckpts:
        ckpts += sorted(glob.glob(p)) if any(ch in p for ch in "*?[") else [p]
    ckpts = list(dict.fromkeys(str(Path(c).resolve()) for c in ckpts if Path(c).exists()))
    if not ckpts:
        sys.exit("no checkpoints found")
    maps = tuple(m for m in a.maps.split(",") if m)
    print(f"evaluating {len(ckpts)} checkpoint(s) x ({a.episodes} maps x 5 tiers + {len(maps)} Freedoom maps), "
          f"{a.workers} workers", flush=True)
    res = evaluate(ckpts, a.episodes, maps=maps, workers=a.workers)

    keys = [f"tier{t}" for t in range(5)] + [f"fd:{m}" for m in maps]
    print("\n" + f"{'checkpoint':<44}{'SCORE':>7}" + "".join(f"{k[-5:]:>8}" for k in keys) + "   (exit rate per set)")
    for c, v in sorted(res.items(), key=lambda kv: -kv[1]["score"]):
        name = str(Path(c).relative_to(ROOT)) if Path(c).is_relative_to(ROOT) else c
        print(f"{name[-44:]:<44}{v['score']:7.2f}" + "".join(f"{v['per'].get(k, {}).get('exit', float('nan')):8.2f}"
                                                         for k in keys))
    best = max(res, key=lambda c: res[c]["score"])
    print(f"\nbest: {best}  score {res[best]['score']:.2f}")
    if a.winner_file:
        Path(a.winner_file).write_text(best + "\n")

    prev_best = float("-inf")
    if a.log:
        log = Path(a.log)
        if log.exists():
            with open(log) as f:
                prev_best = max([float(r["score"]) for r in csv.DictReader(f)] or [float("-inf")])
        new = not log.exists()
        with open(log, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "ckpt", "score", *keys])
            for c, v in res.items():
                w.writerow([int(time.time()), c, f"{v['score']:.4f}",
                            *[f"{v['per'].get(k, {}).get('exit', float('nan')):.3f}" for k in keys]])
    if a.keep_best and res[best]["score"] > prev_best:
        export_policy(best, Path(a.keep_best))
        print(f"new best eval score {res[best]['score']:.2f} (previous {prev_best:.2f}) -> {a.keep_best}")
    if a.install:
        dst = ROOT / "submission/weights/policy.pt"
        if dst.exists():
            shutil.copy2(dst, ROOT / "training/runs/policy_before_select.pt")
        export_policy(best, dst)
        print(f"installed {best} -> {dst}")


if __name__ == "__main__":
    main()
