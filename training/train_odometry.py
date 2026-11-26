"""Train OdoNet, the learned odometry corrector used by the agent's spatial memory.

Data: random-ish and wall-hugging trajectories on procedurally generated maps. Labels: the true
displacement from env.privileged_state() (training-time only). The network sees only what the agent
sees at evaluation: its own buttons, ego velocities and two consecutive ray scans.

    uv run python training/train_odometry.py --samples 120000 --workers 8
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "submission"), str(ROOT / "env")]
os.environ.setdefault("ALSOFT_DRIVERS", "null")


def _policy(rng, state):
    """Sticky random buttons, biased forward, with periods of wall hugging and strafing."""
    a = np.zeros(9, np.int8)
    if rng.random() < 0.08:
        state["turn"] = int(rng.integers(-1, 2))
    if rng.random() < 0.05:
        state["mode"] = int(rng.integers(0, 4))
    mode = state["mode"]
    a[0] = rng.random() < (0.9 if mode != 3 else 0.3)
    a[1] = rng.random() < (0.05 if mode != 3 else 0.4)
    if mode == 1 or rng.random() < 0.1:
        a[2 + int(rng.integers(2))] = 1
    if state["turn"] < 0:
        a[4] = 1
    elif state["turn"] > 0:
        a[5] = 1
    if mode == 2 and rng.random() < 0.5:
        a[4:6] = 0
    a[7] = rng.random() < 0.03
    return a


def collect(args):
    wid, n_samples, seed = args
    import torch
    from brain.memory import odometry_features
    from doom_env import make_env
    from levelgen import generate, write_level

    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("buttons", "mom", "vel", "turnheld", "prev_rays", "rays", "disp")}
    tmp = Path(f"/tmp/odo_maps_{os.getpid()}")
    k = 0
    while len(out["disp"]) < n_samples:
        tier = int(rng.choice([0, 1, 2, 3, 4], p=[0.15, 0.2, 0.25, 0.2, 0.2]))
        lv = generate(tier, 10_000 * wid + k)
        k += 1
        env = make_env(str(write_level(lv, tmp / "m.wad")), render_buffers=False)
        for ep in range(2):
            obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
            ps = env.privileged_state()
            mom = np.zeros(2, np.float32)
            th = 0.0
            state = {"turn": 0, "mode": 0}
            prev_rays = obs["rays"]["distance"].copy()
            for t in range(int(rng.integers(150, 500))):
                a = _policy(rng, state)
                obs, _, term, trunc, _ = env.step(a)
                if term or trunc:
                    break
                p2 = env.privileged_state()
                d = (p2["angle"] - ps["angle"] + 180) % 360 - 180
                mid = math.radians(ps["angle"] + d / 2)
                dx, dy = p2["x"] - ps["x"], p2["y"] - ps["y"]
                vel = obs["vitals"][14:16].astype(np.float32)
                out["buttons"].append(a.copy())
                out["mom"].append(mom.copy())
                out["vel"].append(vel)
                out["turnheld"].append(th)
                out["prev_rays"].append(prev_rays)
                out["rays"].append(obs["rays"]["distance"].copy())
                out["disp"].append([dx * math.cos(mid) + dy * math.sin(mid), dx * math.sin(mid) - dy * math.cos(mid)])
                f = odometry_features(torch.tensor(a[None]), torch.tensor(mom[None]), torch.tensor(vel[None]),
                                      torch.tensor([th], dtype=torch.float32))
                th = float(f["turnheld"][0])
                mom = vel
                prev_rays = obs["rays"]["distance"].copy()
                ps = p2
        env.close()
    return {k: np.asarray(v[:n_samples], np.float32) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=120_000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--out", default=str(ROOT / "submission/weights/odonet.pt"))
    ap.add_argument("--data", default="/tmp/odometry_data.npz", help="cache of collected samples")
    a = ap.parse_args()

    import torch
    from brain.memory import OdoNet, odometry_features, odonet_input

    if Path(a.data).exists():
        data = dict(np.load(a.data))
        print(f"loaded {len(data['disp'])} samples from {a.data}")
    else:
        t0 = time.time()
        per = a.samples // a.workers + 1
        with mp.get_context("spawn").Pool(a.workers) as pool:
            parts = pool.map(collect, [(w, per, 1234 + w) for w in range(a.workers)])
        data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        np.savez(a.data, **data)
        print(f"collected {len(data['disp'])} samples in {time.time() - t0:.0f}s")

    T = {k: torch.tensor(v) for k, v in data.items()}
    n = len(T["disp"])
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    te, tr = perm[: n // 10], perm[n // 10:]

    def batch(idx):
        f = odometry_features(T["buttons"][idx], T["mom"][idx], T["vel"][idx], T["turnheld"][idx])
        return odonet_input(T["prev_rays"][idx], T["rays"][idx], f, T["vel"][idx]), f["disp"], T["disp"][idx]

    net = OdoNet()
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    for it in range(a.iters):
        inp, phys, y = batch(torch.tensor(rng.choice(tr, 512)))
        loss = torch.nn.functional.smooth_l1_loss(phys + net(inp), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 1000 == 0 or it == a.iters - 1:
            with torch.no_grad():
                inp, phys, y = batch(torch.tensor(te))
                e_net = (phys + net(inp) - y).norm(dim=1)
                e_phy = (phys - y).norm(dim=1)
            print(f"it {it:5d} loss {loss.item():.3f} | test err/step: physics {e_phy.mean():.2f} "
                  f"OdoNet {e_net.mean():.2f} (p99 {e_net.quantile(0.99):.1f}) units")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
