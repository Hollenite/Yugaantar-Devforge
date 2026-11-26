"""Recurrent PPO + privileged-teacher distillation, written from scratch for DoomNet.

Pipeline per update:
  1. Roll out T steps in N parallel envs (multiprocessing workers, several envs each).
     The agent's perception (EgoMap + OdoNet) runs batched in the main process, exactly as in
     DoomAgent. With probability beta the teacher's action is executed instead of the policy's
     (DAgger mixing), so the student also visits states it would reach on its own.
  2. Rewards = shaped extrinsic reward (denv.TrainEnv) + map-coverage curiosity: the number of new
     cells the agent's OWN EgoMap revealed this step (no privileged information).
  3. GAE(lambda) advantages; PPO clipped objective over BPTT chunks of length L with stored GRU
     states; entropy bonus; value loss; plus an imitation loss (cross-entropy to the teacher's
     action) whose weight anneals from il_start to il_end.
  4. Curriculum: tier sampling weights follow per-tier exit rates.

    uv run python training/ppo.py --run first --workers 8 --envs-per-worker 4 --updates 3000
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "submission"), str(ROOT / "env")]
os.environ.setdefault("ALSOFT_DRIVERS", "null")


# ---------------------------------------------------------------------------------- workers
def worker_main(conn, wid, n_envs, seed, cfg):
    import torch

    torch.set_num_threads(1)
    from brain.model import flatten_obs
    from denv import TrainEnv, load_wad_pool

    pool = load_wad_pool(cfg.get("wads", [])) if cfg.get("wads") else []
    envs = [TrainEnv(wid * 100 + i, seed + 7919 * i, tier_weights=cfg["tiers"], wad_pool=pool,
                     wad_prob=cfg.get("wad_prob", 0.0), episodes_per_level=cfg.get("episodes_per_level", 2))
            for i in range(n_envs)]

    def pack(i, obs, r, done, info):
        t, ok = envs[i].teacher_action()
        return flatten_obs(obs), r, done, t, ok, info

    while True:
        cmd, data = conn.recv()
        if cmd == "reset":
            conn.send([pack(i, e.reset(), 0.0, True, None) for i, e in enumerate(envs)])
        elif cmd == "step":
            out = []
            for i, e in enumerate(envs):
                obs, r, done, _ = e.step(data[i])
                info = None
                if done:
                    info = dict(e.ep)
                    obs = e.reset()
                out.append(pack(i, obs, r, done, info))
            conn.send(out)
        elif cmd == "tiers":
            for e in envs:
                e.set_tier_weights(data)
            conn.send(True)
        elif cmd == "close":
            for e in envs:
                e.close()
            conn.send(True)
            break


class VecEnv:
    def __init__(self, workers, envs_per_worker, seed, cfg):
        ctx = mp.get_context("spawn")
        self.pipes, self.procs = [], []
        for w in range(workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=worker_main, args=(b, w, envs_per_worker, seed + 1000 * w, cfg), daemon=True)
            p.start()
            self.pipes.append(a)
            self.procs.append(p)
        self.k = envs_per_worker
        self.n = workers * envs_per_worker

    def _gather(self):
        out = []
        for p in self.pipes:
            out += p.recv()
        return out

    def reset(self):
        for p in self.pipes:
            p.send(("reset", None))
        return self._gather()

    def step(self, buttons):
        for w, p in enumerate(self.pipes):
            p.send(("step", buttons[w * self.k:(w + 1) * self.k]))
        return self._gather()

    def set_tiers(self, w):
        for p in self.pipes:
            p.send(("tiers", list(w)))
        for p in self.pipes:
            p.recv()

    def close(self):
        for p in self.pipes:
            try:
                p.send(("close", None))
                p.recv()
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)


# ---------------------------------------------------------------------------------- trainer
def linear(start, end, frac):
    return start + (end - start) * min(max(frac, 0.0), 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="run")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--envs-per-worker", type=int, default=4)
    ap.add_argument("--steps", type=int, default=128, help="rollout length T")
    ap.add_argument("--bptt", type=int, default=32)
    ap.add_argument("--updates", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--curiosity", type=float, default=0.002, help="reward per newly revealed EgoMap cell")
    ap.add_argument("--beta-start", type=float, default=0.5, help="prob. of executing the teacher action")
    ap.add_argument("--beta-updates", type=int, default=400)
    ap.add_argument("--beta-end", type=float, default=0.0, help="floor of the per-step teacher mixing")
    ap.add_argument("--rollin-prob", type=float, default=0.0,
                    help="fraction of episodes where the teacher drives the first K steps, then hands over to the "
                         "policy (gives the policy practice in late-level states: holding a key, near a locked door)")
    ap.add_argument("--rollin-max", type=int, default=450, help="K ~ U(20, rollin-max) decisions")
    ap.add_argument("--solo-prob", type=float, default=0.0,
                    help="fraction of episodes the policy plays entirely alone (no teacher mixing): practice + honest metrics")
    ap.add_argument("--sched-start", type=int, default=0, help="update at which the beta/il schedules start")
    ap.add_argument("--tier-max", type=float, default=0.6, help="cap on any tier's share of training levels")
    ap.add_argument("--tier-min", type=float, default=0.05)
    ap.add_argument("--il-start", type=float, default=1.0)
    ap.add_argument("--il-end", type=float, default=0.05)
    ap.add_argument("--il-updates", type=int, default=1500)
    ap.add_argument("--tiers", default="0.2,0.2,0.3,0.2,0.1", help="initial tier weights 0..4")
    ap.add_argument("--wads", nargs="*", default=[], help="extra WADs ('file.wad' or 'file.wad:MAP03')")
    ap.add_argument("--wad-prob", type=float, default=0.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--mem-device", default="cpu", help="device for the EgoMap (cpu is safest on MPS)")
    ap.add_argument("--odonet", default=str(ROOT / "submission/weights/odonet.pt"))
    ap.add_argument("--resume", default="")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--export", default=str(ROOT / "submission/weights/policy.pt"))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    import torch
    from brain.memory import OdoNet
    from brain.model import (DoomNet, Perception, comps_to_buttons, count_params, entropy, imitation_loss, log_prob,
                             sample, stack_obs)

    torch.manual_seed(a.seed)
    if a.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        dev = a.device
    device, mdev = torch.device(dev), torch.device(a.mem_device)
    run_dir = ROOT / "training/runs" / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(a), indent=2))

    net = DoomNet().to(device)
    odonet = None
    if a.odonet and Path(a.odonet).exists():
        odonet = OdoNet().to(mdev)
        odonet.load_state_dict(torch.load(a.odonet, map_location=mdev))
        odonet.eval()
        print("using OdoNet", a.odonet)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
    start_update = 0
    tier_w = np.array([float(x) for x in a.tiers.split(",")])
    if a.resume:
        ck = torch.load(a.resume, map_location=device, weights_only=False)  # our own checkpoint
        net.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
            for g in opt.param_groups:  # use this run's --lr, not the one saved in the checkpoint
                g["lr"] = a.lr
        start_update = ck.get("update", 0)
        tier_w = np.array(ck.get("tier_weights", tier_w))
        print("resumed from", a.resume, "at update", start_update)
    print(f"DoomNet params: {count_params(net):,}  device={device} mem-device={mdev}")

    cfg = dict(tiers=list(tier_w), wads=a.wads, wad_prob=a.wad_prob)
    venv = VecEnv(a.workers, a.envs_per_worker, a.seed * 100_000, cfg)
    N, T, L = venv.n, a.steps, a.bptt
    assert T % L == 0
    per = Perception(N, mdev, odonet)

    def unpack(items):
        obs = stack_obs([it[0] for it in items], mdev)
        rew = torch.tensor([it[1] for it in items], dtype=torch.float32)
        done = torch.tensor([it[2] for it in items], dtype=torch.bool)
        tch = torch.tensor(np.stack([it[3] for it in items]), dtype=torch.long)
        tok = torch.tensor([it[4] for it in items], dtype=torch.bool)
        infos = [(i, it[5]) for i, it in enumerate(items) if it[5] is not None]
        return obs, rew, done, tch, tok, infos

    obs, _, fresh, tch, tok, _ = unpack(venv.reset())
    h = net.initial_state(N, device)
    prev_comps = torch.zeros(N, 6, dtype=torch.long)
    last_buttons = torch.zeros(N, 9, dtype=torch.long, device=mdev)
    maps = per.observe(obs, None, fresh)

    stats = defaultdict(lambda: deque(maxlen=200))
    tier_exit = {t: deque(maxlen=100) for t in range(-1, 5)}
    ret_rms = [0.0, 1.0, 1e-4]  # mean, var, count of discounted returns (reward scaling)
    disc_ret = torch.zeros(N)
    log_f = open(run_dir / "log.csv", "a", newline="")
    logw = csv.writer(log_f)
    gen = torch.Generator().manual_seed(a.seed)

    def new_rollin(n):
        k = torch.randint(20, max(21, a.rollin_max), (n,), generator=gen)
        return torch.where(torch.rand(n, generator=gen) < a.rollin_prob, k, torch.zeros_like(k))

    rollin = new_rollin(N)              # teacher-driven decisions left at the start of each episode
    solo_ep = (torch.rand(N, generator=gen) < a.solo_prob) & (rollin == 0)  # whole episode without the teacher
    ep_teacher = torch.zeros(N, dtype=torch.bool)  # did the teacher act at all in this episode?
    best_score = -1.0

    for update in range(start_update, a.updates):
        t_start = time.time()
        su = update - a.sched_start
        beta = linear(a.beta_start, a.beta_end, su / max(1, a.beta_updates))
        il_coef = linear(a.il_start, a.il_end, su / max(1, a.il_updates))

        buf = defaultdict(list)
        h0_store = []
        net.eval()
        for t in range(T):
            if t % L == 0:
                h0_store.append(h.clone())
            with torch.no_grad():
                obs_d = {k: v.to(device) for k, v in obs.items()}
                start = fresh.to(device)
                h = h * (~start).float().unsqueeze(1)
                pc = prev_comps * (~fresh).long().unsqueeze(1)
                logits, value, h = net.step(obs_d, maps.to(device), pc.to(device), h)
                pol = sample(logits.cpu(), generator=gen)
                use_t = (((torch.rand(N, generator=gen) < beta) & ~solo_ep) | (rollin > 0)) & tok
                exec_c = torch.where(use_t.unsqueeze(1), tch, pol)
                lp = log_prob(logits.cpu(), exec_c)
            for k, v in obs.items():
                buf["obs_" + k].append(v.cpu())
            buf["maps"].append((maps.clamp(0, 1) * 255).to(torch.uint8).cpu())
            buf["prev"].append(pc)
            buf["act"].append(exec_c)
            buf["logp"].append(lp)
            buf["val"].append(value.cpu())
            buf["start"].append(fresh.clone())
            buf["tch"].append(tch)
            buf["tok"].append(tok)
            buf["by_teacher"].append(use_t)

            buttons = comps_to_buttons(exec_c)
            out = venv.step(buttons.numpy())
            obs, rew, done, tch, tok, infos = unpack(out)
            maps = per.observe(obs, buttons.to(mdev), done)
            r_int = per.mem.new_free.cpu() * a.curiosity * (~done).float()
            buf["rew"].append(rew + r_int)
            buf["done"].append(done)
            stats["r_int"].append(float(r_int.mean()))
            fresh = done
            prev_comps = exec_c
            ep_teacher |= use_t
            rollin = (rollin - 1).clamp_min(0)
            for i, inf in infos:
                tier_exit[inf["tier"]].append(inf["exit"])
                for k in ("ret", "steps", "kills", "keys", "doors", "explore", "exit", "death"):
                    stats[k].append(inf[k])
                stats[f"exit_t{inf['tier']}"].append(inf["exit"])
                if not ep_teacher[i]:  # the policy played this whole episode alone: the honest number
                    stats["solo_exit"].append(inf["exit"])
                    stats["solo_death"].append(inf["death"])
                    stats[f"solo_t{inf['tier']}"].append(inf["exit"])
            if done.any():
                ep_teacher[done] = False
                nd = int(done.sum())
                rollin[done] = new_rollin(nd)
                solo_ep[done] = (torch.rand(nd, generator=gen) < a.solo_prob) & (rollin[done] == 0)

        # ---- returns / advantages
        with torch.no_grad():
            obs_d = {k: v.to(device) for k, v in obs.items()}
            hh = h * (~fresh.to(device)).float().unsqueeze(1)
            pc = prev_comps * (~fresh).long().unsqueeze(1)
            _, last_v, _ = net.step(obs_d, maps.to(device), pc.to(device), hh)
            last_v = last_v.cpu()
        B = {k: torch.stack(v) for k, v in buf.items()}
        # reward scaling by the running std of discounted returns
        for t in range(T):
            disc_ret = disc_ret * a.gamma * (~B["done"][t]).float() + B["rew"][t]
            x = disc_ret
            bm, bv, bc = float(x.mean()), float(x.var(unbiased=False)), x.numel()
            m, v, c = ret_rms
            tot = c + bc
            delta = bm - m
            ret_rms = [m + delta * bc / tot, (v * c + bv * bc + delta ** 2 * c * bc / tot) / tot, tot]
        scale = float(np.sqrt(ret_rms[1]) + 1e-6)
        rew = B["rew"] / scale
        adv = torch.zeros(T, N)
        gae = torch.zeros(N)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else B["val"][t + 1]
            nonterm = (~B["done"][t]).float()
            delta = rew[t] + a.gamma * nv * nonterm - B["val"][t]
            gae = delta + a.gamma * a.lam * nonterm * gae
            adv[t] = gae
        ret = adv + B["val"]

        # ---- PPO update over BPTT chunks
        net.train()
        n_chunks = T // L
        chunk_ids = [(c, e) for c in range(n_chunks) for e in range(N)]
        h0_all = torch.stack(h0_store)  # (n_chunks, N, H)
        mb_size = max(1, len(chunk_ids) // a.minibatches)
        losses = defaultdict(list)
        for epoch in range(a.epochs):
            perm = np.random.permutation(len(chunk_ids))
            for i in range(0, len(perm), mb_size):
                sel = [chunk_ids[j] for j in perm[i:i + mb_size]]
                cs = torch.tensor([s[0] for s in sel])
                es = torch.tensor([s[1] for s in sel])
                tt = (cs.unsqueeze(0) * L + torch.arange(L).unsqueeze(1))  # (L, M)
                ee = es.unsqueeze(0).expand(L, -1)

                def g(x):
                    return x[tt, ee].to(device)

                o = {k[4:]: g(B[k]) for k in B if k.startswith("obs_")}
                logits, value = net.unroll(o, g(B["maps"]).float() / 255, g(B["prev"]), h0_all[cs, es].to(device), g(B["start"]))
                act, old_lp, A, R = g(B["act"]), g(B["logp"]), g(adv), g(ret)
                pol_mask = (~g(B["by_teacher"])).float()
                A_n = (A - A.mean()) / (A.std() + 1e-8)
                lp = log_prob(logits, act)
                ratio = torch.exp(lp - old_lp)
                pg = -torch.min(ratio * A_n, ratio.clamp(1 - a.clip, 1 + a.clip) * A_n)
                pg = (pg * pol_mask).sum() / pol_mask.sum().clamp_min(1)
                v_loss = 0.5 * (value - R).pow(2).mean()
                ent = entropy(logits).mean()
                tmask = g(B["tok"]).float()
                il = (imitation_loss(logits, g(B["tch"])) * tmask).sum() / tmask.sum().clamp_min(1)
                loss = pg + a.vf * v_loss - a.ent * ent + il_coef * il
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    losses["pg"].append(pg.item())
                    losses["v"].append(v_loss.item())
                    losses["ent"].append(ent.item())
                    losses["il"].append(il.item())
                    losses["kl"].append((((ratio - 1) - torch.log(ratio)) * pol_mask).sum().item() / max(pol_mask.sum().item(), 1))

        # ---- curriculum: more weight on tiers that are being learned, less on solved ones
        rates = {t: (np.mean(tier_exit[t]) if len(tier_exit[t]) >= 20 else None) for t in range(5)}
        new_w = tier_w.copy()
        for t, r in rates.items():
            if r is not None:
                new_w[t] = tier_w[t] * (0.9 if r > 0.85 else 1.1 if r < 0.6 else 1.0)
        for _ in range(5):  # clip to [tier_min, tier_max] and renormalise (a few passes converge)
            new_w = np.clip(new_w / new_w.sum(), a.tier_min, a.tier_max)
        new_w /= new_w.sum()
        if np.abs(new_w - tier_w).max() > 0.01:
            tier_w = new_w
            venv.set_tiers(tier_w)

        fps = N * T / (time.time() - t_start)
        m = {k: float(np.mean(v)) if len(v) else float("nan") for k, v in stats.items()}
        lm = {k: float(np.mean(v)) for k, v in losses.items()}
        line = (f"upd {update:5d} fps {fps:5.0f} beta {beta:.2f} il {il_coef:.2f} | ret {m.get('ret', 0):6.2f} "
                f"exit {m.get('exit', 0):.2f} death {m.get('death', 0):.2f} kills {m.get('kills', 0):.2f} "
                f"keys {m.get('keys', 0):.2f} doors {m.get('doors', 0):.2f} expl {m.get('explore', 0):5.1f} | "
                + " ".join(f"t{t}:{m.get(f'exit_t{t}', float('nan')):.2f}" for t in range(5))
                + f" | solo {m.get('solo_exit', float('nan')):.2f} "
                + " ".join(f"s{t}:{m.get(f'solo_t{t}', float('nan')):.2f}" for t in range(1, 5))
                + f" | pg {lm['pg']:.3f} v {lm['v']:.3f} ent {lm['ent']:.2f} il {lm['il']:.3f} kl {lm['kl']:.4f}")
        print(line, flush=True)
        logw.writerow([update, fps, beta, il_coef, *[m.get(k, float('nan')) for k in
                       ("ret", "exit", "death", "kills", "keys", "doors", "explore")],
                       *[m.get(f"exit_t{t}", float("nan")) for t in range(5)], *[lm[k] for k in ("pg", "v", "ent", "il", "kl")],
                       *tier_w, m.get("solo_exit", float("nan")), m.get("solo_death", float("nan")),
                       *[m.get(f"solo_t{t}", float("nan")) for t in range(5)]])
        log_f.flush()

        if (update + 1) % a.save_every == 0 or update == a.updates - 1:
            ck = {"model": net.state_dict(), "opt": opt.state_dict(), "update": update + 1, "tier_weights": [float(x) for x in tier_w],
                  "args": vars(a)}
            torch.save(ck, run_dir / "latest.pt")
            if (update + 1) % (a.save_every * 4) == 0:
                torch.save(ck, run_dir / f"ckpt_{update + 1:05d}.pt")
            export(net, odonet, a.export)
            # keep the policy that scores best when playing ALONE (tier-weighted like the evaluation)
            solo = {t: stats.get(f"solo_t{t}") for t in range(1, 5)}
            if all(v is not None and len(v) >= 15 for v in solo.values()):
                score = sum(t * float(np.mean(solo[t])) for t in range(1, 5)) / 10
                if score > best_score:
                    best_score = score
                    export(net, odonet, run_dir / "best_policy.pt")
                    torch.save(ck, run_dir / "best.pt")
                    print(f"new best solo score {score:.3f} at update {update + 1} -> {run_dir / 'best_policy.pt'}", flush=True)

    venv.close()


def export(net, odonet, path):
    """Write the inference checkpoint DoomAgent loads (weights only, CPU tensors)."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "odonet": None if odonet is None else {k: v.detach().cpu() for k, v in odonet.state_dict().items()}},
               path)


if __name__ == "__main__":
    main()
