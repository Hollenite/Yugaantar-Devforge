"""DoomNet: the agent's neural network, plus the shared perception pipeline.

Architecture (all trained from scratch in training/ppo.py):

    rays (64 x [dist, 1-dist, hit one-hot])  -> circular Conv1d tower        -> 192
    visible entities (<= 32, masked set)     -> per-entity MLP + attention pool (2 queries) -> 128
    vitals (health, ammo, weapon, keys, vel) -> MLP                          -> 64
    EgoMap crops (2 scales x 8 channels)     -> Conv2d tower                 -> 256
    previous action (factorised one-hot)                                     -> 15
                         concat -> Linear 384 -> GRU 256 (recurrent memory)
                         -> 6 factorised action heads + value head

Factorised actions: movement {none, fwd, back} x strafe {none, left, right} x turn {none, left,
right} x attack x use x next-weapon. Every combination maps to a valid MultiBinary(9) action.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import NUM_CH, EgoMap, OdoNet

HEAD_SIZES = (3, 3, 3, 2, 2, 2)  # move, strafe, turn, attack, use, weapon
NUM_COMP = len(HEAD_SIZES)
ACT_ONEHOT = sum(HEAD_SIZES)
NUM_ENTITY_TYPES = 12
MAP_CH = 2 * NUM_CH


def comps_to_buttons(comps: torch.Tensor) -> torch.Tensor:
    """(B, 6) component indices -> (B, 9) {0,1} buttons."""
    B = comps.shape[0]
    b = torch.zeros(B, 9, dtype=torch.int64, device=comps.device)
    b[:, 0] = comps[:, 0] == 1
    b[:, 1] = comps[:, 0] == 2
    b[:, 2] = comps[:, 1] == 1
    b[:, 3] = comps[:, 1] == 2
    b[:, 4] = comps[:, 2] == 1
    b[:, 5] = comps[:, 2] == 2
    b[:, 6] = comps[:, 3]
    b[:, 7] = comps[:, 4]
    b[:, 8] = comps[:, 5]
    return b


def buttons_to_comps(b: torch.Tensor) -> torch.Tensor:
    """(B, 9) buttons -> (B, 6) components (opposite buttons pressed together count as none)."""
    b = b.long()
    move = torch.where((b[:, 0] == 1) & (b[:, 1] == 0), 1, torch.where((b[:, 1] == 1) & (b[:, 0] == 0), 2, 0))
    strafe = torch.where((b[:, 2] == 1) & (b[:, 3] == 0), 1, torch.where((b[:, 3] == 1) & (b[:, 2] == 0), 2, 0))
    turn = torch.where((b[:, 4] == 1) & (b[:, 5] == 0), 1, torch.where((b[:, 5] == 1) & (b[:, 4] == 0), 2, 0))
    return torch.stack([move, strafe, turn, b[:, 6], b[:, 7], b[:, 8]], 1)


def comps_onehot(comps: torch.Tensor) -> torch.Tensor:
    return torch.cat([F.one_hot(comps[:, i].long(), n).float() for i, n in enumerate(HEAD_SIZES)], 1)


# ---------------------------------------------------------------------------------- observations
OBS_KEYS = ("ray_d", "ray_h", "ent_f", "ent_t", "ent_m", "vitals")


def flatten_obs(obs: dict) -> dict:
    """Official observation dict -> compact numpy arrays (no rgb/depth)."""
    r, e = obs["rays"], obs["visible_entities"]
    return {
        "ray_d": np.asarray(r["distance"], np.float32),
        "ray_h": np.asarray(r["hit_type"], np.int8),
        "ent_f": np.asarray(e["features"], np.float32),
        "ent_t": np.asarray(e["entity_type"], np.int8),
        "ent_m": np.asarray(e["mask"], np.int8),
        "vitals": np.asarray(obs["vitals"], np.float32),
    }


def stack_obs(items: list[dict], device) -> dict:
    return {k: torch.as_tensor(np.stack([it[k] for it in items]), device=device) for k in OBS_KEYS}


class Perception:
    """Owns the EgoMap and the odometry state for a batch of episodes (B=1 inside DoomAgent)."""

    def __init__(self, batch: int, device, odonet: OdoNet | None = None):
        self.mem = EgoMap(batch, device)
        self.odonet = odonet
        self.prev_rays: torch.Tensor | None = None
        self.device = torch.device(device)

    def reset(self, idx=None):
        self.mem.reset(idx)

    @torch.no_grad()
    def observe(self, obs: dict, last_buttons: torch.Tensor | None, fresh: torch.Tensor | None = None) -> torch.Tensor:
        """Integrate one observation. `fresh` (B,) bool marks episodes that just (re)started: their
        memory is cleared and no motion is integrated. Returns the map crops (B, 16, 32, 32)."""
        if fresh is not None and bool(fresh.any()):
            self.mem.reset(torch.nonzero(fresh).flatten())
        if last_buttons is not None and self.prev_rays is not None:
            moving = torch.ones(obs["ray_d"].shape[0], dtype=torch.bool, device=self.device) if fresh is None else ~fresh
            if bool(moving.any()):
                pose, mom, th = self.mem.pose.clone(), self.mem.mom.clone(), self.mem.turnheld.clone()
                self.mem.move(last_buttons, obs["vitals"][:, 14:16], self.prev_rays, obs["ray_d"], self.odonet)
                keep = ~moving
                self.mem.pose[keep], self.mem.mom[keep], self.mem.turnheld[keep] = pose[keep], mom[keep], th[keep]
        self.mem.write(obs["ray_d"], obs["ray_h"], obs["ent_f"], obs["ent_t"], obs["ent_m"])
        self.prev_rays = obs["ray_d"].clone()
        return self.mem.read()


# ---------------------------------------------------------------------------------- network
class RayEncoder(nn.Module):
    def __init__(self, out=192):
        super().__init__()
        c = 48
        self.net = nn.Sequential(
            nn.Conv1d(8, c, 5, padding=2, padding_mode="circular"), nn.GELU(),
            nn.Conv1d(c, c, 5, stride=2, padding=2, padding_mode="circular"), nn.GELU(),
            nn.Conv1d(c, c, 5, stride=2, padding=2, padding_mode="circular"), nn.GELU(),
        )
        self.fc = nn.Linear(c * 16, out)

    def forward(self, d, h):
        x = torch.cat([d.unsqueeze(1), (1 - d).pow(4).unsqueeze(1), F.one_hot(h.long().clamp(0, 5), 6).permute(0, 2, 1).float()], 1)
        return F.gelu(self.fc(self.net(x).flatten(1)))


class EntityEncoder(nn.Module):
    """Permutation-invariant encoder for a variable number of entities (masked attention pooling)."""

    def __init__(self, d=64, out=128, queries=2, heads=4):
        super().__init__()
        self.emb = nn.Embedding(NUM_ENTITY_TYPES, 16)
        self.mlp = nn.Sequential(nn.Linear(5 + 16 + 2, d), nn.GELU(), nn.Linear(d, d))
        self.q = nn.Parameter(torch.randn(queries, d) * 0.1)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.fc = nn.Linear(queries * d + NUM_ENTITY_TYPES, out)

    def forward(self, f, t, m):
        dist = f[..., 0:1].clamp(0, 4)
        feats = torch.cat([dist, f[..., 1:3], (f[..., 3:5] / 10).clamp(-3, 3), self.emb(t.long().clamp(0, NUM_ENTITY_TYPES - 1)),
                           (1 - dist / 4).pow(4), (f[..., 2:3] > 0.995).float()], -1)  # last: "in crosshair"
        x = self.mlp(feats)
        mask = m.bool()
        empty = ~mask.any(1)
        key_pad = ~mask
        key_pad[empty, 0] = False  # avoid all-masked rows (NaNs); the slot is zero-padded anyway
        q = self.q.unsqueeze(0).expand(x.shape[0], -1, -1)
        pooled, _ = self.attn(q, x, x, key_padding_mask=key_pad, need_weights=False)
        pooled = pooled * (~empty).float().view(-1, 1, 1)
        counts = (F.one_hot(t.long().clamp(0, NUM_ENTITY_TYPES - 1), NUM_ENTITY_TYPES).float() * m.unsqueeze(-1).float()).sum(1)
        return F.gelu(self.fc(torch.cat([pooled.flatten(1), counts.clamp(0, 8) / 4], 1)))


def vitals_features(v):
    w = v[:, 2].long().clamp(0, 7)
    return torch.cat([
        v[:, 0:1] / 100, v[:, 1:2] / 100, F.one_hot(w, 8).float(), (v[:, 3:4] / 50).clamp(0, 4),
        v[:, 4:8] / torch.tensor([200.0, 50.0, 50.0, 300.0], device=v.device), v[:, 8:14],
        (v[:, 14:17] / 8).clamp(-3, 3), (v[:, 3:4] <= 0).float(),
    ], 1)  # 25


class MapEncoder(nn.Module):
    def __init__(self, out=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(MAP_CH, 32, 3, stride=2, padding=1), nn.GELU(),  # 16x16
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),  # 8x8
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(),  # 4x4
        )
        self.fc = nn.Linear(64 * 16, out)

    def forward(self, m):
        return F.gelu(self.fc(self.net(m).flatten(1)))


class DoomNet(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.rays = RayEncoder()
        self.ents = EntityEncoder()
        self.vit = nn.Sequential(nn.Linear(25, 64), nn.GELU())
        self.map = MapEncoder()
        self.fuse = nn.Sequential(nn.Linear(192 + 128 + 64 + 256 + ACT_ONEHOT, 384), nn.GELU(), nn.Linear(384, hidden), nn.GELU())
        self.gru = nn.GRUCell(hidden, hidden)
        self.pi = nn.Sequential(nn.Linear(hidden * 2, 256), nn.GELU(), nn.Linear(256, ACT_ONEHOT))
        self.v = nn.Sequential(nn.Linear(hidden * 2, 256), nn.GELU(), nn.Linear(256, 1))
        self.hidden = hidden
        nn.init.orthogonal_(self.pi[-1].weight, 0.01)
        nn.init.zeros_(self.pi[-1].bias)

    def initial_state(self, batch, device):
        return torch.zeros(batch, self.hidden, device=device)

    def encode(self, obs, maps, prev_comps):
        return self.fuse(torch.cat([
            self.rays(obs["ray_d"], obs["ray_h"]),
            self.ents(obs["ent_f"], obs["ent_t"], obs["ent_m"]),
            self.vit(vitals_features(obs["vitals"])),
            self.map(maps.float()),
            comps_onehot(prev_comps),
        ], 1))

    def heads(self, x, h):
        z = torch.cat([x, h], 1)
        return self.pi(z), self.v(z).squeeze(1)

    def step(self, obs, maps, prev_comps, h):
        """One decision for a batch: returns logits (B, 15), value (B,), new hidden (B, H)."""
        x = self.encode(obs, maps, prev_comps)
        h = self.gru(x, h)
        logits, value = self.heads(x, h)
        return logits, value, h

    def unroll(self, obs, maps, prev_comps, h0, starts):
        """Training unroll over (T, B, ...) tensors. starts (T, B): 1 where an episode starts at t
        (the hidden state is zeroed before that step)."""
        T, B = starts.shape
        flat = {k: v.reshape(T * B, *v.shape[2:]) for k, v in obs.items()}
        x = self.encode(flat, maps.reshape(T * B, *maps.shape[2:]), prev_comps.reshape(T * B, -1)).view(T, B, -1)
        h, hs = h0, []
        for t in range(T):
            h = h * (1 - starts[t].float()).unsqueeze(1)
            h = self.gru(x[t], h)
            hs.append(h)
        hs = torch.stack(hs)
        logits, value = self.heads(x.reshape(T * B, -1), hs.reshape(T * B, -1))
        return logits.view(T, B, -1), value.view(T, B)


# ---------------------------------------------------------------------------------- distributions
def split_logits(logits):
    return torch.split(logits, HEAD_SIZES, dim=-1)


def sample(logits, generator=None, greedy=False):
    comps = []
    for lg in split_logits(logits):
        if greedy:
            comps.append(lg.argmax(-1))
        else:
            p = torch.softmax(lg.float(), -1)
            comps.append(torch.multinomial(p.reshape(-1, p.shape[-1]), 1, generator=generator).view(p.shape[:-1]))
    return torch.stack(comps, -1)


def log_prob(logits, comps):
    lp = 0
    for i, lg in enumerate(split_logits(logits)):
        lp = lp + torch.log_softmax(lg.float(), -1).gather(-1, comps[..., i:i + 1].long()).squeeze(-1)
    return lp


def entropy(logits):
    ent = 0
    for lg in split_logits(logits):
        p = torch.log_softmax(lg.float(), -1)
        ent = ent - (p.exp() * p).sum(-1)
    return ent


def imitation_loss(logits, comps, weights=(1.0, 1.0, 1.0, 1.0, 2.0, 0.3)):
    loss = 0
    for i, lg in enumerate(split_logits(logits)):
        loss = loss + weights[i] * F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), comps[..., i].reshape(-1).long(), reduction="none")
    return loss.view(comps.shape[:-1])


def count_params(m):
    return sum(p.numel() for p in m.parameters())


__all__ = ["DoomNet", "Perception", "OdoNet", "flatten_obs", "stack_obs", "comps_to_buttons", "buttons_to_comps",
           "sample", "log_prob", "entropy", "imitation_loss", "HEAD_SIZES", "math"]
