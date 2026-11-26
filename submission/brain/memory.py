"""Odometry-anchored spatial memory ("EgoMap").

The official observations contain no global coordinates, so the agent builds its own map:

1. **Odometry.** Doom's player physics are simple and deterministic: every tic a held movement
   button adds a fixed thrust, the position moves by the momentum, and friction multiplies the
   momentum by 0xE800/0x10000. Turning uses a fixed angle per tic that doubles after the button has
   been held for 6 tics. Replaying these rules from the agent's OWN previous action and the observed
   ego-frame velocity (vitals 14-15) gives a dead-reckoned pose (x, y, heading) relative to where the
   episode started. When the observed velocity disagrees with the replay (the player hit a wall),
   the displacement is rescaled towards what the velocity says.

2. **Writing.** Each step the 64 rays are projected into an allocentric grid (fixed to the start
   pose): free space along each ray, and the hit cell into the channel of its hit type (wall, door,
   locked door, item, enemy). Visible entities are written as well (keys get their own channel), and
   the agent's own trail marks visited cells. Writing into a world-fixed grid avoids the blur of
   re-warping an ego-centric map every step.

3. **Reading.** Two ego-centric, heading-aligned crops are resampled from the grid every step: a
   fine local view and a coarse wide view. A small CNN in the policy encodes them. The crops are the
   network's input; all decisions come from the trained network (no planning on this map).

The same code runs batched inside the trainer and with batch size 1 inside DoomAgent.
Everything is torch, so it runs on CPU, CUDA or MPS.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

NUM_RAYS = 64
RAY_RANGE = 1024.0

# Doom player physics (p_user.c / g_game.c), in map units and degrees.
FRICTION = 0xE800 / 0x10000  # 0.90625 per tic
FORWARD_THRUST = 25 * 2048 / 65536  # forwardmove[0] * 2048 >> 16 = 0.78125 units/tic
SIDE_THRUST = 24 * 2048 / 65536  # sidemove[0]
TURN_SLOW = 320 / 65536 * 360  # angleturn[2] per tic (first SLOWTURNTICS tics)
TURN_NORMAL = 640 / 65536 * 360  # angleturn[0] per tic
SLOWTURNTICS = 6
TICS = 4  # frame skip

# memory channels
CH_FREE, CH_WALL, CH_DOOR, CH_LOCKED, CH_ITEM, CH_ENEMY, CH_KEY, CH_TRAIL = range(8)
NUM_CH = 8
# ray hit_type -> channel (0 none -> nothing)
_HIT_TO_CH = torch.tensor([-1, CH_WALL, CH_DOOR, CH_LOCKED, CH_ENEMY, CH_ITEM])
# entity class ids (doom_env.spaces.ENTITY_TYPES): 1-3 enemies, 4 projectile, 5-8 pickups, 9 key
ENEMY_TYPES = (1, 2, 3)
ITEM_TYPES = (5, 6, 7, 8)
KEY_TYPE = 9


class EgoMap:
    """Batched odometry + allocentric grid + ego-centric crops.

    grid_size cells of cell_units map units each, centred on the start position.
    """

    def __init__(self, batch: int, device="cpu", grid_size: int = 192, cell_units: float = 40.0,
                 crop: int = 32, local_units: float = 40.0, wide_units: float = 160.0):
        k = max(1, int(round(wide_units / cell_units)))
        if grid_size % k:
            raise ValueError("grid_size must be divisible by wide_units / cell_units")
        self.B, self.device = batch, torch.device(device)
        self.G, self.cell = grid_size, cell_units
        self.crop, self.local_units, self.wide_units = crop, local_units, wide_units
        self.grid = torch.zeros(batch, NUM_CH, grid_size, grid_size, device=self.device)
        self.pose = torch.zeros(batch, 3, device=self.device)  # x, y, heading(rad), start-relative
        self.mom = torch.zeros(batch, 2, device=self.device)  # predicted momentum (fwd, right) units/tic
        self.turnheld = torch.zeros(batch, device=self.device)
        self.new_free = torch.zeros(batch, device=self.device)  # cells revealed this step (curiosity)
        # ray directions relative to heading: index 0 ahead, counter-clockwise
        self.ray_ang = torch.arange(NUM_RAYS, device=self.device) * (2 * math.pi / NUM_RAYS)
        self.samples = torch.linspace(0.0, 1.0, 24, device=self.device)  # points along each ray
        self._hit_to_ch = _HIT_TO_CH.to(self.device)

    # ------------------------------------------------------------------ episode control
    def reset(self, idx=None):
        if idx is None:
            idx = torch.arange(self.B, device=self.device)
        idx = torch.as_tensor(idx, device=self.device, dtype=torch.long)
        if idx.numel() == 0:
            return
        self.grid[idx] = 0
        self.pose[idx] = 0
        self.mom[idx] = 0
        self.turnheld[idx] = 0
        self.new_free[idx] = 0

    # ------------------------------------------------------------------ odometry
    @torch.no_grad()
    def move(self, buttons: torch.Tensor, vel: torch.Tensor, prev_rays: torch.Tensor | None = None,
             rays: torch.Tensor | None = None, odonet: "OdoNet | None" = None):
        """Advance the dead-reckoned pose by one decision step.

        buttons: (B, 9) the action that was just executed (0/1).
        vel: (B, 2) observed ego velocity (forward, right) AFTER the step (vitals[14:16]).
        prev_rays / rays: (B, 64) ray distances before / after the step; with `odonet` they are used
        to correct the physics replay when the player slid along a wall or bumped into something.
        """
        feats = odometry_features(buttons, self.mom, vel, self.turnheld)
        dtheta, disp_phys, m_end, th = feats["dtheta"], feats["disp"], feats["m_end"], feats["turnheld"]
        self.turnheld = th
        if odonet is not None and prev_rays is not None and rays is not None:
            disp = disp_phys + odonet(odonet_input(prev_rays, rays, feats, vel))
        else:
            # fallback heuristic: when the engine's velocity is smaller than predicted we were blocked
            pred_speed = m_end.norm(dim=1)
            ratio = torch.where(pred_speed > 0.05, (vel.norm(dim=1) / pred_speed.clamp_min(1e-3)).clamp(0.0, 1.5),
                                torch.ones_like(pred_speed))
            blocked = (ratio < 0.85).float().unsqueeze(1)
            disp = blocked * (0.5 * disp_phys * ratio.unsqueeze(1) + 0.5 * vel * 4.6) + (1 - blocked) * disp_phys
        self.mom = vel.clone()  # re-anchor the momentum to what the engine reports

        heading = self.pose[:, 2] + torch.deg2rad(dtheta) * 0.5  # mid-step heading
        c, s = torch.cos(heading), torch.sin(heading)
        self.pose[:, 0] += disp[:, 0] * c + disp[:, 1] * s
        self.pose[:, 1] += disp[:, 0] * s - disp[:, 1] * c
        self.pose[:, 2] = torch.remainder(self.pose[:, 2] + torch.deg2rad(dtheta) + math.pi, 2 * math.pi) - math.pi

    # ------------------------------------------------------------------ writing
    def _to_cells(self, x, y):
        gx = torch.floor(x / self.cell + self.G / 2).long()
        gy = torch.floor(y / self.cell + self.G / 2).long()
        ok = (gx >= 0) & (gx < self.G) & (gy >= 0) & (gy < self.G)
        return gx.clamp(0, self.G - 1), gy.clamp(0, self.G - 1), ok

    @torch.no_grad()
    def write(self, ray_dist, ray_hit, ent_feat, ent_type, ent_mask):
        """ray_dist (B,64) in [0,1], ray_hit (B,64) int, ent_* from visible_entities."""
        B, G = self.B, self.G
        px, py, hd = self.pose[:, 0:1], self.pose[:, 1:2], self.pose[:, 2:3]
        ang = hd + self.ray_ang.unsqueeze(0)  # (B,64)
        cos, sin = torch.cos(ang), torch.sin(ang)
        dist = ray_dist * RAY_RANGE
        flat = self.grid.view(B, NUM_CH, G * G)

        # decay transient channels
        self.grid[:, CH_ENEMY].mul_(0.85)

        # free space along each ray (stop half a cell before the hit)
        t = self.samples.view(1, 1, -1) * (dist.unsqueeze(2) - self.cell * 0.5).clamp_min(0)  # (B,64,S)
        fx = px.unsqueeze(2) + cos.unsqueeze(2) * t
        fy = py.unsqueeze(2) + sin.unsqueeze(2) * t
        gx, gy, ok = self._to_cells(fx.reshape(B, -1), fy.reshape(B, -1))
        lin = gy * G + gx
        before = flat[:, CH_FREE].gather(1, lin)
        newly = ((before < 0.5) & ok).float()
        # count each newly revealed cell once
        tmp = torch.zeros(B, G * G, device=self.device)
        tmp.scatter_(1, lin, newly)
        self.new_free = tmp.sum(1)
        val = ok.float()
        flat[:, CH_FREE].scatter_reduce_(1, lin, val, reduce="amax")
        # seeing through a cell weakens obstacle evidence there (doors that opened, dead enemies)
        for ch in (CH_DOOR, CH_LOCKED, CH_ENEMY, CH_ITEM):
            cur = flat[:, ch].gather(1, lin)
            flat[:, ch].scatter_(1, lin, cur * (1 - 0.5 * val))

        # hits
        hit_ch = self._hit_to_ch[ray_hit.long().clamp(0, 5)]  # (B,64)
        hx = px + cos * dist
        hy = py + sin * dist
        gx, gy, ok = self._to_cells(hx, hy)
        lin = gy * G + gx
        valid = ok & (hit_ch >= 0)
        ch = hit_ch.clamp_min(0)
        flat_all = self.grid.view(B, NUM_CH * G * G)
        idx = ch * (G * G) + lin
        flat_all.scatter_reduce_(1, idx, valid.float(), reduce="amax")

        # entities: distance/1024, sin, cos of angle relative to heading
        m = ent_mask.float()
        d = ent_feat[..., 0] * RAY_RANGE
        a = hd + torch.atan2(ent_feat[..., 1], ent_feat[..., 2])
        ex, ey = px + torch.cos(a) * d, py + torch.sin(a) * d
        gx, gy, ok = self._to_cells(ex, ey)
        lin = gy * G + gx
        et = ent_type.long()
        ech = torch.full_like(et, -1)
        for t_ in ENEMY_TYPES:
            ech = torch.where(et == t_, torch.full_like(et, CH_ENEMY), ech)
        for t_ in ITEM_TYPES:
            ech = torch.where(et == t_, torch.full_like(et, CH_ITEM), ech)
        ech = torch.where(et == KEY_TYPE, torch.full_like(et, CH_KEY), ech)
        valid = ok & (ech >= 0) & (m > 0)
        idx = ech.clamp_min(0) * (G * G) + lin
        flat_all.scatter_reduce_(1, idx, valid.float(), reduce="amax")

        # own trail
        gx, gy, ok = self._to_cells(self.pose[:, 0], self.pose[:, 1])
        lin = (gy * G + gx).unsqueeze(1)
        flat[:, CH_TRAIL].scatter_(1, lin, torch.ones_like(lin, dtype=flat.dtype))

    # ------------------------------------------------------------------ reading
    @torch.no_grad()
    def read(self) -> torch.Tensor:
        """(B, 2*NUM_CH, crop, crop): local and wide ego-centric views, heading pointing 'up'."""
        return torch.cat([self._crop(self.local_units), self._crop(self.wide_units)], 1)

    def _crop(self, units_per_px: float) -> torch.Tensor:
        """Heading-aligned crop: output row 0 is straight ahead (far), the centre is the agent,
        columns grow to the agent's right.

        For output pixel coords (xo, yo) in [-1, 1] (xo = column, yo = row, downwards):
            ego right   u = xo * k,  ego forward v = -yo * k,   k = crop/2 * units_per_px
            world = pos + u * (sin h, -cos h) + v * (cos h, sin h)
        and grid_sample's input coords are world / half_world (x -> columns, y -> rows).
        """
        B, G, C = self.B, self.G, self.crop
        half_world = G * self.cell / 2
        hd = self.pose[:, 2]
        c, s = torch.cos(hd), torch.sin(hd)
        scale = C / 2 * units_per_px / half_world
        theta = torch.zeros(B, 2, 3, device=self.device)
        theta[:, 0, 0] = s * scale
        theta[:, 0, 1] = -c * scale
        theta[:, 1, 0] = -c * scale
        theta[:, 1, 1] = -s * scale
        theta[:, 0, 2] = self.pose[:, 0] / half_world
        theta[:, 1, 2] = self.pose[:, 1] / half_world
        grid = F.affine_grid(theta, (B, NUM_CH, C, C), align_corners=False)
        k = max(1, int(round(units_per_px / self.cell)))
        src = F.max_pool2d(self.grid, k, k) if k > 1 else self.grid  # G % k == 0 (checked in __init__)
        return F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


# ---------------------------------------------------------------------------------- odometry model
def odometry_features(buttons: torch.Tensor, mom: torch.Tensor, vel: torch.Tensor, turnheld: torch.Tensor) -> dict:
    """Replay Doom's movement physics for one 4-tic step from the executed buttons.

    mom: momentum (fwd, right) at the end of the previous step, in the previous heading frame.
    Returns the turn (deg), the displacement predicted without collisions (in the mid-step frame),
    the predicted end momentum and the updated turn-held counter.
    """
    b = buttons.float()
    thrust = torch.stack([(b[:, 0] - b[:, 1]) * FORWARD_THRUST, (b[:, 3] - b[:, 2]) * SIDE_THRUST], 1)
    turning = (b[:, 4] + b[:, 5]) > 0
    turn_dir = b[:, 4] - b[:, 5]  # +1 left (CCW)
    th = turnheld.clone()
    m = mom.clone()
    disp = torch.zeros_like(mom)
    dtheta = torch.zeros_like(turnheld)
    for _ in range(TICS):
        th = torch.where(turning, th + 1, torch.zeros_like(th))
        rate = torch.where(th < SLOWTURNTICS, torch.full_like(th, TURN_SLOW), torch.full_like(th, TURN_NORMAL))
        dtheta = dtheta + turn_dir * rate
        m = m + thrust
        disp = disp + m
        m = m * FRICTION
    return {"dtheta": dtheta, "disp": disp, "m_end": m, "turnheld": th, "thrust": thrust, "mom": mom}


def roll_rays(rays: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """out[i] = rays[(i + shift) mod 64] with linear interpolation; shift (B,) in rays (may be fractional)."""
    n = rays.shape[1]
    idx = torch.arange(n, device=rays.device).unsqueeze(0) + shift.unsqueeze(1)
    lo = torch.floor(idx)
    w = idx - lo
    lo = lo.long() % n
    hi = (lo + 1) % n
    return rays.gather(1, lo) * (1 - w) + rays.gather(1, hi) * w


def odonet_input(prev_rays, rays, feats, vel):
    # a static wall at angle phi before a CCW turn of d appears at phi - d afterwards
    shift = feats["dtheta"] / (360.0 / NUM_RAYS)
    aligned_prev = roll_rays(prev_rays.float(), shift)
    scans = torch.stack([aligned_prev, rays.float(), rays.float() - aligned_prev], 1)  # (B,3,64)
    scal = torch.cat([feats["mom"], vel, feats["thrust"], feats["disp"] / 4, feats["m_end"],
                      (feats["dtheta"] / 14.0).unsqueeze(1)], 1)  # (B,11)
    return scans, scal


class OdoNet(torch.nn.Module):
    """Learned odometry correction: predicts (true displacement - physics replay) from two
    consecutive ray scans plus the physics features. Trained with privileged ground truth
    (training/train_odometry.py); uses only agent-side observations at inference."""

    def __init__(self, width: int = 32):
        super().__init__()
        nn = torch.nn
        self.conv = nn.Sequential(
            nn.Conv1d(3, width, 5, padding=2, padding_mode="circular"), nn.GELU(),
            nn.Conv1d(width, width, 5, stride=2, padding=2, padding_mode="circular"), nn.GELU(),
            nn.Conv1d(width, width, 5, stride=2, padding=2, padding_mode="circular"), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(width * 16 + 11, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(),
                                  nn.Linear(64, 2))

    def forward(self, inp):
        scans, scal = inp
        h = self.conv(scans * 4.0).flatten(1)
        return self.head(torch.cat([h, scal], 1)) * 4.0
