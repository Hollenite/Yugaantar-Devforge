"""Training environment: level sampling, reward shaping and a privileged teacher.

Everything in this file is TRAINING-TIME scaffolding (allowed by docs/RULES.md §2):
- it reads env.privileged_state() (true position, keys, door sectors) and the generator's cell grid,
- it shapes rewards to mirror the evaluation categories (exit, progression, exploration, combat),
- it runs a privileged teacher whose actions supervise the student network (DAgger-style
  distillation). The submitted agent never sees any of this: it acts from its own observations.
"""

from __future__ import annotations

import heapq
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mapgen import CELL, Level, generate, write_level

os.environ.setdefault("ALSOFT_DRIVERS", "null")

ENEMY_TYPES = (1, 2, 3)
KEY_COLOURS = {"blue": (0, 3), "yellow": (1, 4), "red": (2, 5)}  # vitals key indices: card, skull


# ---------------------------------------------------------------------------------- navigation fields
def distance_field(passable: np.ndarray, targets: list[tuple[int, int]], wall_cost: float = 1.5) -> np.ndarray:
    """Dijkstra distance (in cells) to the nearest target, 8-connected, no corner cutting,
    with extra cost next to walls so paths stay in the middle of rooms and corridors."""
    H, W = passable.shape
    near_wall = np.zeros_like(passable, dtype=bool)
    p = np.pad(passable, 1, constant_values=False)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            near_wall |= ~p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
    extra = np.where(near_wall, wall_cost, 0.0)
    dist = np.full((H, W), np.inf)
    pq = []
    for t in targets:
        dist[t] = 0.0
        heapq.heappush(pq, (0.0, t))
    steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    while pq:
        d, (y, x) = heapq.heappop(pq)
        if d > dist[y, x]:
            continue
        for dy, dx, c in steps:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < H and 0 <= nx < W) or not passable[ny, nx]:
                continue
            if dy and dx and not (passable[y + dy, x] and passable[y, x + dx]):
                continue
            nd = d + c + extra[ny, nx]
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                heapq.heappush(pq, (nd, (ny, nx)))
    return dist


@dataclass
class LevelNav:
    level: Level
    fields: list[np.ndarray]  # per stage
    door_cells: np.ndarray  # (H, W) sector id if the cell is a door, else -1

    @classmethod
    def build(cls, lv: Level, door_sectors: set[int]):
        fields = []
        for stage in range(len(lv.lock_order) + 1):
            passable = lv.walkable & (lv.region >= 0) & (lv.region <= stage)
            targets = [lv.key_cells[stage][1]] if stage < len(lv.lock_order) else list(lv.exit_cells)
            fields.append(distance_field(passable, targets))
        door = np.where(np.isin(lv.grid, list(door_sectors)), lv.grid, -1) if door_sectors else np.full(lv.grid.shape, -1)
        return cls(lv, fields, door)

    def stage(self, keys: np.ndarray) -> int:
        s = 0
        for colour in self.level.lock_order:
            card, skull = KEY_COLOURS[str(colour)]
            if keys[card] or keys[skull]:
                s += 1
            else:
                break
        return s

    def cell(self, x, y):
        return int(y // CELL), int(x // CELL)

    def dist(self, x, y, stage) -> float:
        f = self.fields[stage]
        gy, gx = self.cell(x, y)
        if 0 <= gy < f.shape[0] and 0 <= gx < f.shape[1] and np.isfinite(f[gy, gx]):
            return float(f[gy, gx])
        # standing in a cell the field doesn't cover (e.g. inside a door): nearest covered neighbour
        best = np.inf
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = gy + dy, gx + dx
                if 0 <= yy < f.shape[0] and 0 <= xx < f.shape[1]:
                    best = min(best, f[yy, xx] + 1)
        return float(best)


# ---------------------------------------------------------------------------------- teacher
class Teacher:
    """Privileged expert: follows the geodesic field to the next key / the exit, opens doors,
    aims at visible enemies. Outputs factorised action components (move, strafe, turn, atk, use, wpn)."""

    def __init__(self, nav: LevelNav, rng: np.random.Generator):
        self.nav, self.rng = nav, rng
        self.use_cd = 0
        self.hist: list[tuple[float, float]] = []
        self.unstick = 0
        self.unstick_dir = 1
        self.strafe_t = 0
        self.strafe_dir = 1
        self.wpn_cd = 0
        self.last_use = 0

    def act(self, ps: dict, obs: dict) -> np.ndarray:
        nav = self.nav
        x, y, ang = ps["x"], ps["y"], math.radians(ps["angle"])
        stage = nav.stage(ps["keys"])
        f = nav.fields[stage]
        comps = np.zeros(6, np.int64)
        self.use_cd = max(0, self.use_cd - 1)
        self.wpn_cd = max(0, self.wpn_cd - 1)
        self.hist.append((x, y))
        self.hist = self.hist[-10:]

        # ---- next waypoint: greedy descent on the distance field
        cy, cx = nav.cell(x, y)
        H, W = f.shape
        cur = (min(max(cy, 0), H - 1), min(max(cx, 0), W - 1))
        if not np.isfinite(f[cur]):
            cand = [(f[cur[0] + dy, cur[1] + dx], (cur[0] + dy, cur[1] + dx)) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                    if 0 <= cur[0] + dy < H and 0 <= cur[1] + dx < W]
            cur = min(cand)[1]
        path = [cur]
        for _ in range(4):
            yy, xx = path[-1]
            best = min(((f[yy + dy, xx + dx], (yy + dy, xx + dx)) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                        if 0 <= yy + dy < H and 0 <= xx + dx < W), key=lambda t: t[0])
            if best[0] >= f[yy, xx]:
                break
            path.append(best[1])
        ty, tx = path[-1]
        tgt_x, tgt_y = tx * CELL + CELL / 2, ty * CELL + CELL / 2
        at_goal = f[cur] <= 0.5 or (len(path) == 1 and f[cur] < 2)

        lv = nav.level
        if at_goal and stage == len(lv.lock_order) and lv.meta.get("exit_switch"):
            # face the switch wall (the solid neighbour of the exit cell) and press use
            ey, ex = lv.exit_cells[0]
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if not lv.walkable[ey + dy, ex + dx]:
                    tgt_x, tgt_y = (ex + dx) * CELL + CELL / 2, (ey + dy) * CELL + CELL / 2
                    break
        desired = math.atan2(tgt_y - y, tgt_x - x)
        diff = (desired - ang + math.pi) % (2 * math.pi) - math.pi
        deg = math.degrees(diff)
        comps[2] = 1 if deg > 6 else 2 if deg < -6 else 0
        comps[0] = 1 if abs(deg) < 45 else 0
        if at_goal and stage == len(lv.lock_order) and lv.meta.get("exit_switch"):
            if abs(deg) < 20 and self.use_cd == 0:
                comps[4] = 1
                self.use_cd = 4

        # ---- doors on the way: press use once, then wait for it to open
        closed_ahead = False
        for (yy, xx) in path[1:4] + [cur]:
            sid = nav.door_cells[yy, xx]
            if sid >= 0:
                fl, ce = ps["sector_floor"], ps["sector_ceiling"]
                if fl is not None and sid < len(fl) and ce[sid] - fl[sid] < 60:
                    closed_ahead = True
                    break
        if closed_ahead and abs(deg) < 35:
            if self.use_cd == 0:
                comps[4] = 1
                self.use_cd = 30

        # ---- stuck detection
        if self.unstick > 0:
            self.unstick -= 1
            comps[0] = 2 if self.unstick > 3 else 1
            comps[1] = 1 if self.unstick_dir > 0 else 2
            if self.use_cd == 0:
                comps[4] = 1
                self.use_cd = 12
        elif len(self.hist) == 10 and comps[0] == 1:
            moved = math.hypot(self.hist[-1][0] - self.hist[0][0], self.hist[-1][1] - self.hist[0][1])
            if moved < 12:
                self.unstick = 8
                self.unstick_dir = int(self.rng.choice([-1, 1]))

        # ---- combat: aim at the nearest visible enemy
        ent = obs["visible_entities"]
        best = None
        for i in range(len(ent["mask"])):
            if ent["mask"][i] and int(ent["entity_type"][i]) in ENEMY_TYPES:
                d = float(ent["features"][i, 0]) * 1024
                if d < 1400 and (best is None or d < best[0]):
                    best = (d, math.degrees(math.atan2(ent["features"][i, 1], ent["features"][i, 2])))
        ammo = float(obs["vitals"][3])
        if best is not None:
            d, rel = best
            comps[2] = 1 if rel > 3 else 2 if rel < -3 else 0
            if abs(rel) < 5 + 600 / max(d, 60):
                comps[3] = 1
            if d < 700 and self.unstick == 0:
                comps[0] = 0 if d > 200 else 2
                self.strafe_t -= 1
                if self.strafe_t <= 0:
                    self.strafe_t = int(self.rng.integers(4, 12))
                    self.strafe_dir = int(self.rng.choice([-1, 1]))
                comps[1] = 1 if self.strafe_dir < 0 else 2
        if ammo <= 0 and self.wpn_cd == 0 and obs["vitals"][4:8].sum() > 0:
            comps[5] = 1
            self.wpn_cd = 20
        # USE fires on the press edge: never hold it two steps in a row
        if comps[4] and self.last_use:
            comps[4] = 0
        self.last_use = int(comps[4])
        return comps


# ---------------------------------------------------------------------------------- shaped env
@dataclass
class RewardCfg:
    exit: float = 10.0
    exit_health: float = 2.0
    exit_time: float = 2.0
    death: float = -5.0
    kill: float = 1.0
    damage: float = -0.02
    key: float = 3.0
    door: float = 0.5
    locked_door: float = 2.0
    explore: float = 0.03  # per new 64x64 area visited
    potential: float = 0.04  # per cell of geodesic progress towards the next goal
    step: float = -0.002
    gamma: float = 0.995


class TrainEnv:
    """One training episode stream. Levels come from the generator (with teacher + geodesic
    shaping) or from external WADs (event rewards and exploration only)."""

    def __init__(self, worker_id: int, seed: int, tier_weights=(0.15, 0.2, 0.3, 0.2, 0.15), wad_pool=None,
                 wad_prob: float = 0.0, episodes_per_level: int = 2, reward: RewardCfg | None = None, tmpdir=None):
        self.wid = worker_id
        self.rng = np.random.default_rng(seed)
        self.tier_weights = np.asarray(tier_weights, float)
        self.wad_pool = wad_pool or []
        self.wad_prob = wad_prob
        self.episodes_per_level = episodes_per_level
        self.rc = reward or RewardCfg()
        self.tmp = Path(tmpdir or f"/tmp/doom_train_{os.getpid()}_{worker_id}")
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.env = None
        self.nav: LevelNav | None = None
        self.level_eps = 10 ** 9
        self.level_counter = 0
        self.tier = -1

    # ------------------------------------------------------------------ level management
    def _new_level(self):
        from doom_env import make_env

        if self.env is not None:
            self.env.close()
        if self.wad_pool and self.rng.random() < self.wad_prob:
            wad, mp = self.wad_pool[int(self.rng.integers(len(self.wad_pool)))]
            self.env = make_env(wad, mp, render_buffers=False, episode_ticks=int(self.rng.integers(2100, 4201)))
            self.nav, self.tier = None, -1
        else:
            self.tier = int(self.rng.choice(len(self.tier_weights), p=self.tier_weights / self.tier_weights.sum()))
            seed = int(self.rng.integers(1 << 31))
            lv = generate(self.tier, seed)
            path = write_level(lv, self.tmp / f"L{self.level_counter % 2}.wad")
            ticks = int(self.rng.integers(1400, 2100)) if self.tier == 0 else int(self.rng.integers(2100, 4201))
            self.env = make_env(str(path), render_buffers=False, episode_ticks=ticks)
            self.nav = LevelNav.build(lv, set(self.env.mapinfo.door_sectors))
        self.level_counter += 1
        self.level_eps = 0

    def set_tier_weights(self, w):
        self.tier_weights = np.asarray(w, float)

    def reset(self):
        if self.level_eps >= self.episodes_per_level:
            self._new_level()
        self.level_eps += 1
        obs, _ = self.env.reset(seed=int(self.rng.integers(1 << 30)))
        ps = self.env.privileged_state()
        self.teacher = Teacher(self.nav, self.rng) if self.nav is not None else None
        self.visited = set()
        self._visit(ps)
        self.prev = ps
        self.doors_open: set[int] = set()
        self.ep = dict(ret=0.0, steps=0, kills=0, keys=0, doors=0, explore=0, exit=0, death=0, tier=self.tier,
                       teacher=int(self.teacher is not None))
        self.stage = self.nav.stage(ps["keys"]) if self.nav else 0
        self.phi = self._phi(ps, self.stage)
        self.obs = obs
        return obs

    def _visit(self, ps):
        k = (int(ps["x"] // 64), int(ps["y"] // 64))
        if k not in self.visited:
            self.visited.add(k)
            return 1
        return 0

    def _phi(self, ps, stage):
        if self.nav is None:
            return 0.0
        d = self.nav.dist(ps["x"], ps["y"], stage)
        return -self.rc.potential * min(d, 400.0) if np.isfinite(d) else -self.rc.potential * 400.0

    def teacher_action(self):
        if self.teacher is None:
            return np.zeros(6, np.int64), False
        return self.teacher.act(self.prev, self.obs), True

    # ------------------------------------------------------------------ step
    def step(self, buttons: np.ndarray):
        rc = self.rc
        obs, _, term, trunc, info = self.env.step(buttons)
        ps = self.env.privileged_state()
        prev = self.prev
        r = rc.step
        kills = ps["kills"] - prev["kills"]
        r += rc.kill * kills
        dmg = max(0.0, ps["damage_taken"] - prev["damage_taken"])
        r += rc.damage * dmg
        new_keys = int(np.sum(ps["keys"] > prev["keys"]))
        r += rc.key * new_keys
        if ps["sector_ceiling"] is not None and ps["door_lock"] is not None:
            open_now = np.nonzero((ps["door_lock"] >= 0) & (ps["sector_ceiling"] - ps["sector_floor"] > 56))[0]
            for sid in open_now:
                if int(sid) not in self.doors_open:
                    self.doors_open.add(int(sid))
                    r += rc.locked_door if ps["door_lock"][sid] > 0 else rc.door
                    self.ep["doors"] += 1
        new_area = self._visit(ps)
        r += rc.explore * new_area
        # potential-based shaping towards the current goal (skip on stage changes)
        stage = self.nav.stage(ps["keys"]) if self.nav else 0
        phi = self._phi(ps, stage)
        if stage == self.stage and not (term or trunc):
            r += rc.gamma * phi - self.phi
        self.stage, self.phi = stage, phi
        if ps["exit_reached"]:
            r += rc.exit + rc.exit_health * max(ps["health"], 0) / 100 + rc.exit_time * (1 - ps["tic"] / ps["episode_ticks"])
            self.ep["exit"] = 1
        if ps["dead"]:
            r += rc.death
            self.ep["death"] = 1
        self.ep["ret"] += r
        self.ep["steps"] += 1
        self.ep["kills"] += kills
        self.ep["keys"] += new_keys
        self.ep["explore"] += new_area
        self.prev, self.obs = ps, obs
        done = term or trunc
        return obs, float(r), done, info

    def close(self):
        if self.env is not None:
            self.env.close()


def load_wad_pool(spec: list[str]) -> list[tuple[str, str]]:
    """'path.wad' (all maps) or 'path.wad:MAP03' entries -> [(wad, map)]."""
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "env"))
    from doom_env.wad import list_maps, read_wad

    out = []
    for s in spec:
        if ":" in s:
            w, m = s.rsplit(":", 1)
            out.append((w, m))
        else:
            out += [(s, m) for m in list_maps(read_wad(s))]
    return out
