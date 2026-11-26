"""Watch a random agent play through the official doom_env wrapper, with a live sensor view.

Throwaway demo for checking your local setup: it is NOT the submission agent.

    uv run python training/watch_random.py                      # sticky-random movement, real time
    uv run python training/watch_random.py --policy human       # WASD + mouse look, click to shoot
    uv run python training/watch_random.py --policy uniform     # pure uniform random buttons
    uv run python training/watch_random.py --map maps/your.wad:MAP01 --fps 0   # as fast as possible
    uv run python training/watch_random.py --policy human --pool               # Freedoom-first mix

Left:   160x120 RGB (Tab = stretched depth). The bottom sprite is YOUR weapon, not an enemy.
Middle: first-person 3D from map sectors + 64 rays. Bright wedge is the 90° entity FOV.
Right:  top-down radar and live observation / privileged data.
Keys:  P pause, R new episode, Tab rgb/depth, G cycle 3D/radar, ESC quit.
Human (FPS): WASD (A/D strafe while mouse is grabbed), mouse yaw only (vanilla Doom: no look up/down),
       LMB shoot, RMB/Space USE, wheel/X weapon. Click window to grab, M or Esc to release.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pygame  # noqa: E402  (pygame-ce ships with the template's dependencies)

from doom_env import make_env  # noqa: E402
from doom_env import spaces as S  # noqa: E402

RAY_COLORS = {  # hit_type -> colour
    0: (60, 60, 60),  # nothing within range
    1: (150, 150, 150),  # wall
    2: (80, 160, 255),  # door
    3: (255, 80, 200),  # locked door
    4: (255, 60, 60),  # enemy
    5: (80, 255, 120),  # item
}
ENTITY_COLORS = {
    "enemy": (255, 60, 60),
    "item": (80, 255, 120),
    "projectile": (255, 200, 0),
    "key": (255, 80, 200),
}
HIT_NAMES = ["none", "wall", "door", "locked door", "enemy", "item"]
# ViZDoom SELECTED_WEAPON is the Doom slot (1=fist … 7=chainsaw), not a 0-based index.
WEAPON_NAMES = {1: "fist", 2: "pistol", 3: "shotgun", 4: "chaingun", 5: "rocket", 6: "plasma", 7: "bfg", 8: "chainsaw"}
EYE_HEIGHT = 41.0
NEAR_Z = 8.0
PLAYER_FIT = 56.0  # vanilla Doom height used for door blocking


class RandomPolicy:
    """Observation-free random behaviour, only so there is something to look at."""

    def __init__(self, kind: str, seed: int):
        self.kind = kind
        self.rng = np.random.default_rng(seed)
        self.turn = 0  # -1 left, 0 none, +1 right
        self.use_cooldown = 0

    def reset(self):
        self.turn = 0
        self.use_cooldown = 0

    def act(self) -> np.ndarray:
        if self.kind == "uniform":
            return self.rng.integers(0, 2, size=9, dtype=np.int8)
        a = np.zeros(9, dtype=np.int8)
        if self.rng.random() < 0.9:
            a[0] = 1  # mostly forward
        if self.rng.random() < 0.15:  # sticky turning
            self.turn = int(self.rng.integers(-1, 2))
        if self.turn < 0:
            a[4] = 1
        elif self.turn > 0:
            a[5] = 1
        if self.rng.random() < 0.3:
            a[6] = 1  # shoot now and then
        if self.use_cooldown > 0:
            self.use_cooldown -= 1  # USE fires on button-down, so pulse it with gaps
        elif self.rng.random() < 0.08:
            a[7] = 1
            self.use_cooldown = 6
        return a


class HumanPolicy:
    """FPS keyboard + analog mouse yaw. Vanilla Doom has no pitch; dy is ignored.

    Returns (9 binary buttons, yaw_delta). The trained agent never sees delta look.
    """

    def __init__(self, mouse_sens: float = 0.365):
        self.mouse_sens = mouse_sens
        self.use_held = False
        self.weapon_held = False
        self.grabbed = False
        self.queue_weapon = False
        self.ignore_click = False
        self.yaw_accum = 0.0

    def reset(self):
        self.use_held = False
        self.weapon_held = False
        self.queue_weapon = False
        self.ignore_click = False
        self.yaw_accum = 0.0
        pygame.mouse.get_rel()

    def set_grab(self, on: bool) -> None:
        self.grabbed = on
        pygame.event.set_grab(on)
        pygame.mouse.set_visible(not on)
        if hasattr(pygame.mouse, "set_relative_mode"):
            pygame.mouse.set_relative_mode(on)
        pygame.mouse.get_rel()
        self.yaw_accum = 0.0

    def act(self) -> tuple[np.ndarray, int]:
        keys = pygame.key.get_pressed()
        mouse = pygame.mouse.get_pressed()
        dx, _dy = pygame.mouse.get_rel()
        yaw = 0
        if self.grabbed:
            self.yaw_accum += dx * self.mouse_sens
            yaw = int(self.yaw_accum)
            self.yaw_accum -= yaw
        a = np.zeros(9, dtype=np.int8)
        a[0] = keys[pygame.K_w] or keys[pygame.K_UP]
        a[1] = keys[pygame.K_s] or keys[pygame.K_DOWN]
        if self.grabbed:
            a[2] = keys[pygame.K_a] or keys[pygame.K_q]
            a[3] = keys[pygame.K_d] or keys[pygame.K_e]
            a[4] = keys[pygame.K_LEFT]
            a[5] = keys[pygame.K_RIGHT]
        else:
            a[2] = keys[pygame.K_q]
            a[3] = keys[pygame.K_e]
            a[4] = keys[pygame.K_a] or keys[pygame.K_LEFT]
            a[5] = keys[pygame.K_d] or keys[pygame.K_RIGHT]
        shoot = bool(keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL] or keys[pygame.K_f] or mouse[0])
        a[6] = shoot and not self.ignore_click
        self.ignore_click = False
        use = bool(keys[pygame.K_SPACE] or keys[pygame.K_RETURN] or mouse[2])
        if use and not self.use_held:
            a[7] = 1
        self.use_held = use
        weap = bool(keys[pygame.K_x] or self.queue_weapon)
        if weap and not self.weapon_held:
            a[8] = 1
        self.weapon_held = weap
        self.queue_weapon = False
        return a, yaw


def entity_kind(name: str) -> str:
    if name.startswith("enemy"):
        return "enemy"
    if name == "projectile":
        return "projectile"
    if name == "key":
        return "key"
    return "item"


def blit_text(surf: pygame.Surface, text: str, pos: tuple[int, int], color, font) -> None:
    # Render onto an opaque surface first. Direct SRCALPHA blits onto the radar Surface
    # come out as solid black bars on macOS pygame-ce.
    img = font.render(text, True, color, (12, 12, 16))
    surf.blit(img, pos)


def stretch_depth(depth: np.ndarray) -> np.ndarray:
    """ViZDoom depth on small maps only uses a slice of 0–255; stretch so D is readable."""
    d = depth.astype(np.float32)
    hi = float(d.max())
    vis = np.zeros(depth.shape, np.uint8) if hi < 1.0 else np.clip(d * (255.0 / hi), 0, 255).astype(np.uint8)
    return np.repeat(vis[:, :, None], 3, axis=2)


def _heading_rad(priv: dict) -> float:
    return math.radians(float(priv["angle"]))


def world_to_ego(wx: float, wy: float, px: float, py: float, heading: float) -> tuple[float, float]:
    """World XY -> (forward, right) in the player frame. Same convention as doom_env.sensors."""
    dx, dy = wx - px, wy - py
    c, s = math.cos(heading), math.sin(heading)
    return dx * c + dy * s, dx * s - dy * c


def _clip_near(p0, p1, zmin: float = NEAR_Z):
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    if z0 >= zmin and z1 >= zmin:
        return p0, p1
    if z0 < zmin and z1 < zmin:
        return None
    t = (zmin - z0) / (z1 - z0 + 1e-9)
    pc = (x0 + t * (x1 - x0), y0 + t * (y1 - y0), zmin)
    return (pc, p1) if z0 < zmin else (p0, pc)


def _project(x: float, y: float, z: float, w: int, h: int, focal: float) -> tuple[int, int] | None:
    if z < NEAR_Z:
        return None
    return int(w * 0.5 + x * focal / z), int(h * 0.5 - y * focal / z)


def draw_radar(surf: pygame.Surface, obs: dict, font) -> None:
    surf.fill((12, 12, 16))
    w, h = surf.get_size()
    cx, cy = w // 2, h // 2
    radius = min(w, h) // 2 - 10
    for frac in (0.25, 0.5, 0.75, 1.0):  # range rings
        pygame.draw.circle(surf, (35, 35, 45), (cx, cy), int(radius * frac), 1)
    dist = obs["rays"]["distance"]
    hit = obs["rays"]["hit_type"]
    for i in range(S.NUM_RAYS):
        ang = math.radians(i * 360.0 / S.NUM_RAYS)  # 0 = ahead, counter-clockwise
        d = float(dist[i]) * radius
        ex, ey = cx - math.sin(ang) * d, cy - math.cos(ang) * d
        col = RAY_COLORS[int(hit[i])]
        pygame.draw.line(surf, col, (cx, cy), (ex, ey), 1)
        if hit[i] != 0:
            pygame.draw.circle(surf, col, (int(ex), int(ey)), 4)
    ents = obs["visible_entities"]
    n_vis = 0
    for k in range(S.MAX_ENTITIES):
        if not ents["mask"][k]:
            continue
        n_vis += 1
        d, s, c = ents["features"][k][:3]
        name = S.ENTITY_TYPES[int(ents["entity_type"][k])]
        col = ENTITY_COLORS[entity_kind(name)]
        px = cx - s * min(d, 1.0) * radius
        py = cy - c * min(d, 1.0) * radius
        pygame.draw.circle(surf, col, (int(px), int(py)), 6, 2)
    # 90 degree field of view (matches visible_entities, not necessarily the RGB camera)
    for sign in (-1, 1):
        a = math.radians(45 * sign)
        pygame.draw.line(surf, (90, 90, 140), (cx, cy), (cx - math.sin(a) * radius, cy - math.cos(a) * radius), 1)
    pygame.draw.polygon(surf, (255, 255, 255), [(cx, cy - 14), (cx - 7, cy - 4), (cx + 7, cy - 4)])
    pygame.draw.circle(surf, (255, 255, 255), (cx, cy), 5)
    blit_text(surf, "UP = ahead", (8, 6), (230, 230, 240), font)
    blit_text(surf, f"visible_entities {n_vis}  (90 deg FOV)", (8, 24), (180, 180, 190), font)
    ly = h - 20 * len(RAY_COLORS) - 6
    for t, name in enumerate(HIT_NAMES):
        pygame.draw.circle(surf, RAY_COLORS[t], (16, ly + t * 20 + 8), 5)
        blit_text(surf, name, (28, ly + t * 20), (230, 230, 235), font)


def _fov_ray_indices() -> list[int]:
    """64 rays, 0 = ahead, CCW. 90° POV is rays 8..0 then 63..56 (left to right)."""
    return list(range(8, -1, -1)) + list(range(S.NUM_RAYS - 1, S.NUM_RAYS - 9, -1))


def draw_pov_3d(surf: pygame.Surface, obs: dict, priv: dict, sectors: list, font) -> None:
    """90° sensor POV from rays, with in-front map wireframe overlaid."""
    surf.fill((18, 18, 24))
    w, h = surf.get_size()
    pygame.draw.rect(surf, (22, 24, 32), (0, 0, w, h // 2))  # ceiling
    pygame.draw.rect(surf, (28, 26, 24), (0, h // 2, w, h // 2))  # floor
    focal = (w * 0.5) / math.tan(math.radians(45))  # 90° matches entity FOV
    dist, hit = obs["rays"]["distance"], obs["rays"]["hit_type"]
    idxs = _fov_ray_indices()
    col_w = max(w / len(idxs), 1.0)
    for n, i in enumerate(idxs):
        d = max(float(dist[i]) * S.RAY_RANGE, 8.0)
        slice_h = (64.0 / d) * focal
        x0 = int(n * col_w)
        x1 = int((n + 1) * col_w)
        y0 = int(h * 0.5 - slice_h * 0.5)
        y1 = int(h * 0.5 + slice_h * 0.5)
        col = RAY_COLORS[int(hit[i])]
        pygame.draw.rect(surf, col, (x0, y0, max(x1 - x0, 1), max(y1 - y0, 1)))

    px, py = float(priv["x"]), float(priv["y"])
    heading = _heading_rad(priv)
    z_eye = float(priv["z"]) + EYE_HEIGHT
    floor_h = priv["sector_floor"]
    ceil_h = priv["sector_ceiling"]
    door_lock = priv["door_lock"]
    if sectors and floor_h is not None:
        for sid, segs in sectors:
            if segs is None or len(segs) == 0:
                continue
            lock = int(door_lock[sid]) if door_lock is not None and sid < len(door_lock) else -1
            f = float(floor_h[sid]) if sid < len(floor_h) else 0.0
            c = float(ceil_h[sid]) if sid < len(ceil_h) else 128.0
            opening = c - f
            if opening < PLAYER_FIT:
                col = RAY_COLORS[3] if lock > 0 else RAY_COLORS[2] if lock == 0 else (200, 200, 200)
            else:
                col = (160, 160, 170)
            for x1, yw1, x2, yw2 in segs:
                fwd0, right0 = world_to_ego(float(x1), float(yw1), px, py, heading)
                fwd1, right1 = world_to_ego(float(x2), float(yw2), px, py, heading)
                lo, hi = (f - z_eye, (f + (72.0 if opening < PLAYER_FIT else opening)) - z_eye)
                for y in (lo, hi):
                    edge = _clip_near((right0, y, fwd0), (right1, y, fwd1))
                    if edge is None:
                        continue
                    a = _project(*edge[0], w, h, focal)
                    b = _project(*edge[1], w, h, focal)
                    if a and b:
                        pygame.draw.line(surf, col, a, b, 1)
                for right, fwd in ((right0, fwd0), (right1, fwd1)):
                    edge = _clip_near((right, lo, fwd), (right, hi, fwd))
                    if edge is None:
                        continue
                    a = _project(*edge[0], w, h, focal)
                    b = _project(*edge[1], w, h, focal)
                    if a and b:
                        pygame.draw.line(surf, col, a, b, 1)

    ents = obs["visible_entities"]
    for k in range(S.MAX_ENTITIES):
        if not ents["mask"][k]:
            continue
        d, s, c = (float(x) for x in ents["features"][k][:3])
        d = min(max(d, 0.02), 1.0) * S.RAY_RANGE
        name = S.ENTITY_TYPES[int(ents["entity_type"][k])]
        col = ENTITY_COLORS[entity_kind(name)]
        p = _project(s * d, 0.0, max(c * d, NEAR_Z + 1), w, h, focal)
        if p:
            pygame.draw.circle(surf, col, p, 8, 2)
    pygame.draw.line(surf, (220, 220, 80), (w // 2 - 8, h // 2), (w // 2 + 8, h // 2), 1)
    pygame.draw.line(surf, (220, 220, 80), (w // 2, h // 2 - 8), (w // 2, h // 2 + 8), 1)
    blit_text(surf, "90 deg POV from rays + map wireframe", (8, 6), (230, 230, 240), font)


def draw_data_panel(surf: pygame.Surface, obs: dict, priv: dict, info: dict, steps: int, ret: float, font) -> None:
    surf.fill((16, 16, 22))
    v, ix = obs["vitals"], S.VITALS_INDEX
    weap = int(v[ix["selected_weapon"]])
    weap_name = WEAPON_NAMES.get(weap, f"slot{weap}")
    keys = [n.replace("key_", "") for n in S.VITALS_FIELDS if n.startswith("key_") and v[ix[n]] > 0]
    hit = obs["rays"]["hit_type"]
    counts = ", ".join(f"{HIT_NAMES[t][0]}={int((hit == t).sum())}" for t in range(6) if (hit == t).any())
    n_vis = int(obs["visible_entities"]["mask"].sum())
    tic, cap = int(priv["tic"]), int(priv["episode_ticks"])
    lines = [
        ("DATA  (privileged is training-only)", (180, 180, 120)),
        (f"pos {priv['x']:.0f},{priv['y']:.0f}  ang {priv['angle']:.0f}", (230, 230, 235)),
        (f"tic {tic}/{cap}  step {steps}  ret {ret:+.2f}", (200, 200, 210)),
        (f"hp {v[ix['health']]:.0f}  arm {v[ix['armor']]:.0f}  {weap_name} ammo {int(v[ix['selected_weapon_ammo']])}", (230, 230, 235)),
        (f"vel {v[ix['vel_forward']]:+.1f} {v[ix['vel_right']]:+.1f}  keys {','.join(keys) or '-'}", (200, 200, 210)),
        (f"kills {priv['kills']}  items {priv['items']}  dmg {priv['damage_taken']:.0f}", (200, 200, 210)),
        (f"rays  {counts}", (180, 200, 220)),
        (f"visible_entities {n_vis}/{S.MAX_ENTITIES}  (90 deg + LOS)", (180, 200, 220)),
    ]
    y = 8
    for text, col in lines:
        blit_text(surf, text, (8, y), col, font)
        y += 18
    ents = obs["visible_entities"]
    y += 6
    blit_text(surf, "entities in POV", (8, y), (180, 180, 120), font)
    y += 18
    shown = 0
    for k in range(S.MAX_ENTITIES):
        if not ents["mask"][k]:
            continue
        d, s, c = (float(x) for x in ents["features"][k][:3])
        name = S.ENTITY_TYPES[int(ents["entity_type"][k])]
        ang = math.degrees(math.atan2(s, c))
        blit_text(surf, f"  {name:16s} d={d * S.RAY_RANGE:5.0f}  {ang:+.0f} deg", (8, y), ENTITY_COLORS[entity_kind(name)], font)
        y += 16
        shown += 1
        if shown >= 8 or y > surf.get_height() - 28:
            break
    if shown == 0:
        blit_text(surf, "  (none — pickups outside 90 deg are rays only)", (8, y), (140, 140, 150), font)


def vitals_text(obs: dict, info: dict, steps: int, ret: float) -> str:
    v = obs["vitals"]
    ix = S.VITALS_INDEX
    keys = [n.replace("key_", "") for n in S.VITALS_FIELDS if n.startswith("key_") and v[ix[n]] > 0]
    ammo = int(v[ix["selected_weapon_ammo"]])
    n_vis = int(obs["visible_entities"]["mask"].sum())
    return (
        f"step {steps}  health {v[ix['health']]:.0f}  armor {v[ix['armor']]:.0f}  ammo {ammo}  "
        f"keys {','.join(keys) or '-'}  ents {n_vis}  speed {v[ix['vel_forward']]:.1f}  return {ret:+.2f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default=str(ROOT / "maps" / "two_rooms.wad") + ":MAP01", help="file.wad[:MAPxx]")
    ap.add_argument("--pool", action="store_true", help="sample Freedoom-first mix (ignores --map)")
    ap.add_argument("--generated-frac", type=float, default=None, help="pool: fraction of generated S0–S3 maps")
    ap.add_argument("--split", choices=("train", "val"), default="train", help="pool: Freedoom train MAP01–24 / val MAP25–32")
    ap.add_argument("--policy", choices=["sticky", "uniform", "human"], default="sticky")
    ap.add_argument("--mouse-sens", type=float, default=0.365, help="mouse yaw sensitivity (vanilla: no pitch)")
    ap.add_argument("--fps", type=float, default=None, help="decisions/s, 0 = unlimited (human defaults to 35)")
    ap.add_argument("--scale", type=int, default=4, help="RGB upscale factor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episode-ticks", type=int, default=None, help="cap tics (default 2100, or the sampled spec if --pool)")
    ap.add_argument("--episodes", type=int, default=0, help="stop after N episodes (0 = until you quit)")
    ap.add_argument("--screenshot", default=None, help="save a PNG after --shot-steps steps and exit (for testing)")
    ap.add_argument("--shot-steps", type=int, default=120)
    ap.add_argument("--depth", action="store_true", help="start in stretched depth view (as if you pressed Tab)")
    args = ap.parse_args()

    if args.fps is None:
        args.fps = 35.0 if args.policy == "human" else 35.0 / S.FRAME_SKIP
    skip = 1 if args.policy == "human" else S.FRAME_SKIP
    rng = np.random.default_rng(args.seed)
    tmp_dir = ROOT / ".mapgen_tmp"
    spec = None

    def open_env(next_spec=None):
        if args.pool:
            from training.pool import GENERATED_FRAC, make_training_env, sample_spec

            frac = GENERATED_FRAC if args.generated_frac is None else args.generated_frac
            spec_ = next_spec or sample_spec(rng, generated_frac=frac, split=args.split)
            ticks = args.episode_ticks if args.episode_ticks is not None else spec_.episode_ticks
            env_ = make_training_env(
                spec_,
                tmp_dir=tmp_dir,
                episode_ticks=ticks,
                frame_skip=skip,
                analog_look=(args.policy == "human"),
            )
            return env_, spec_
        wad, _, map_name = args.map.partition(":")
        ticks = 2100 if args.episode_ticks is None else args.episode_ticks
        env_ = make_env(
            wad,
            map_name or "MAP01",
            episode_ticks=ticks,
            frame_skip=skip,
            analog_look=(args.policy == "human"),
        )
        return env_, None

    env, spec = open_env()
    policy = HumanPolicy(args.mouse_sens) if args.policy == "human" else RandomPolicy(args.policy, args.seed)

    pygame.init()
    game_w, game_h = S.SCREEN_WIDTH * args.scale, S.SCREEN_HEIGHT * args.scale
    geo_w = game_h
    data_w = 360
    bar_h = 78 if args.policy == "human" else 56
    screen = pygame.display.set_mode((game_w + geo_w + data_w, game_h + bar_h))
    def _caption():
        who = "human" if args.policy == "human" else "random agent"
        if spec is None:
            return f"doom_env: {who}"
        if spec.source == "generated":
            return f"doom_env: {who}  {spec.stage} seed={spec.seed}"
        return f"doom_env: {who}  freedoom {spec.map}"

    pygame.display.set_caption(_caption())
    if args.policy == "human":
        policy.set_grab(True)
    font = pygame.font.SysFont("menlo,monaco,consolas,dejavu sans mono,monospace", 15)
    geo = pygame.Surface((geo_w, game_h)).convert()
    data = pygame.Surface((data_w, game_h)).convert()
    radar = pygame.Surface((data_w, 200)).convert()
    clock = pygame.time.Clock()

    episode, seed = 0, args.seed
    obs, info = env.reset(seed=seed)
    steps, ret, paused, show_depth, show_3d = 0, 0.0, False, args.depth, True
    last_event = "episode started"

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.MOUSEWHEEL and args.policy == "human":
                policy.queue_weapon = True
            elif ev.type == pygame.MOUSEBUTTONDOWN and args.policy == "human":
                if ev.button == 1 and not policy.grabbed:
                    policy.set_grab(True)
                    policy.ignore_click = True
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    if args.policy == "human" and policy.grabbed:
                        policy.set_grab(False)
                    else:
                        running = False
                elif ev.key == pygame.K_m and args.policy == "human":
                    policy.set_grab(not policy.grabbed)
                elif ev.key == pygame.K_p:
                    paused = not paused
                    if args.policy == "human":
                        policy.set_grab(not paused)
                elif ev.key == pygame.K_TAB:
                    show_depth = not show_depth
                elif ev.key == pygame.K_g:
                    show_3d = not show_3d
                elif ev.key == pygame.K_r:
                    seed += 1
                    if args.pool:
                        env.close()
                        env, spec = open_env()
                        pygame.display.set_caption(_caption())
                    obs, info = env.reset(seed=seed)
                    policy.reset()
                    steps, ret = 0, 0.0

        if not paused:
            if args.policy == "human":
                action, yaw = policy.act()
                obs, reward, terminated, truncated, info = env.step_human(action, yaw)
            else:
                obs, reward, terminated, truncated, info = env.step(policy.act())
            steps += 1
            ret += reward
            if terminated or truncated:
                st = env.privileged_state()
                outcome = "EXIT" if st["exit_reached"] else "DIED" if st["dead"] else "time up"
                last_event = f"episode {episode}: {outcome}, kills {st['kills']}, {steps} steps"
                print(last_event, flush=True)
                episode += 1
                if args.episodes and episode >= args.episodes:
                    break
                seed += 1
                if args.pool:
                    env.close()
                    env, spec = open_env()
                    pygame.display.set_caption(_caption())
                obs, info = env.reset(seed=seed)
                policy.reset()
                steps, ret = 0, 0.0

        # draw
        priv = env.privileged_state()
        try:
            sectors = env.sector_polygons()
        except RuntimeError:
            sectors = []
        if show_depth:
            img = stretch_depth(obs["depth"])
            caption = "depth stretched  (Tab back to RGB)"
        else:
            img = obs["rgb"]
            caption = "your gun (not an enemy)"
        frame = pygame.surfarray.make_surface(np.ascontiguousarray(img.swapaxes(0, 1)))
        screen.blit(pygame.transform.scale(frame, (game_w, game_h)), (0, 0))
        cx, cy = game_w // 2, game_h // 2
        pygame.draw.line(screen, (220, 220, 80), (cx - 6, cy), (cx + 6, cy), 1)
        pygame.draw.line(screen, (220, 220, 80), (cx, cy - 6), (cx, cy + 6), 1)
        screen.blit(font.render(caption, True, (255, 220, 80), (20, 20, 26)), (8, game_h - 22))
        if show_3d:
            draw_pov_3d(geo, obs, priv, sectors, font)
        else:
            draw_radar(geo, obs, font)
        screen.blit(geo, (game_w, 0))
        draw_data_panel(data, obs, priv, info, steps, ret, font)
        rh = radar.get_height()
        draw_radar(radar, obs, font)
        data.blit(radar, (0, game_h - rh))
        screen.blit(data, (game_w + geo_w, 0))
        pygame.draw.rect(screen, (20, 20, 26), (0, game_h, game_w + geo_w + data_w, bar_h))
        screen.blit(font.render(vitals_text(obs, info, steps, ret), True, (230, 230, 235), (20, 20, 26)), (8, game_h + 8))
        screen.blit(font.render(
            f"last: {last_event}   [P pause | R reset | Tab depth | G 3D/radar | ESC quit]",
            True, (190, 190, 200), (20, 20, 26)), (8, game_h + 30))
        if args.policy == "human":
            grab = "mouse ON" if policy.grabbed else "mouse OFF (M or click window)"
            screen.blit(font.render(
                f"FPS: WASD (A/D strafe)  mouse yaw (no pitch)  LMB shoot  RMB/Space USE  wheel weapon  grab ({grab})",
                True, (210, 210, 180), (20, 20, 26)), (8, game_h + 52))
        pygame.display.flip()

        if args.screenshot and steps >= args.shot_steps:
            pygame.image.save(screen, args.screenshot)
            print("saved", args.screenshot)
            break
        if args.fps > 0:
            clock.tick(args.fps)

    env.close()
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
