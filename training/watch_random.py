"""Watch a random agent play through the official doom_env wrapper, with a live sensor view.

Throwaway demo for checking your local setup: it is NOT the submission agent.

    uv run python training/watch_random.py                      # sticky-random movement, real time
    uv run python training/watch_random.py --policy uniform     # pure uniform random buttons
    uv run python training/watch_random.py --map maps/your.wad:MAP01 --fps 0   # as fast as possible

Left:  the game's 160x120 RGB frame (press D to toggle the depth buffer).
Right: ego-centric radar. You are the white dot, facing UP. Each of the 64 rays is drawn to its hit
       distance and coloured by hit type; dots are visible_entities.
Keys:  SPACE pause, R new episode, D rgb/depth, ESC quit.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

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


def entity_kind(name: str) -> str:
    if name.startswith("enemy"):
        return "enemy"
    if name == "projectile":
        return "projectile"
    if name == "key":
        return "key"
    return "item"


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
    for k in range(S.MAX_ENTITIES):
        if not ents["mask"][k]:
            continue
        d, s, c = ents["features"][k][:3]
        name = S.ENTITY_TYPES[int(ents["entity_type"][k])]
        col = ENTITY_COLORS[entity_kind(name)]
        px = cx - s * min(d, 1.0) * radius
        py = cy - c * min(d, 1.0) * radius
        pygame.draw.circle(surf, col, (int(px), int(py)), 6, 2)
    # 90 degree field of view
    for sign in (-1, 1):
        a = math.radians(45 * sign)
        pygame.draw.line(surf, (70, 70, 110), (cx, cy), (cx - math.sin(a) * radius, cy - math.cos(a) * radius), 1)
    pygame.draw.circle(surf, (255, 255, 255), (cx, cy), 5)
    surf.blit(font.render("ego-centric rays (facing up)", True, (160, 160, 170)), (8, 6))
    ly = h - 20 * len(RAY_COLORS) - 6
    for t, name in enumerate(HIT_NAMES):
        pygame.draw.circle(surf, RAY_COLORS[t], (16, ly + t * 20 + 8), 5)
        surf.blit(font.render(name, True, (190, 190, 200)), (28, ly + t * 20))


def vitals_text(obs: dict, info: dict, steps: int, ret: float) -> str:
    v = obs["vitals"]
    ix = S.VITALS_INDEX
    keys = [n.replace("key_", "") for n in S.VITALS_FIELDS if n.startswith("key_") and v[ix[n]] > 0]
    ammo = int(v[ix["selected_weapon_ammo"]])
    return (
        f"step {steps}  health {v[ix['health']]:.0f}  armor {v[ix['armor']]:.0f}  ammo {ammo}  "
        f"keys {','.join(keys) or '-'}  speed {v[ix['vel_forward']]:.1f}  return {ret:+.2f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default=str(ROOT / "maps" / "two_rooms.wad") + ":MAP01", help="file.wad[:MAPxx]")
    ap.add_argument("--policy", choices=["sticky", "uniform"], default="sticky")
    ap.add_argument("--fps", type=float, default=S.FRAME_SKIP and 35 / S.FRAME_SKIP, help="decisions/s, 0 = unlimited")
    ap.add_argument("--scale", type=int, default=4, help="RGB upscale factor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episode-ticks", type=int, default=2100)
    ap.add_argument("--episodes", type=int, default=0, help="stop after N episodes (0 = until you quit)")
    ap.add_argument("--screenshot", default=None, help="save a PNG after --shot-steps steps and exit (for testing)")
    ap.add_argument("--shot-steps", type=int, default=120)
    args = ap.parse_args()

    wad, _, map_name = args.map.partition(":")
    env = make_env(wad, map_name or "MAP01", episode_ticks=args.episode_ticks)
    policy = RandomPolicy(args.policy, args.seed)

    pygame.init()
    game_w, game_h = S.SCREEN_WIDTH * args.scale, S.SCREEN_HEIGHT * args.scale
    radar_w = game_h
    bar_h = 56
    screen = pygame.display.set_mode((game_w + radar_w, game_h + bar_h))
    pygame.display.set_caption("doom_env: random agent")
    font = pygame.font.SysFont("monospace", 15)
    radar = pygame.Surface((radar_w, game_h))
    clock = pygame.time.Clock()

    episode, seed = 0, args.seed
    obs, info = env.reset(seed=seed)
    steps, ret, paused, show_depth = 0, 0.0, False, False
    last_event = "episode started"

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_d:
                    show_depth = not show_depth
                elif ev.key == pygame.K_r:
                    seed += 1
                    obs, info = env.reset(seed=seed)
                    policy.reset()
                    steps, ret = 0, 0.0

        if not paused:
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
                obs, info = env.reset(seed=seed)
                policy.reset()
                steps, ret = 0, 0.0

        # draw
        if show_depth:
            img = np.repeat(obs["depth"][:, :, None], 3, axis=2)
        else:
            img = obs["rgb"]
        frame = pygame.surfarray.make_surface(np.ascontiguousarray(img.swapaxes(0, 1)))
        screen.blit(pygame.transform.scale(frame, (game_w, game_h)), (0, 0))
        draw_radar(radar, obs, font)
        screen.blit(radar, (game_w, 0))
        pygame.draw.rect(screen, (20, 20, 26), (0, game_h, game_w + radar_w, bar_h))
        screen.blit(font.render(vitals_text(obs, info, steps, ret), True, (230, 230, 235)), (8, game_h + 8))
        screen.blit(font.render(f"last: {last_event}   [SPACE pause | R reset | D depth | ESC quit]",
                                True, (150, 150, 160)), (8, game_h + 30))
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
