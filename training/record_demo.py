"""Watch (or record to MP4) the privileged teacher bot playing generated levels, drawn with
the watch_random viewer (RGB | 3D ray/sector view | data panel + radar).

    uv run python training/record_demo.py --live                      # window, plays by itself forever
    uv run python training/record_demo.py --live --tier 3 --fps 18    # only tier-3 levels, faster
    uv run python training/record_demo.py --tier 3 --seed 5 --out demo.mp4   # headless MP4

The teacher sees the whole map (training-only scaffolding); it is NOT the submission agent.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

if "--live" not in sys.argv:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "submission"), str(ROOT / "env")]

import numpy as np  # noqa: E402
import pygame  # noqa: E402
import torch  # noqa: E402

import denv  # noqa: E402
import doom_env  # noqa: E402
import watch_random as W  # noqa: E402
from brain.model import comps_to_buttons  # noqa: E402

_make = doom_env.make_env


def _make_rgb(*a, **k):
    k["render_buffers"] = True
    return _make(*a, **k)


doom_env.make_env = _make_rgb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="show a window instead of writing an MP4")
    ap.add_argument("--fps", type=float, default=35 / 4, help="live playback decisions/s (8.75 = real time)")
    ap.add_argument("--tier", type=int, default=-1, help="0-4, or -1 for a random tier each level")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=0, help="0 = forever in --live, 1 for MP4")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--out", default="demo.mp4")
    args = ap.parse_args()

    if args.episodes == 0 and not args.live:
        args.episodes = 1
    w = [1.0] * 5 if args.tier < 0 else [float(i == args.tier) for i in range(5)]
    env = denv.TrainEnv(0, args.seed, tier_weights=w)
    S = W.S
    pygame.init()
    game_w, game_h = S.SCREEN_WIDTH * args.scale, S.SCREEN_HEIGHT * args.scale
    geo_w, data_w, bar_h = game_h, 360, 56
    W_, H_ = game_w + geo_w + data_w, game_h + bar_h
    W_, H_ = W_ + W_ % 2, H_ + H_ % 2
    if args.live:
        screen = pygame.display.set_mode((W_, H_))
        pygame.display.set_caption("teacher bot on generated levels  (P pause, N next level, Esc quit)")
    else:
        pygame.display.set_mode((1, 1))
        screen = pygame.Surface((W_, H_))
    font = pygame.font.SysFont("dejavu sans mono,monospace", 14)
    geo = pygame.Surface((geo_w, game_h))
    data = pygame.Surface((data_w, game_h))
    radar = pygame.Surface((data_w, 200))

    fps = 35 / S.FRAME_SKIP
    ff = None if args.live else subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W_}x{H_}",
         "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out],
        stdin=subprocess.PIPE)

    clock = pygame.time.Clock()
    quit_, paused = False, False
    last = "episode started"
    ep = 0
    while not quit_ and (args.episodes == 0 or ep < args.episodes):
        obs = env.reset()
        steps, ret = 0, 0.0
        info = {}
        skip = False
        for _ in range(args.max_steps):
            if args.live:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                        quit_ = True
                    elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_p:
                        paused = not paused
                    elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_n:
                        skip = True
                if quit_ or skip:
                    break
                if paused:
                    clock.tick(20)
                    continue
            comps, _ok = env.teacher_action()
            buttons = comps_to_buttons(torch.tensor(np.asarray(comps)[None]))[0].numpy()
            obs, r, done, info = env.step(buttons)
            steps += 1
            ret += r
            priv = env.env.privileged_state()
            try:
                sectors = env.env.sector_polygons()
            except RuntimeError:
                sectors = []
            screen.fill((20, 20, 26))
            frame = pygame.surfarray.make_surface(np.ascontiguousarray(obs["rgb"].swapaxes(0, 1)))
            screen.blit(pygame.transform.scale(frame, (game_w, game_h)), (0, 0))
            W.draw_pov_3d(geo, obs, priv, sectors, font)
            screen.blit(geo, (game_w, 0))
            W.draw_data_panel(data, obs, priv, info, steps, ret, font)
            W.draw_radar(radar, obs, font)
            data.blit(radar, (0, game_h - radar.get_height()))
            screen.blit(data, (game_w + geo_w, 0))
            screen.blit(font.render(W.vitals_text(obs, info, steps, ret), True, (230, 230, 235)), (8, game_h + 6))
            screen.blit(font.render(f"teacher bot  |  generated tier-{env.tier} level  |  last: {last}",
                                    True, (190, 190, 200)), (8, game_h + 28))
            if args.live:
                pygame.display.flip()
                clock.tick(args.fps)
            else:
                ff.stdin.write(pygame.surfarray.array3d(screen).swapaxes(0, 1).tobytes())
            if done:
                break
        if quit_:
            break
        e = env.ep
        last = f"ep {ep}: {'EXIT' if e['exit'] else 'DIED' if e['death'] else 'time up'}, keys {e['keys']}, doors {e['doors']}, kills {e['kills']}, {steps} steps"
        print(f"tier {env.tier} " + last, flush=True)
        ep += 1
        if ff is not None:
            for _ in range(int(fps * 1.5)):  # hold last frame
                ff.stdin.write(pygame.surfarray.array3d(screen).swapaxes(0, 1).tobytes())
        else:
            env.level_eps = 10 ** 9  # new level every episode when watching
            pygame.time.wait(1200)
    if ff is not None:
        ff.stdin.close()
        ff.wait()
    pygame.quit()
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
