"""Watch (or record to MP4) generated levels with the watch_random viewer.

    uv run python training/record_demo.py --live --agent submission/weights/policy.pt
    uv run python training/record_demo.py --live                      # privileged teacher
    uv run python training/record_demo.py --live --tier 3 --fps 18
    uv run python training/record_demo.py --tier 3 --seed 5 --out demo.mp4

`--agent` plays the submitted DoomAgent and draws its own EgoMap in the middle panel.
Without it, the privileged teacher drives (training-only; it sees the whole map).
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


def draw_egomap(surf, mem, font, view_cells: int = 96):
    """Draw the agent's OWN memory map (the EgoMap DoomNet reads), centred on its estimated pose."""
    import math

    g = mem.grid[0].cpu().numpy()  # (8, G, G) indexed [ch, y, x]
    G = g.shape[1]
    x0, y0, hd = (float(v) for v in mem.pose[0].cpu())
    cx, cy = int(x0 / mem.cell + G / 2), int(y0 / mem.cell + G / 2)
    h = view_cells // 2
    ys, xs = np.clip(np.arange(cy - h, cy + h), 0, G - 1), np.clip(np.arange(cx - h, cx + h), 0, G - 1)
    win = g[:, ys][:, :, xs]
    img = np.full((view_cells, view_cells, 3), 14, np.uint8)
    cols = [("wall", 1, (225, 225, 230)), ("door", 2, (235, 190, 60)), ("locked", 3, (235, 60, 50)),
            ("item", 4, (80, 200, 90)), ("enemy", 5, (255, 140, 0)), ("key", 6, (220, 80, 240)),
            ("trail", 7, (40, 140, 160))]
    img[win[0] > 0.5] = (62, 62, 72)  # free space it has seen
    for ch in (7, 4, 2, 1, 3, 5, 6):
        img[win[ch] > 0.5] = next(c for _, k, c in cols if k == ch)
    img = img[::-1]  # world y up
    S_ = surf.get_width()
    oy = (surf.get_height() - S_) // 2
    surf.fill((14, 14, 18))
    pic = pygame.surfarray.make_surface(np.ascontiguousarray(img.swapaxes(0, 1)))
    surf.blit(pygame.transform.scale(pic, (S_, S_)), (0, oy))
    c = S_ / 2
    pts = [(c + math.cos(hd + a) * r, oy + c - math.sin(hd + a) * r) for a, r in ((0, 14), (2.5, 8), (-2.5, 8))]
    pygame.draw.polygon(surf, (255, 255, 255), pts)
    surf.blit(font.render("what the agent remembers (its own map)", True, (235, 235, 240)), (8, 6))
    x = 8
    for name, _, col in cols:
        pygame.draw.rect(surf, col, (x, surf.get_height() - 18, 10, 10))
        surf.blit(font.render(name, True, (200, 200, 205)), (x + 13, surf.get_height() - 22))
        x += 13 + font.size(name)[0] + 8


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
    ap.add_argument("--agent", default="", help="play with the trained DoomAgent using this checkpoint "
                                                 "(e.g. submission/weights/policy.pt) instead of the teacher; "
                                                 "the middle panel then shows the agent's own EgoMap memory")
    ap.add_argument("--label", default="", help="caption for the agent (e.g. 'v4, update 2600')")
    args = ap.parse_args()

    if args.episodes == 0 and not args.live:
        args.episodes = 1
    w = [1.0] * 5 if args.tier < 0 else [float(i == args.tier) for i in range(5)]
    env = denv.TrainEnv(0, args.seed, tier_weights=w)
    agent = None
    if args.agent:
        os.environ["DOOM_AGENT_POLICY"] = str(Path(args.agent).resolve())
        from agent import DoomAgent

        agent = DoomAgent(None, None, "cpu")
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
        if agent is not None:
            agent.reset()
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
            if agent is not None:
                buttons = agent.act(obs)
            else:
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
            if agent is not None:
                draw_egomap(geo, agent.perception.mem, font)
            else:
                W.draw_pov_3d(geo, obs, priv, sectors, font)
            screen.blit(geo, (game_w, 0))
            W.draw_data_panel(data, obs, priv, info, steps, ret, font)
            W.draw_radar(radar, obs, font)
            data.blit(radar, (0, game_h - radar.get_height()))
            screen.blit(data, (game_w + geo_w, 0))
            screen.blit(font.render(W.vitals_text(obs, info, steps, ret), True, (230, 230, 235)), (8, game_h + 6))
            who = f"TRAINED AGENT {args.label}".strip() if agent is not None else "teacher bot"
            screen.blit(font.render(f"{who}  |  unseen tier-{env.tier} level  |  last: {last}",
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
