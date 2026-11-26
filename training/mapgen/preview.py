"""Top-down PNG preview of a generated map (headless-safe)."""

from __future__ import annotations

import os
from pathlib import Path

from .layout import BuiltMap
from .wadbuild import THING_PLAYER

_COLOR = {
    "room": (58, 64, 78),
    "door": (186, 122, 48),
    "exit": (176, 52, 52),
}
_BG = (18, 18, 22)
_WALL = (220, 220, 226)
_PLAYER = (80, 220, 120)
_EXIT_MARK = (255, 80, 80)


def _init_pygame():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    pygame.display.init()
    pygame.font.init()
    return pygame


def render_preview(built: BuiltMap, path: str | Path, size: int = 800, pad: int = 24) -> Path:
    pygame = _init_pygame()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pts = [p for s in built.map.sectors for p in s["poly"]]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max(max_x - min_x, 1)
    h = max(max_y - min_y, 1)
    scale = (size - 2 * pad) / max(w, h)
    img_w = int(w * scale) + 2 * pad
    img_h = int(h * scale) + 2 * pad

    def xy(p):
        x = pad + (p[0] - min_x) * scale
        # pygame y grows down; map y grows up
        y = pad + (max_y - p[1]) * scale
        return int(round(x)), int(round(y))

    surf = pygame.Surface((img_w, img_h))
    surf.fill(_BG)
    for s in built.map.sectors:
        poly = [xy(p) for p in s["poly"]]
        if len(poly) >= 3:
            pygame.draw.polygon(surf, _COLOR.get(s["kind"], _COLOR["room"]), poly)
            pygame.draw.polygon(surf, _WALL, poly, width=1)

    for x, y, ang, typ in built.map.things:
        px, py = xy((x, y))
        if typ == THING_PLAYER:
            pygame.draw.circle(surf, _PLAYER, (px, py), 5)
            rad = 12
            import math

            a = math.radians(ang)
            pygame.draw.line(
                surf,
                _PLAYER,
                (px, py),
                (int(px + math.cos(a) * rad), int(py - math.sin(a) * rad)),
                2,
            )

    ex, ey = xy(built.exit_xy)
    pygame.draw.circle(surf, _EXIT_MARK, (ex, ey), 4)

    pygame.image.save(surf, str(path))
    return path
