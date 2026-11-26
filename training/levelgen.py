"""Procedural DOOM level generator for training (UDMF / ZDoom namespace).

Every level is laid out on a grid of 32-unit cells. Rooms sit in blocks of a coarse grid and are
connected by corridors along a random spanning tree (plus a few loops). Connections may carry a door
or a locked door; keys are hidden in side branches so the agent has to explore, remember and
backtrack. The exit is a walk-over pad or an exit switch.

Tiers follow docs/EVALUATION.md:
    0  bootstrap: one or two rooms, no monsters (learn to move / find the exit)
    1  combat & survival: big arenas, many monsters, pickups
    2  multi-room exploration: 4-9 rooms, doors, dead ends
    3  long-horizon progression: keycard -> locked door -> exit
    4  full gauntlet: 2-3 locks, ambushes, scarce resources

The generator also returns the cell grid and the key/exit cells, so the training env can compute
geodesic distances for reward shaping. That information is used only during training; the agent
itself only ever sees the official doom_env observations.

    uv run python training/levelgen.py --tier 3 --seed 7 --out /tmp/t3.wad
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "env"))
from doom_env.wad import Lump, write_wad  # noqa: E402

CELL = 32  # map units per grid cell
SOLID = -1

# ---------------------------------------------------------------------------------- content tables
WALLS = ["STARTAN2", "STONE2", "BROWN1", "BRICK7", "GRAY1", "METAL", "STARG3", "TEKGREN2", "ICKWALL3",
         "CEMENT9", "BROWN96", "COMPBLUE", "MARBLE1", "BRICK10", "SILVER1", "STONE3", "WOOD1", "PIPE4"]
FLOORS = ["FLOOR0_1", "FLOOR4_8", "FLAT5_4", "FLAT1", "FLAT14", "FLOOR5_1", "FLAT20", "FLOOR1_1", "RROCK09"]
CEILS = ["CEIL1_1", "CEIL3_5", "CEIL5_1", "FLAT1", "TLITE6_4"]

# key colour -> (ZDoom lock number accepting card OR skull, card thing, skull thing, door texture)
KEYS = {
    "red": (129, 13, 38, "DOORRED"),
    "blue": (130, 5, 40, "DOORBLU"),
    "yellow": (131, 6, 39, "DOORYEL"),
}

# thing type ids (Doom 2 / Freedoom 2)
PLAYER = 1
MONSTERS = {  # type: (name, radius, tier weight row [t1, t2, t3, t4])
    3004: ("zombieman", 20, [5, 5, 4, 3]),
    9: ("shotgunguy", 20, [4, 4, 4, 4]),
    3001: ("imp", 20, [5, 4, 4, 5]),
    3002: ("demon", 30, [3, 3, 3, 3]),
    58: ("spectre", 30, [1, 1, 1, 2]),
    3006: ("lostsoul", 16, [1, 1, 1, 2]),
    65: ("chaingunner", 20, [1, 1, 1, 2]),
    3005: ("cacodemon", 31, [1, 0, 1, 2]),
}
HEALTH = [2011, 2011, 2012, 2014]  # stimpack, stimpack, medikit, health bonus
ARMOR = [2018, 2015]  # green armor, armor bonus
AMMO = [2007, 2007, 2008, 2048, 2049]  # clip, clip, shells, ammo box, shell box
WEAPONS = [2001, 2002]  # shotgun, chaingun

TIERS = {
    #     grid choices         rooms   block  locks  monsters/room  items/room  loop_p  door_p
    0: dict(grids=[(1, 1), (1, 2)], rooms=(1, 2), block=(14, 20), locks=0, mon=(0, 0), items=(0, 1), loop=0.0, door=0.5),
    1: dict(grids=[(1, 1), (1, 2)], rooms=(1, 2), block=(22, 30), locks=0, mon=(3, 6), items=(2, 4), loop=0.0, door=0.3),
    2: dict(grids=[(2, 2), (2, 3), (3, 3)], rooms=(4, 8), block=(14, 20), locks=0, mon=(0, 2), items=(0, 2), loop=0.25, door=0.6),
    3: dict(grids=[(2, 3), (3, 3), (3, 4)], rooms=(5, 9), block=(14, 20), locks=1, mon=(0, 2), items=(0, 2), loop=0.2, door=0.6),
    4: dict(grids=[(3, 3), (3, 4), (4, 4)], rooms=(6, 10), block=(14, 20), locks=(2, 3), mon=(0, 2), items=(1, 2), loop=0.2, door=0.6),
}


@dataclass
class Level:
    """Everything the training env needs to know about a generated map."""

    textmap: str
    grid: np.ndarray  # (H, W) sector label per cell, -1 = solid
    walkable: np.ndarray  # (H, W) bool: cells the player can stand in (doors count as walkable)
    region: np.ndarray  # (H, W) lock region per cell (0 = start region), -1 = solid
    exit_cells: list[tuple[int, int]]  # (gy, gx)
    key_cells: list[tuple[str, tuple[int, int]]]  # (colour, (gy, gx)) in the order they are needed
    lock_order: list[str]  # colours of the locks from start towards the exit
    start: tuple[int, int]
    tier: int
    num_monsters: int
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------- layout
def _spanning_tree(rng, rows, cols, n_rooms):
    start = (int(rng.integers(rows)), int(rng.integers(cols)))
    rooms = {start}
    edges = []
    frontier = []

    def push(c):
        r, k = c
        for dr, dk in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            n = (r + dr, k + dk)
            if 0 <= n[0] < rows and 0 <= n[1] < cols:
                frontier.append((c, n))

    push(start)
    while len(rooms) < n_rooms and frontier:
        i = int(rng.integers(len(frontier)))
        a, b = frontier.pop(i)
        if b in rooms:
            continue
        rooms.add(b)
        edges.append((a, b))
        push(b)
    return start, rooms, edges


def _tree_dist(edges, src):
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    dist, parent = {src: 0}, {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in dist:
                dist[v], parent[v] = dist[u] + 1, u
                q.append(v)
    return dist, parent, adj


def generate(tier: int, seed: int) -> Level:
    """Generate one level. Deterministic for (tier, seed)."""
    rng = np.random.default_rng([tier, seed])
    cfg = TIERS[tier]
    rows, cols = cfg["grids"][int(rng.integers(len(cfg["grids"])))]
    n_rooms = int(min(rows * cols, rng.integers(cfg["rooms"][0], cfg["rooms"][1] + 1)))
    n_rooms = max(n_rooms, 1)
    B = int(rng.integers(cfg["block"][0], cfg["block"][1] + 1))
    cw = int(rng.integers(3, 5))  # corridor width in cells
    c0 = B // 2 - cw // 2  # corridor band inside a block: [c0, c0 + cw)

    start, rooms, tree = _spanning_tree(rng, rows, cols, n_rooms)
    dist, parent, adj = _tree_dist(tree, start)
    exit_room = max(rooms, key=lambda r: (dist[r], rng.random()))

    # ---- locks along the start -> exit path
    path = [exit_room]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()
    path_edges = list(zip(path[:-1], path[1:]))
    nlocks = cfg["locks"] if isinstance(cfg["locks"], int) else int(rng.integers(cfg["locks"][0], cfg["locks"][1] + 1))
    nlocks = min(nlocks, len(path_edges), 3)
    lock_idx = sorted(rng.choice(len(path_edges), size=nlocks, replace=False).tolist()) if nlocks else []
    colours = list(rng.permutation(list(KEYS)))[:nlocks]
    locked = {frozenset(path_edges[i]): colours[j] for j, i in enumerate(lock_idx)}

    # regions: components of the tree once locked edges are cut, numbered along the main path
    region_of = {}
    for ridx, seed_room in enumerate([path[0]] + [path_edges[i][1] for i in lock_idx]):
        q = deque([seed_room])
        region_of[seed_room] = ridx
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                if v not in region_of and frozenset((u, v)) not in locked:
                    region_of[v] = ridx
                    q.append(v)

    # extra loop edges, only inside one region (never bypass a lock)
    conns = list(tree)
    for r in rooms:
        for dr, dk in ((0, 1), (1, 0)):
            n = (r[0] + dr, r[1] + dk)
            if n in rooms and (r, n) not in conns and (n, r) not in conns and region_of[r] == region_of[n]:
                if rng.random() < cfg["loop"]:
                    conns.append((r, n))

    # ---- rasterise rooms and corridors
    H, W = rows * B + 2, cols * B + 2
    grid = np.full((H, W), SOLID, np.int32)
    sectors: list[dict] = []
    room_rect = {}

    def new_sector(kind, **kw):
        sectors.append(dict(kind=kind, **kw))
        return len(sectors) - 1

    for r in sorted(rooms):
        gy, gx = 1 + r[0] * B, 1 + r[1] * B
        x0 = int(rng.integers(2, max(3, c0)))
        x1 = int(rng.integers(min(c0 + cw + 1, B - 2), B - 1))
        y0 = int(rng.integers(2, max(3, c0)))
        y1 = int(rng.integers(min(c0 + cw + 1, B - 2), B - 1))
        ceil = int(rng.choice([128, 160, 192, 256])) if tier != 1 else int(rng.choice([192, 256]))
        sid = new_sector("room", floor=0, ceil=ceil, light=int(rng.integers(128, 225)),
                         ftex=str(rng.choice(FLOORS)), ctex=str(rng.choice(CEILS)), wall=str(rng.choice(WALLS)), room=r)
        grid[gy + y0: gy + y1, gx + x0: gx + x1] = sid
        room_rect[r] = (gy + y0, gx + x0, gy + y1, gx + x1, sid)
        # pillars in larger rooms (cover for combat, clutter for navigation)
        h, w = y1 - y0, x1 - x0
        if h >= 9 and w >= 9 and rng.random() < 0.6:
            for _ in range(int(rng.integers(1, 4 if tier != 1 else 7))):
                ph, pw = int(rng.integers(1, 3)), int(rng.integers(1, 3))
                py = int(rng.integers(gy + y0 + 2, gy + y1 - 2 - ph + 1))
                px = int(rng.integers(gx + x0 + 2, gx + x1 - 2 - pw + 1))
                # keep the corridor bands (rows and columns through the block centre) clear
                def hits_band(lo, n):
                    return lo < c0 + cw + 1 and lo + n > c0 - 1
                if not (hits_band(py - gy, ph) or hits_band(px - gx, pw)):
                    grid[py: py + ph, px: px + pw] = SOLID

    door_lines = {}  # door sector id -> lock colour or None
    for a, b in conns:
        (ra, rb) = sorted((a, b))
        ya0, xa0, ya1, xa1, sa = room_rect[ra]
        yb0, xb0, yb1, xb1, sb = room_rect[rb]
        csid = new_sector("corridor", floor=0, ceil=128, light=int(rng.integers(112, 200)), ftex=str(rng.choice(FLOORS)),
                          ctex=str(rng.choice(CEILS)), wall=sectors[sa]["wall"])
        colour = locked.get(frozenset((a, b)))
        has_door = colour is not None or rng.random() < cfg["door"]
        if ra[0] == rb[0]:  # horizontal neighbours
            band = slice(1 + ra[0] * B + c0, 1 + ra[0] * B + c0 + cw)
            xs, xe = xa1, xb0
            grid[band, xs:xe] = csid
            if has_door:
                mid = (xs + xe) // 2
                dsid = new_sector("door", floor=0, ceil=0, light=160, ftex="FLAT20", ctex="FLAT20",
                                  wall="DOORTRAK", lock=colour)
                grid[band, mid] = dsid
                door_lines[dsid] = colour
        else:  # vertical neighbours
            band = slice(1 + ra[1] * B + c0, 1 + ra[1] * B + c0 + cw)
            ys, ye = ya1, yb0
            grid[ys:ye, band] = csid
            if has_door:
                mid = (ys + ye) // 2
                dsid = new_sector("door", floor=0, ceil=0, light=160, ftex="FLAT20", ctex="FLAT20",
                                  wall="DOORTRAK", lock=colour)
                grid[mid, band] = dsid
                door_lines[dsid] = colour

    walkable = grid != SOLID
    room_cells = {r: [(y, x) for y in range(room_rect[r][0] + 1, room_rect[r][2] - 1)
                      for x in range(room_rect[r][1] + 1, room_rect[r][3] - 1) if grid[y, x] == room_rect[r][4]]
                  for r in rooms}
    used: set[tuple[int, int]] = set()

    def take(r, margin=1):
        cells = [c for c in room_cells[r] if all((c[0] + dy, c[1] + dx) not in used
                                                    for dy in range(-margin, margin + 1) for dx in range(-margin, margin + 1))]
        if not cells:
            cells = room_cells[r]
        c = cells[int(rng.integers(len(cells)))]
        used.add(c)
        return c

    # ---- player start
    things = []
    sc = take(start, margin=2)
    things.append((sc, PLAYER, int(rng.integers(0, 8)) * 45))

    # ---- exit: walk-over pad (2x2) or wall switch
    exit_cells = []
    special_edges = {}  # ((y, x), (y2, x2)) unit edge between a walkable cell and a solid cell -> attrs
    ey0, ex0, ey1, ex1, esid = room_rect[exit_room]
    use_switch = rng.random() < 0.3
    if use_switch:
        # pick a wall cell run on the room's boundary
        side = int(rng.integers(4))
        if side in (0, 1):  # south / north wall
            y = ey0 if side == 0 else ey1 - 1
            x = int(rng.integers(ex0 + 1, ex1 - 2))
            ny = y - 1 if side == 0 else y + 1
            for xx in (x, x + 1):
                if grid[y, xx] == esid and grid[ny, xx] == SOLID:
                    special_edges[((y, xx), (ny, xx))] = "exit_switch"
                    exit_cells.append((y, xx))
        else:
            x = ex0 if side == 2 else ex1 - 1
            y = int(rng.integers(ey0 + 1, ey1 - 2))
            nx = x - 1 if side == 2 else x + 1
            for yy in (y, y + 1):
                if grid[yy, x] == esid and grid[yy, nx] == SOLID:
                    special_edges[((yy, x), (yy, nx))] = "exit_switch"
                    exit_cells.append((yy, x))
    if not exit_cells:
        far = sorted(room_cells[exit_room], key=lambda c: -abs(c[0] - sc[0]) - abs(c[1] - sc[1]))
        for c in far:
            y, x = c
            block = [(y, x), (y + 1, x), (y, x + 1), (y + 1, x + 1)]
            if all(grid[b] == esid for b in block) and all(
                    grid[y + dy, x + dx] == esid for dy in (-1, 0, 1, 2) for dx in (-1, 0, 1, 2)) and \
                    abs(y - sc[0]) + abs(x - sc[1]) > 4:
                psid = new_sector("exit", floor=0, ceil=sectors[esid]["ceil"], light=255, ftex="GATE3",
                                  ctex=sectors[esid]["ctex"], wall=sectors[esid]["wall"])
                for b in block:
                    grid[b] = psid
                    used.add(b)
                exit_cells = block
                break
    if not exit_cells:  # tiny room: fall back to a switch anywhere on its south wall
        for x in range(ex0, ex1):
            if grid[ey0 - 1, x] == SOLID:
                special_edges[((ey0, x), (ey0 - 1, x))] = "exit_switch"
                exit_cells.append((ey0, x))
                break
    for c in exit_cells:
        used.add(c)

    # ---- keys (one per lock), hidden in side branches of the region before the lock
    key_cells = []
    for j, i in enumerate(lock_idx):
        colour = colours[j]
        allowed = [r for r in rooms if region_of[r] <= j and r != exit_room]
        same = [r for r in allowed if region_of[r] == j]
        leaves = [r for r in same if len(adj.get(r, [])) == 1 and r not in path and r != start]
        pool = leaves or [r for r in same if r != start] or same or allowed
        kr = pool[int(rng.integers(len(pool)))]
        kc = take(kr)
        lock_no, card, skull, _ = KEYS[colour]
        things.append((kc, card if rng.random() < 0.7 else skull, 0))
        key_cells.append((colour, kc))

    # ---- monsters and pickups
    tcol = min(max(tier, 1), 4) - 1
    mtypes = list(MONSTERS)
    mweights = np.array([MONSTERS[t][2][tcol] for t in mtypes], float)
    mweights /= mweights.sum()
    nmon = 0
    for r in rooms:
        lo, hi = cfg["mon"]
        n = int(rng.integers(lo, hi + 1))
        if r == start and tier not in (1,):
            n = min(n, 1 if tier >= 4 else 0)
        for _ in range(n):
            things.append((take(r), int(rng.choice(mtypes, p=mweights)), int(rng.integers(0, 8)) * 45))
            nmon += 1
        lo, hi = cfg["items"]
        for _ in range(int(rng.integers(lo, hi + 1))):
            kind = rng.random()
            pool = HEALTH if kind < 0.4 else AMMO if kind < 0.8 else ARMOR if kind < 0.93 else WEAPONS
            things.append((take(r), int(rng.choice(pool)), 0))

    # ---- regions per cell (for key-aware shaping)
    region = np.full((H, W), -1, np.int32)
    for r in rooms:
        y0, x0, y1, x1, _ = room_rect[r]
        region[y0:y1, x0:x1] = np.where(walkable[y0:y1, x0:x1], region_of[r], -1)
    # corridors/doors take the region of the lower-numbered room they connect; locked doors belong to
    # the region behind them so that "reaching the door" needs the key in hand.
    for a, b in conns:
        lo_r = min(region_of[a], region_of[b])
        hi_r = max(region_of[a], region_of[b])
        ra, rb = sorted((a, b))
        ya0, xa0, ya1, xa1, _ = room_rect[ra]
        yb0, xb0, yb1, xb1, _ = room_rect[rb]
        if ra[0] == rb[0]:
            sl = (slice(1 + ra[0] * B + c0, 1 + ra[0] * B + c0 + cw), slice(xa1, xb0))
        else:
            sl = (slice(ya1, yb0), slice(1 + ra[1] * B + c0, 1 + ra[1] * B + c0 + cw))
        sub = grid[sl]
        reg = np.full(sub.shape, lo_r, np.int32)
        for d in np.unique(sub):
            if d >= 0 and sectors[d]["kind"] == "door" and sectors[d].get("lock"):
                reg[sub == d] = hi_r
        region[sl] = np.where(sub != SOLID, reg, region[sl])
    for c in exit_cells:
        region[c] = region_of[exit_room]

    text = _textmap(grid, sectors, things, special_edges)
    return Level(textmap=text, grid=grid, walkable=walkable, region=region, exit_cells=exit_cells,
                 key_cells=key_cells, lock_order=colours, start=sc, tier=tier, num_monsters=nmon,
                 meta=dict(rows=rows, cols=cols, rooms=len(rooms), block=B, exit_switch=use_switch and bool(special_edges)))


# ---------------------------------------------------------------------------------- UDMF writer
def _textmap(grid, sectors, things, special_edges) -> str:
    H, W = grid.shape
    verts: dict[tuple[int, int], int] = {}
    vlist, sides, lines = [], [], []

    def vid(x, y):
        k = (x, y)
        if k not in verts:
            verts[k] = len(vlist)
            vlist.append(k)
        return verts[k]

    def attr(c_in, c_out):
        return special_edges.get((c_in, c_out))

    # unit edges --------------------------------------------------------------
    # vertical edge between cell (y, x) [west] and (y, x+1) [east] at X = (x+1)*CELL
    # horizontal edge between cell (y, x) [south] and (y+1, x) [north] at Y = (y+1)*CELL
    # Runs of edges are split wherever a perpendicular edge touches them, so no T-junctions remain.
    def vertical_has_h(X, Y):  # a horizontal edge touches grid point (X, Y)
        for x in (X - 1, X):
            if 0 <= x < W and 1 <= Y < H and grid[Y - 1, x] != grid[Y, x]:
                return True
        return False

    def horizontal_has_v(X, Y):
        for y in (Y - 1, Y):
            if 0 <= y < H and 1 <= X < W and grid[y, X - 1] != grid[y, X]:
                return True
        return False

    # vertical runs
    for x in range(W - 1):
        X = x + 1
        y = 0
        while y < H:
            a, b = grid[y, x], grid[y, x + 1]
            if a == b:
                y += 1
                continue
            key = (a, b, attr((y, x), (y, x + 1)), attr((y, x + 1), (y, x)))
            y1 = y + 1
            while y1 < H and (grid[y1, x], grid[y1, x + 1]) == (a, b) and \
                    (a, b, attr((y1, x), (y1, x + 1)), attr((y1, x + 1), (y1, x))) == key and not vertical_has_h(X, y1):
                y1 += 1
            _emit(lines, sides, sectors, vid, west=a, east=b, p_lo=(X, y), p_hi=(X, y1), vertical=True,
                  special=key[2] or key[3])
            y = y1
    # horizontal runs
    for y in range(H - 1):
        Y = y + 1
        x = 0
        while x < W:
            a, b = grid[y, x], grid[y + 1, x]
            if a == b:
                x += 1
                continue
            key = (a, b, attr((y, x), (y + 1, x)), attr((y + 1, x), (y, x)))
            x1 = x + 1
            while x1 < W and (grid[y, x1], grid[y + 1, x1]) == (a, b) and \
                    (a, b, attr((y, x1), (y + 1, x1)), attr((y + 1, x1), (y, x1))) == key and not horizontal_has_v(x1, Y):
                x1 += 1
            _emit(lines, sides, sectors, vid, west=a, east=b, p_lo=(x, Y), p_hi=(x1, Y), vertical=False,
                  special=key[2] or key[3])
            x = x1

    out = ['namespace = "zdoom";']
    for x, y in vlist:
        out.append(f"vertex {{ x = {x * CELL}.0; y = {y * CELL}.0; }}")
    for s in sectors:
        out.append(f'sector {{ texturefloor = "{s["ftex"]}"; textureceiling = "{s["ctex"]}"; heightfloor = {s["floor"]}; '
                   f'heightceiling = {s["ceil"]}; lightlevel = {s["light"]}; }}')
    for sd in sides:
        out.append(f'sidedef {{ sector = {sd[0]}; texturetop = "{sd[1]}"; texturemiddle = "{sd[2]}"; texturebottom = "{sd[3]}"; }}')
    for ln in lines:
        out.append("linedef { " + " ".join(f"{k} = {v};" for k, v in ln.items()) + " }")
    for (gy, gx), t, ang in things:
        out.append(f"thing {{ x = {gx * CELL + CELL // 2}.0; y = {gy * CELL + CELL // 2}.0; angle = {ang}; type = {t}; "
                   "skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; }")
    return "\n".join(out) + "\n"


def _emit(lines, sides, sectors, vid, *, west, east, p_lo, p_hi, vertical, special):
    """Emit one linedef between two cell labels.

    `west`/`east` are the cells on the low/high side (for horizontal edges: south/north).
    Doom puts the front sidedef on the RIGHT of v1 -> v2.
    """
    # orientation that puts `front` on the right
    def oriented(front_is_high):
        # vertical edge going up (lo -> hi) has east (high side) on its right
        # horizontal edge going east (lo -> hi) has south (low side) on its right
        if vertical:
            return (p_lo, p_hi) if front_is_high else (p_hi, p_lo)
        return (p_hi, p_lo) if front_is_high else (p_lo, p_hi)

    def side(sec, top="-", mid="-", bot="-"):
        sides.append((sec, top, mid, bot))
        return len(sides) - 1

    if west == SOLID or east == SOLID:
        sec = east if west == SOLID else west
        s = sectors[sec]
        mid = "SW1EXIT" if special == "exit_switch" else s["wall"]
        v1, v2 = oriented(front_is_high=(west == SOLID))
        ln = dict(v1=vid(*v1), v2=vid(*v2), sidefront=side(sec, mid=mid), blocking="true")
        if special == "exit_switch":
            ln.update(special=243, playeruse="true")
        lines.append(ln)
        return

    sw, se = sectors[west], sectors[east]
    door = None
    if sw["kind"] == "door":
        door, outer = west, east
    elif se["kind"] == "door":
        door, outer = east, west
    if door is not None and sectors[outer]["kind"] != "door":
        # front = outer side (where the player stands), back = door sector
        v1, v2 = oriented(front_is_high=(outer == east))
        lock = sectors[door].get("lock")
        top = KEYS[lock][3] if lock else "BIGDOOR2"
        ln = dict(v1=vid(*v1), v2=vid(*v2), sidefront=side(outer, top=top, bot=sectors[outer]["wall"]),
                  sideback=side(door), twosided="true", playeruse="true", repeatspecial="true")
        if lock:
            ln.update(special=13, arg0=0, arg1=16, arg2=150, arg3=KEYS[lock][0])
        else:
            ln.update(special=12, arg0=0, arg1=16, arg2=150, monsteruse="true")
        lines.append(ln)
        return

    # ordinary two-sided line (room <-> corridor, room <-> exit pad)
    v1, v2 = oriented(front_is_high=False)
    ln = dict(v1=vid(*v1), v2=vid(*v2), sidefront=side(west, top=sw["wall"], bot=sw["wall"]),
              sideback=side(east, top=se["wall"], bot=se["wall"]), twosided="true")
    if "exit" in (sw["kind"], se["kind"]):
        ln.update(special=243, playercross="true")
    lines.append(ln)


def write_level(level: Level, path: str | Path, map_name: str = "MAP01") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_wad(path, [Lump(map_name, b""), Lump("TEXTMAP", level.textmap.encode()), Lump("ENDMAP", b"")])
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", type=int, default=2, choices=sorted(TIERS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--count", type=int, default=1, help="generate this many consecutive seeds")
    ap.add_argument("--out", default="maps/generated/t{tier}_s{seed}.wad")
    ap.add_argument("--ascii", action="store_true", help="print the cell grid")
    a = ap.parse_args()
    for s in range(a.seed, a.seed + a.count):
        lv = generate(a.tier, s)
        p = write_level(lv, a.out.format(tier=a.tier, seed=s))
        print(f"{p}: {lv.meta} monsters={lv.num_monsters} locks={lv.lock_order} exit={'switch' if lv.meta['exit_switch'] else 'pad'}")
        if a.ascii:
            print(ascii_map(lv))


def ascii_map(lv: Level) -> str:
    ch = np.full(lv.grid.shape, "#", dtype="<U1")
    ch[lv.walkable] = "."
    for y, x in lv.exit_cells:
        ch[y, x] = "E"
    for colour, (y, x) in lv.key_cells:
        ch[y, x] = colour[0].upper()
    ch[lv.start] = "@"
    return "\n".join("".join(r) for r in ch[::-1])


if __name__ == "__main__":
    main()
