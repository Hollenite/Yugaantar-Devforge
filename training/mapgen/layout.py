"""Room-graph layout: grid blob → jittered rects → connectors → UDMF Map."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from .tiers import MAPGEN_VERSION, StageConfig, resolve_stage
from .wadbuild import THING_PLAYER, Map

CELL = 512
SNAP = 32
OPEN_W = 64
ALCOVE = 64
MIN_MARGIN = 32
MIN_GAP = 64
MAX_GAP = 192
MIN_ROOM = 192
MAX_ROOM = 384
CLEARANCE = 32
PLACE_TRIES = 40

# Doom thing angle: 0 east, 90 north, 180 west, 270 south.
ANGLE_EAST, ANGLE_NORTH, ANGLE_WEST, ANGLE_SOUTH = 0, 90, 180, 270
SIDES = ("E", "N", "S", "W")


@dataclass
class Room:
    col: int
    row: int
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass
class Opening:
    side: str
    lo: int
    hi: int


@dataclass
class Connector:
    a: int
    b: int
    side_a: str  # wall of room a that the opening sits on
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass
class BuiltMap:
    map: Map
    stage: str
    seed: int
    rooms: list[Room]
    start_room: int
    exit_room: int
    sector_count: int
    solution_path_length: int
    start_xy: tuple[int, int]
    exit_xy: tuple[int, int]
    cap_ticks: int
    openings: dict[int, list[Opening]] = field(default_factory=dict)


def make_rng(stage_name: str, seed: int) -> np.random.Generator:
    ver = int("".join(ch for ch in MAPGEN_VERSION if ch.isdigit()) or "1")
    stage_id = {"S0": 0, "S1": 1, "S2": 2, "S3": 3}.get(stage_name, 99)
    return np.random.default_rng([ver, stage_id, int(seed)])


def snap(v: int) -> int:
    return int(round(v / SNAP) * SNAP)


def point_in_poly(x: float, y: float, poly: list[tuple[int, int]]) -> bool:
    n = len(poly)
    inside = False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and y1 != y2:
            xin = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xin:
                inside = not inside
    return inside


def _choice(rng: np.random.Generator, seq):
    if not seq:
        raise ValueError("empty choice")
    return seq[int(rng.integers(0, len(seq)))]


def _grow_blob(rng: np.random.Generator, n: int, cols: int, rows: int) -> list[tuple[int, int]]:
    cells = [(c, r) for r in range(rows) for c in range(cols)]
    start = _choice(rng, cells)
    occupied = [start]
    occ = {start}

    def empty_neighbors(cell):
        c, r = cell
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nc, nr = c + dc, r + dr
            if 0 <= nc < cols and 0 <= nr < rows and (nc, nr) not in occ:
                yield (nc, nr)

    while len(occupied) < n:
        frontier = []
        seen = set()
        for cell in occupied:
            for nb in empty_neighbors(cell):
                if nb not in seen:
                    seen.add(nb)
                    frontier.append(nb)
        if not frontier:
            break
        pick = _choice(rng, frontier)
        occupied.append(pick)
        occ.add(pick)
    return occupied


def _place_room(rng: np.random.Generator, col: int, row: int) -> Room:
    cx0, cy0 = col * CELL, row * CELL
    sizes = list(range(MIN_ROOM, MAX_ROOM + 1, SNAP))
    w = int(_choice(rng, sizes))
    h = int(_choice(rng, sizes))
    slack_x = CELL - w
    slack_y = CELL - h

    def pick_margin(slack: int) -> int:
        lo = MIN_MARGIN
        hi = slack - MIN_MARGIN
        choices = list(range(lo, hi + 1, SNAP))
        if not choices:
            return snap(slack // 2)
        return int(_choice(rng, choices))

    left = pick_margin(slack_x)
    bottom = pick_margin(slack_y)
    x0 = cx0 + left
    y0 = cy0 + bottom
    return Room(col=col, row=row, x0=x0, y0=y0, x1=x0 + w, y1=y0 + h)


def _centered_room(col: int, row: int) -> Room:
    w = h = 320
    x0 = col * CELL + (CELL - w) // 2
    y0 = row * CELL + (CELL - h) // 2
    return Room(col=col, row=row, x0=x0, y0=y0, x1=x0 + w, y1=y0 + h)


def _candidate_connectors(rooms: list[Room]) -> list[Connector]:
    by_cell = {(rm.col, rm.row): i for i, rm in enumerate(rooms)}
    out: list[Connector] = []
    for i, a in enumerate(rooms):
        east = by_cell.get((a.col + 1, a.row))
        if east is not None:
            c = _horizontal_connector(i, east, a, rooms[east])
            if c is not None:
                out.append(c)
        north = by_cell.get((a.col, a.row + 1))
        if north is not None:
            c = _vertical_connector(i, north, a, rooms[north])
            if c is not None:
                out.append(c)
    return out


def _horizontal_connector(ia: int, ib: int, west: Room, east: Room) -> Connector | None:
    gap = east.x0 - west.x1
    if gap < MIN_GAP or gap > MAX_GAP:
        return None
    y0 = max(west.y0, east.y0)
    y1 = min(west.y1, east.y1)
    if y1 - y0 < OPEN_W:
        return None
    return Connector(a=ia, b=ib, side_a="E", x0=west.x1, y0=y0, x1=east.x0, y1=y1)


def _vertical_connector(ia: int, ib: int, south: Room, north: Room) -> Connector | None:
    gap = north.y0 - south.y1
    if gap < MIN_GAP or gap > MAX_GAP:
        return None
    x0 = max(south.x0, north.x0)
    x1 = min(south.x1, north.x1)
    if x1 - x0 < OPEN_W:
        return None
    return Connector(a=ia, b=ib, side_a="N", x0=x0, y0=south.y1, x1=x1, y1=north.y0)


def _connected(n: int, edges: list[Connector]) -> bool:
    if n <= 1:
        return True
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges:
        parent[find(e.a)] = find(e.b)
    return len({find(i) for i in range(n)}) == 1


def _kruskal(n: int, rng: np.random.Generator, cands: list[Connector], extra_loops: int) -> list[Connector]:
    order = [int(i) for i in rng.permutation(len(cands))]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree: list[Connector] = []
    leftover: list[Connector] = []
    for idx in order:
        e = cands[idx]
        ra, rb = find(e.a), find(e.b)
        if ra != rb:
            parent[ra] = rb
            tree.append(e)
        else:
            leftover.append(e)
    extra_loops = min(int(extra_loops), len(leftover))
    if extra_loops:
        pick = [int(i) for i in rng.permutation(len(leftover))[:extra_loops]]
        tree.extend(leftover[i] for i in pick)
    return tree


def _pick_opening_range(rng: np.random.Generator, lo: int, hi: int) -> tuple[int, int]:
    """Pick a SNAP-aligned OPEN_W window inside [lo, hi)."""
    choices = list(range(snap(lo) if snap(lo) >= lo else snap(lo) + SNAP, hi - OPEN_W + 1, SNAP))
    if not choices:
        start = lo
        if start + OPEN_W > hi:
            start = hi - OPEN_W
        return int(start), int(start + OPEN_W)
    start = int(_choice(rng, choices))
    return start, start + OPEN_W


def _materialize_connector(rng: np.random.Generator, e: Connector) -> Connector:
    if e.side_a == "E":
        y_lo, y_hi = _pick_opening_range(rng, e.y0, e.y1)
        return Connector(a=e.a, b=e.b, side_a="E", x0=e.x0, y0=y_lo, x1=e.x1, y1=y_hi)
    x_lo, x_hi = _pick_opening_range(rng, e.x0, e.x1)
    return Connector(a=e.a, b=e.b, side_a="N", x0=x_lo, y0=e.y0, x1=x_hi, y1=e.y1)


def _openings_from_connectors(n: int, conns: list[Connector]) -> dict[int, list[Opening]]:
    out: dict[int, list[Opening]] = defaultdict(list)
    for e in conns:
        if e.side_a == "E":
            out[e.a].append(Opening("E", e.y0, e.y1))
            out[e.b].append(Opening("W", e.y0, e.y1))
        else:
            out[e.a].append(Opening("N", e.x0, e.x1))
            out[e.b].append(Opening("S", e.x0, e.x1))
    for i in range(n):
        out.setdefault(i, [])
    return out


def _rect_poly(room: Room, openings: list[Opening]) -> list[tuple[int, int]]:
    x0, y0, x1, y1 = room.x0, room.y0, room.x1, room.y1
    by_side: dict[str, list[Opening]] = defaultdict(list)
    for o in openings:
        by_side[o.side].append(o)
    verts: list[tuple[int, int]] = []
    verts.append((x0, y1))
    for o in sorted(by_side["N"], key=lambda o: o.lo):
        verts.append((o.lo, y1))
        verts.append((o.hi, y1))
    verts.append((x1, y1))
    for o in sorted(by_side["E"], key=lambda o: -o.hi):
        verts.append((x1, o.hi))
        verts.append((x1, o.lo))
    verts.append((x1, y0))
    for o in sorted(by_side["S"], key=lambda o: -o.hi):
        verts.append((o.hi, y0))
        verts.append((o.lo, y0))
    verts.append((x0, y0))
    for o in sorted(by_side["W"], key=lambda o: o.lo):
        verts.append((x0, o.lo))
        verts.append((x0, o.hi))
    return verts


def _connector_poly(e: Connector) -> list[tuple[int, int]]:
    return [(e.x0, e.y1), (e.x1, e.y1), (e.x1, e.y0), (e.x0, e.y0)]


def _free_segments(room: Room, openings: list[Opening], side: str) -> list[tuple[int, int]]:
    if side in ("E", "W"):
        lo, hi = room.y0, room.y1
    else:
        lo, hi = room.x0, room.x1
    taken = sorted((o.lo, o.hi) for o in openings if o.side == side)
    segs: list[tuple[int, int]] = []
    cursor = lo
    for a, b in taken:
        if a > cursor:
            segs.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < hi:
        segs.append((cursor, hi))
    return [(a, b) for a, b in segs if b - a >= ALCOVE]


def _place_in_segment(lo: int, hi: int, width: int = ALCOVE) -> tuple[int, int]:
    mid = (lo + hi) // 2
    start = snap(mid - width // 2)
    if start < lo:
        start = lo if lo % SNAP == 0 else snap(lo) + SNAP
        if start < lo:
            start = lo
    if start + width > hi:
        start = hi - width
        if start < lo:
            start = lo
    return int(start), int(start + width)


def _alcove_poly(room: Room, side: str, lo: int, hi: int) -> list[tuple[int, int]]:
    if side == "E":
        return [(room.x1, hi), (room.x1 + ALCOVE, hi), (room.x1 + ALCOVE, lo), (room.x1, lo)]
    if side == "W":
        return [(room.x0 - ALCOVE, hi), (room.x0, hi), (room.x0, lo), (room.x0 - ALCOVE, lo)]
    if side == "N":
        return [(lo, room.y1 + ALCOVE), (hi, room.y1 + ALCOVE), (hi, room.y1), (lo, room.y1)]
    return [(lo, room.y0), (hi, room.y0), (hi, room.y0 - ALCOVE), (lo, room.y0 - ALCOVE)]


def _alcove_center(room: Room, side: str, lo: int, hi: int) -> tuple[int, int]:
    mid = (lo + hi) // 2
    if side == "E":
        return room.x1 + ALCOVE // 2, mid
    if side == "W":
        return room.x0 - ALCOVE // 2, mid
    if side == "N":
        return mid, room.y1 + ALCOVE // 2
    return mid, room.y0 - ALCOVE // 2


def _pick_alcove(
    room: Room, openings: list[Opening], prefer_east: bool
) -> tuple[str, int, int] | None:
    order = ("E", "N", "S", "W") if prefer_east else SIDES
    for side in order:
        segs = _free_segments(room, openings, side)
        if not segs:
            continue
        # Prefer the longest remaining stretch (stable: first in wall order).
        segs = sorted(segs, key=lambda s: (-(s[1] - s[0]), s[0]))
        lo, hi = _place_in_segment(*segs[0])
        if hi - lo >= ALCOVE:
            return side, lo, hi
    return None


def _bfs(n: int, conns: list[Connector], src: int) -> tuple[dict[int, int], dict[int, int]]:
    adj: dict[int, list[int]] = defaultdict(list)
    for e in conns:
        adj[e.a].append(e.b)
        adj[e.b].append(e.a)
    dist = {src: 0}
    parent = {src: src}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
    if len(dist) != n:
        missing = [i for i in range(n) if i not in dist]
        raise RuntimeError(f"disconnected rooms {missing}")
    return dist, parent


def _facing_angle(start: Room, nxt: Room) -> int:
    dx = (nxt.x0 + nxt.x1) - (start.x0 + start.x1)
    dy = (nxt.y0 + nxt.y1) - (start.y0 + start.y1)
    if abs(dx) >= abs(dy):
        return ANGLE_EAST if dx >= 0 else ANGLE_WEST
    return ANGLE_NORTH if dy >= 0 else ANGLE_SOUTH


def _player_xy(room: Room) -> tuple[int, int]:
    x = (room.x0 + room.x1) // 2
    y = (room.y0 + room.y1) // 2
    x = min(max(x, room.x0 + CLEARANCE), room.x1 - CLEARANCE)
    y = min(max(y, room.y0 + CLEARANCE), room.y1 - CLEARANCE)
    return int(x), int(y)


def build(stage: str | StageConfig = "S0", seed: int = 0) -> BuiltMap:
    cfg = stage if isinstance(stage, StageConfig) else resolve_stage(stage)
    rng = make_rng(cfg.name, seed)
    cols, rows = cfg.grid
    n = int(rng.integers(cfg.rooms[0], cfg.rooms[1] + 1))
    n = min(n, cols * rows)
    extra = int(rng.integers(cfg.extra_loops[0], cfg.extra_loops[1] + 1))
    occupied = _grow_blob(rng, n, cols, rows)
    n = len(occupied)

    rooms: list[Room] | None = None
    cands: list[Connector] = []
    for _ in range(PLACE_TRIES):
        rooms = [_place_room(rng, c, r) for c, r in occupied]
        cands = _candidate_connectors(rooms)
        if _connected(n, cands):
            break
    else:
        rooms = [_centered_room(c, r) for c, r in occupied]
        cands = _candidate_connectors(rooms)
        if not _connected(n, cands):
            raise RuntimeError(f"layout failed to connect {n} rooms (seed={seed} stage={cfg.name})")

    conns = [_materialize_connector(rng, e) for e in _kruskal(n, rng, cands, extra)]
    openings = _openings_from_connectors(n, conns)

    start_i = 0
    dist, parent = _bfs(n, conns, start_i)
    # Farthest; tie-break by cell for stability.
    exit_i = max(range(n), key=lambda i: (dist[i], rooms[i].row, rooms[i].col))
    alcove = _pick_alcove(rooms[exit_i], openings[exit_i], prefer_east=cfg.exit_visible_from_start or n == 1)
    if alcove is None:
        # Last resort: regenerate with centered rooms (more wall).
        rooms = [_centered_room(c, r) for c, r in occupied]
        cands = _candidate_connectors(rooms)
        conns = [_materialize_connector(rng, e) for e in _kruskal(n, rng, cands, extra)]
        openings = _openings_from_connectors(n, conns)
        dist, parent = _bfs(n, conns, start_i)
        exit_i = max(range(n), key=lambda i: (dist[i], rooms[i].row, rooms[i].col))
        alcove = _pick_alcove(rooms[exit_i], openings[exit_i], prefer_east=True)
        if alcove is None:
            raise RuntimeError(f"no wall for exit alcove (seed={seed} stage={cfg.name})")
    side, alo, ahi = alcove
    openings[exit_i].append(Opening(side, alo, ahi))

    m = Map()
    for i, rm in enumerate(rooms):
        m.sector(_rect_poly(rm, openings[i]), kind="room")
    conn_kind = "door" if cfg.door_mode == "plain" else "room"
    for e in conns:
        if conn_kind == "door":
            m.sector(_connector_poly(e), kind="door", lock=0)
        else:
            m.sector(_connector_poly(e), kind="room")
    m.sector(_alcove_poly(rooms[exit_i], side, alo, ahi), kind="exit")

    px, py = _player_xy(rooms[start_i])
    if n == 1 or cfg.exit_visible_from_start:
        ang = ANGLE_EAST if side == "E" else {"N": ANGLE_NORTH, "S": ANGLE_SOUTH, "W": ANGLE_WEST}[side]
    else:
        hop = exit_i
        while parent[hop] != start_i and hop != start_i:
            hop = parent[hop]
        ang = _facing_angle(rooms[start_i], rooms[hop if hop != start_i else exit_i])
    m.things.append((px, py, ang, THING_PLAYER))

    ex, ey = _alcove_center(rooms[exit_i], side, alo, ahi)
    return BuiltMap(
        map=m,
        stage=cfg.name,
        seed=int(seed),
        rooms=rooms,
        start_room=start_i,
        exit_room=exit_i,
        sector_count=len(m.sectors),
        solution_path_length=int(dist[exit_i]),
        start_xy=(px, py),
        exit_xy=(ex, ey),
        cap_ticks=cfg.cap_ticks,
        openings=dict(openings),
    )
