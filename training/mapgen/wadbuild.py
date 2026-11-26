"""Polygon-sector UDMF builder. Sectors are clockwise polygons (y up); shared edges become two-sided lines."""

from __future__ import annotations

from pathlib import Path

from doom_env.wad import Lump, write_wad

DOOR_RAISE = 12
DOOR_LOCKED_RAISE = 13
EXIT_NORMAL = 243
THING_PLAYER = 1
THING_RED_CARD = 13
THING_ZOMBIEMAN = 3004


def _clean_poly(poly):
    """Drop consecutive duplicate vertices (openings flush with a corner)."""
    pts = [(int(x), int(y)) for x, y in poly]
    if not pts:
        raise ValueError("empty polygon")
    out = [pts[0]]
    for p in pts[1:]:
        if p != out[-1]:
            out.append(p)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    if len(out) < 3:
        raise ValueError(f"polygon degenerated to {out}")
    return out


class Map:
    def __init__(self):
        self.sectors = []  # dict(poly, kind, floor, ceil, tex)
        self.things = []

    def sector(self, poly, kind="room", ceil=128, **kw):
        self.sectors.append(dict(poly=_clean_poly(poly), kind=kind, ceil=ceil, **kw))
        return len(self.sectors) - 1

    def udmf(self) -> str:
        verts, vid = [], {}

        def V(p):
            if p not in vid:
                vid[p] = len(verts)
                verts.append(p)
            return vid[p]

        edges = {}  # frozenset(v1,v2) -> list of (sector, v1, v2)
        for si, s in enumerate(self.sectors):
            p = s["poly"]
            for i in range(len(p)):
                a, b = V(p[i]), V(p[(i + 1) % len(p)])
                edges.setdefault(frozenset((a, b)), []).append((si, a, b))
        sides, lines = [], []

        def side(sec, tex_mid="STARTAN2", top="-", bot="-"):
            sides.append(
                f'sidedef {{ sector = {sec}; texturemiddle = "{tex_mid}"; '
                f'texturetop = "{top}"; texturebottom = "{bot}"; }}'
            )
            return len(sides) - 1

        for key, owners in edges.items():
            if len(owners) == 1:
                si, a, b = owners[0]
                f = side(si)
                lines.append(f"linedef {{ v1 = {a}; v2 = {b}; sidefront = {f}; blocking = true; }}")
            elif len(owners) == 2:
                (s1, a1, b1), (s2, a2, b2) = owners
                k1, k2 = self.sectors[s1]["kind"], self.sectors[s2]["kind"]
                # front = non-door sector so the door is the BACK sector (tag 0 specials open the back sector)
                if k1 == "door":
                    (s1, a1, b1), (s2, a2, b2) = (s2, a2, b2), (s1, a1, b1)
                    k1, k2 = k2, k1
                door = self.sectors[s2] if k2 == "door" else None
                extra = ""
                top = "-"
                if door:
                    top = "BIGDOOR2"
                    if door.get("lock"):
                        extra = (
                            f"special = {DOOR_LOCKED_RAISE}; arg0 = 0; arg1 = 16; arg2 = 150; "
                            f"arg3 = {door['lock']}; playeruse = true; repeatspecial = true;"
                        )
                    else:
                        extra = (
                            f"special = {DOOR_RAISE}; arg0 = 0; arg1 = 16; arg2 = 150; "
                            f"playeruse = true; repeatspecial = true;"
                        )
                if self.sectors[s2]["kind"] == "exit" or self.sectors[s1]["kind"] == "exit":
                    extra = f"special = {EXIT_NORMAL}; playercross = true;"
                f = side(s1, "-", top, top)
                b = side(s2, "-", top, top)
                lines.append(
                    f"linedef {{ v1 = {a1}; v2 = {b1}; sidefront = {f}; sideback = {b}; twosided = true; {extra} }}"
                )
            else:
                raise ValueError(f"T-junction: edge {key} has {len(owners)} owners")
        out = ['namespace = "zdoom";']
        out += [f"vertex {{ x = {int(x)}.0; y = {int(y)}.0; }}" for x, y in verts]
        out += lines + sides
        for s in self.sectors:
            ceil = 0 if s["kind"] == "door" else s["ceil"]
            out.append(
                f'sector {{ texturefloor = "FLOOR4_8"; textureceiling = "CEIL1_1"; '
                f"heightfloor = 0; heightceiling = {ceil}; lightlevel = 192; }}"
            )
        for x, y, ang, typ in self.things:
            out.append(
                f"thing {{ x = {int(x)}.0; y = {int(y)}.0; angle = {int(ang)}; type = {int(typ)}; "
                f"skill1=true; skill2=true; skill3=true; skill4=true; skill5=true; single=true; }}"
            )
        return "\n".join(out) + "\n"

    def lumps(self, name: str = "MAP01") -> list[Lump]:
        return [Lump(name, b""), Lump("TEXTMAP", self.udmf().encode()), Lump("ENDMAP", b"")]

    def save(self, path: str | Path, name: str = "MAP01") -> None:
        write_wad(path, self.lumps(name))


def build_spike_map(lock: int = 1, with_key: bool = True) -> Map:
    """Room A → door → room B → exit alcove. Same geometry as the feasibility spike."""
    m = Map()
    # clockwise polygons (y up): TL -> TR -> BR -> BL
    # Room A: x 0..384, y 0..256 ; door connector x 384..448 y 96..160 ; Room B: x 448..832
    m.sector([(0, 256), (384, 256), (384, 160), (384, 96), (384, 0), (0, 0)])
    m.sector([(384, 160), (448, 160), (448, 96), (384, 96)], kind="door", lock=lock)
    m.sector(
        [
            (448, 256),
            (832, 256),
            (832, 160),
            (832, 96),
            (832, 0),
            (448, 0),
            (448, 96),
            (448, 160),
        ]
    )
    m.sector([(832, 160), (896, 160), (896, 96), (832, 96)], kind="exit")
    m.things.append((64, 128, 0, THING_PLAYER))  # facing east
    if with_key:
        m.things.append((160, 128, 0, THING_RED_CARD))
    m.things.append((700, 200, 180, THING_ZOMBIEMAN))
    return m
