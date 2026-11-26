"""Batch-generate packed training WADs + manifest.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from doom_env.wad import write_wad

from .layout import BuiltMap, build
from .preview import render_preview
from .tiers import MAPGEN_VERSION, resolve_stage

MAPS_PER_WAD = 32


def _map_name(i: int) -> str:
    return f"MAP{i:02d}"


def _wad_filename(stage: str, first_seed: int) -> str:
    return f"{stage}_{first_seed:04d}.wad"


def _entry(built: BuiltMap, wad_name: str, map_name: str) -> dict:
    return {
        "wad": wad_name,
        "map": map_name,
        "stage": built.stage,
        "seed": built.seed,
        "rooms": len(built.rooms),
        "locks": 0,
        "sector_count": built.sector_count,
        "episode_ticks": built.cap_ticks,
        "start": list(built.start_xy),
        "exit": list(built.exit_xy),
        "solution_path_length": built.solution_path_length,
    }


def generate_batch(
    stage: str,
    n: int,
    seed: int,
    out: Path,
    preview: int = 8,
) -> dict:
    cfg = resolve_stage(stage)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    maps_meta: list[dict] = []
    wad_lumps: list = []
    wad_first = seed
    wad_index_in_file = 0

    def flush(first: int, lumps: list) -> None:
        if not lumps:
            return
        write_wad(out / _wad_filename(cfg.name, first), lumps)

    built_for_preview: list[BuiltMap] = []
    for i in range(n):
        s = seed + i
        built = build(cfg, s)
        if len(built_for_preview) < preview:
            built_for_preview.append(built)
        if wad_index_in_file == 0:
            wad_first = s
            wad_lumps = []
        name = _map_name(wad_index_in_file + 1)
        wad_name = _wad_filename(cfg.name, wad_first)
        wad_lumps.extend(built.map.lumps(name))
        maps_meta.append(_entry(built, wad_name, name))
        wad_index_in_file += 1
        if wad_index_in_file == MAPS_PER_WAD:
            flush(wad_first, wad_lumps)
            wad_lumps = []
            wad_index_in_file = 0
    flush(wad_first, wad_lumps)

    for built in built_for_preview:
        render_preview(built, out / f"{cfg.name}_{built.seed:04d}.png")

    manifest = {
        "version": MAPGEN_VERSION,
        "stage": cfg.name,
        "n": n,
        "seed": seed,
        "maps": maps_meta,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_manifest(out: Path, manifest: dict) -> None:
    from doom_env import make_env

    out = Path(out)
    for entry in manifest["maps"]:
        wad = out / entry["wad"]
        env = make_env(str(wad), entry["map"], episode_ticks=100, render_buffers=False)
        try:
            env.reset(seed=0)
        finally:
            env.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate packed UDMF training maps.")
    p.add_argument("--stage", default="S1", help="S0–S3 (or T1/T2 aliases)")
    p.add_argument("--n", type=int, default=32, help="number of maps")
    p.add_argument("--seed", type=int, default=0, help="first seed (maps use seed..seed+n-1)")
    p.add_argument("--out", type=Path, default=Path("maps/generated"))
    p.add_argument("--preview", type=int, default=8, help="how many top-down PNGs to write")
    p.add_argument("--verify", action="store_true", help="make_env+reset every packed map")
    args = p.parse_args(argv)

    manifest = generate_batch(args.stage, args.n, args.seed, args.out, preview=args.preview)
    print(f"wrote {manifest['n']} {manifest['stage']} maps to {args.out}", file=sys.stderr)
    if args.verify:
        verify_manifest(args.out, manifest)
        print("verify: all maps loaded", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
