"""Freedoom-first training map pool. Generated S0–S3 maps are a minority sprinkle."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from doom_env import make_env
from doom_env.mapinfo import load_mapinfo
from doom_env.wad import list_maps, read_wad
from doom_env.wrapper import IWAD

from training.mapgen.layout import build
from training.mapgen.tiers import STAGES

GENERATED_FRAC = 0.15
FREEDOOM_TICKS = 4200
TRAIN_MAPS = tuple(f"MAP{i:02d}" for i in range(1, 25))
VAL_MAPS = tuple(f"MAP{i:02d}" for i in range(25, 33))
GENERATED_STAGES = tuple(STAGES)
GENERATED_TRAIN_SEEDS = (0, 10_000)
GENERATED_VAL_SEEDS = (10_000, 1_000_000)
MANIFEST_PATH = Path(__file__).resolve().parent / "data" / "freedoom_manifest.json"
_MAP_NAME = re.compile(r"^MAP\d{2}$")


@dataclass(frozen=True)
class MapSpec:
    wad: str
    map: str
    source: str  # "freedoom" | "generated"
    episode_ticks: int
    stage: str | None = None
    seed: int | None = None


def freedoom_wad() -> str:
    return str(Path(IWAD).resolve())


def load_manifest(path: Path | None = None) -> dict:
    path = Path(path or MANIFEST_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"missing Freedoom inventory {path}; run python -m training.pool --inventory")
    return json.loads(path.read_text())


def _ok_maps(split: str, manifest: dict | None = None) -> list[str]:
    allowed = TRAIN_MAPS if split == "train" else VAL_MAPS
    data = manifest if manifest is not None else load_manifest()
    ok = {e["map"] for e in data.get("maps", []) if e.get("load_ok")}
    return [m for m in allowed if m in ok] or list(allowed)


def inventory_freedoom(*, refresh: bool = False, path: Path | None = None) -> dict:
    """Probe Freedoom MAP01–32. Writes training/data/freedoom_manifest.json."""
    path = Path(path or MANIFEST_PATH)
    if path.is_file() and not refresh:
        return load_manifest(path)

    wad = freedoom_wad()
    names = [n for n in list_maps(read_wad(wad)) if _MAP_NAME.match(n)]
    if not names:
        names = [f"MAP{i:02d}" for i in range(1, 33)]

    entries = []
    for name in names:
        sector_count = None
        error = None
        load_ok = False
        try:
            info = load_mapinfo(wad, name)
            sector_count = int(info.num_sectors)
        except Exception as exc:
            error = f"mapinfo: {exc}"
        try:
            env = make_env(wad, name, episode_ticks=100, render_buffers=False)
            try:
                env.reset(seed=0)
                load_ok = True
            finally:
                env.close()
        except Exception as exc:
            error = str(exc)
            load_ok = False
        if sector_count is not None and sector_count > 2048:
            print(f"warning: {name} has {sector_count} sectors (>2048); rays may miss geometry", file=sys.stderr)
        entries.append(
            {
                "map": name,
                "sector_count": sector_count,
                "load_ok": load_ok,
                "error": error,
            }
        )

    manifest = {"wad": "freedoom2.wad", "maps": entries}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def sample_spec(
    rng: np.random.Generator,
    generated_frac: float = GENERATED_FRAC,
    split: str = "train",
    manifest: dict | None = None,
) -> MapSpec:
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    generated_frac = float(generated_frac)
    if not 0.0 <= generated_frac <= 1.0:
        raise ValueError("generated_frac must be in [0, 1]")

    if rng.random() < generated_frac:
        stage = GENERATED_STAGES[int(rng.integers(0, len(GENERATED_STAGES)))]
        lo, hi = GENERATED_VAL_SEEDS if split == "val" else GENERATED_TRAIN_SEEDS
        seed = int(rng.integers(lo, hi))
        ticks = STAGES[stage].cap_ticks
        return MapSpec(
            wad="",
            map="MAP01",
            source="generated",
            episode_ticks=ticks,
            stage=stage,
            seed=seed,
        )

    maps = _ok_maps(split, manifest)
    name = maps[int(rng.integers(0, len(maps)))]
    return MapSpec(
        wad=freedoom_wad(),
        map=name,
        source="freedoom",
        episode_ticks=FREEDOOM_TICKS,
    )


def make_training_env(spec: MapSpec, *, tmp_dir: str | Path | None = None, **kwargs):
    """Materialize a generated WAD if needed, then call doom_env.make_env."""
    wad = spec.wad
    map_name = spec.map
    ticks = spec.episode_ticks
    if spec.source == "generated":
        if spec.stage is None or spec.seed is None:
            raise ValueError("generated MapSpec needs stage and seed")
        built = build(spec.stage, spec.seed)
        out_dir = Path(tmp_dir) if tmp_dir is not None else Path.cwd() / ".mapgen_tmp"
        out_dir.mkdir(parents=True, exist_ok=True)
        wad_path = out_dir / f"{spec.stage}_{spec.seed}.wad"
        built.map.save(wad_path)
        wad = str(wad_path)
        map_name = "MAP01"
        ticks = built.cap_ticks
    elif not wad:
        wad = freedoom_wad()
    kwargs.setdefault("episode_ticks", ticks)
    return make_env(wad, map_name, **kwargs)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Freedoom-first training map pool.")
    p.add_argument("--inventory", action="store_true", help="probe every Freedoom map and write the manifest")
    p.add_argument("--refresh", action="store_true", help="same as --inventory (rebuild the JSON)")
    p.add_argument("--sample", type=int, default=0, help="print N sampled specs and mix counts")
    p.add_argument("--generated-frac", type=float, default=GENERATED_FRAC)
    p.add_argument("--split", choices=("train", "val"), default="train")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.inventory or args.refresh:
        manifest = inventory_freedoom(refresh=True)
        ok = sum(1 for e in manifest["maps"] if e["load_ok"])
        print(f"inventory: {ok}/{len(manifest['maps'])} maps loaded -> {MANIFEST_PATH}", file=sys.stderr)
    else:
        manifest = load_manifest()

    if args.sample:
        rng = np.random.default_rng(args.seed)
        n_gen = n_fd = 0
        for _ in range(args.sample):
            spec = sample_spec(rng, generated_frac=args.generated_frac, split=args.split, manifest=manifest)
            if spec.source == "generated":
                n_gen += 1
                print(f"generated {spec.stage} seed={spec.seed} ticks={spec.episode_ticks}")
            else:
                n_fd += 1
                print(f"freedoom {spec.map} ticks={spec.episode_ticks}")
        print(f"mix: freedoom={n_fd} generated={n_gen} frac_generated={n_gen / args.sample:.3f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
