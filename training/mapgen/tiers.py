"""Curriculum stage presets for the training map generator (S0–S3 in Phase 1)."""

from __future__ import annotations

from dataclasses import dataclass

MAPGEN_VERSION = "1"


@dataclass(frozen=True)
class StageConfig:
    name: str
    rooms: tuple[int, int]
    grid: tuple[int, int]  # (cols, rows)
    extra_loops: tuple[int, int]
    door_mode: str  # "open" | "plain"
    locks: int = 0
    enemies: int = 0
    cap_ticks: int = 2100
    exit_visible_from_start: bool = False


STAGES: dict[str, StageConfig] = {
    "S0": StageConfig(
        name="S0",
        rooms=(1, 1),
        grid=(1, 1),
        extra_loops=(0, 0),
        door_mode="open",
        cap_ticks=2100,
        exit_visible_from_start=True,
    ),
    "S1": StageConfig(
        name="S1",
        rooms=(2, 3),
        grid=(2, 2),
        extra_loops=(0, 0),
        door_mode="open",
        cap_ticks=2100,
    ),
    "S2": StageConfig(
        name="S2",
        rooms=(2, 4),
        grid=(2, 2),
        extra_loops=(0, 0),
        door_mode="plain",
        cap_ticks=2100,
    ),
    "S3": StageConfig(
        name="S3",
        rooms=(5, 9),
        grid=(3, 3),
        extra_loops=(1, 3),
        door_mode="plain",
        cap_ticks=3000,
    ),
}

# Eval-tier names alias the closest Phase 1 layout (no combat/gating yet).
ALIASES = {"T1": "S1", "T2": "S3"}


def resolve_stage(name: str) -> StageConfig:
    key = name.strip().upper()
    key = ALIASES.get(key, key)
    if key not in STAGES:
        known = ", ".join([*STAGES, *ALIASES])
        raise ValueError(f"unknown stage {name!r}; expected one of {known}")
    return STAGES[key]
