# Training pipeline

Everything here is training-time code. The submitted agent (`submission/agent.py` + `submission/brain/`)
only uses the official observations and its trained weights.

```
levelgen.py        procedural UDMF levels for 5 curriculum tiers (rooms, corridors, doors, keycards,
                   locked doors, exit pads/switches, monsters, pickups) + the cell grid for shaping
denv.py            TrainEnv: level sampling, reward shaping (exit / progression / exploration / combat,
                   geodesic potential), and the privileged Teacher used for distillation
train_odometry.py  supervised training of OdoNet (learned odometry correction for the EgoMap)
ppo.py             recurrent PPO + DAgger-style teacher distillation + map-coverage curiosity + curriculum
evaluate.py        scorer-like metrics on held-out generated maps / any WAD, for agent, teacher or random
watch_random.py    template viewer (unchanged)
```

## Architecture in one paragraph

The agent keeps an **odometry-anchored spatial memory** (EgoMap). It replays Doom's movement physics
from its own previous action and the observed velocity, corrects the result with **OdoNet** (a small
circular-conv net over two consecutive 360° ray scans, ~0.3 map units error per step vs 4.4 for physics alone,
~0.5% drift over a whole episode), and writes rays and entities into an allocentric grid (free space, wall,
door, locked door, item, enemy, key, own trail). Each step it reads two heading-aligned crops
(±640 and ±2560 units). **DoomNet** encodes rays (circular Conv1d), visible entities (masked attention
pooling, so any number of enemies), vitals, and the map crops (CNN), fuses them with the previous action,
and runs a **GRU** core into factorised action heads (move × strafe × turn × attack × use × weapon).
The policy (1.55M params) runs in ~7 ms per step on one CPU core.

Training combines three signals:

1. **Privileged teacher distillation.** A teacher with ground-truth position, keys and the level graph
   follows geodesic paths to keys and exits, opens doors and aims at enemies. Its actions supervise the
   student (cross-entropy). Early on, the teacher's action is executed with probability `beta` (DAgger),
   so the student learns to recover from its own mistakes.
2. **PPO** on the shaped reward (mirrors the scoring: exit + health/time bonus, keys, doors, kills,
   unique areas, damage/death penalties, geodesic potential shaping) plus **map-coverage curiosity**:
   the number of cells the agent's own EgoMap revealed this step.
3. **Curriculum** over the five map tiers driven by per-tier exit rates.

## Setup (Apple Silicon Mac)

```bash
uv sync
uv pip install torch==2.15.0 pillow          # torch is in the scorer image; locally we add it
```

## Recipe

```bash
# 1. odometry (≈5 min): collects ~120k samples with ground truth, trains OdoNet
uv run python training/train_odometry.py --samples 120000 --workers 8

# 2. main run (hours; resume any time with --resume training/runs/main/latest.pt)
uv run python training/ppo.py --run main --workers 8 --envs-per-worker 4 --updates 3000

# 3. check it, then run the official validator
uv run python training/evaluate.py --episodes 5
uv run python validate.py
```

`ppo.py` writes `training/runs/<run>/log.csv`, `latest.pt` (resumable) and exports the inference
weights to `submission/weights/policy.pt` (policy + OdoNet) every `--save-every` updates.
Use `--device mps` to put the network on the Apple GPU (the EgoMap stays on the CPU by default).

Useful knobs: `--beta-updates` (how long the teacher drives), `--il-updates/--il-end` (imitation
weight schedule), `--curiosity`, `--tiers` (initial curriculum weights), `--wads file.wad --wad-prob 0.2`
(mix in real WADs such as community maps for extra variety; those episodes use event rewards only).

Weights are tracked with Git LFS (`.gitattributes`): run `git lfs install` once before committing them.

## Baselines on held-out maps (5 maps per tier)

Run `training/evaluate.py --policy teacher` and `--policy random` to refresh these numbers.
