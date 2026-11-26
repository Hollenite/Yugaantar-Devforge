# Map Generation Plan — "Can It Play DOOM?"

Repo: `Hollenite/Yugaantar-Devforge` (from `Reinforce-SST/devforge-doom-template`)
Goal: a **procedural level generator** that produces unlimited, solvable, tiered training maps so the agent generalizes to the hidden evaluation suite. Training-only tooling: it lives under `training/`, never in `submission/`, and does not touch `env/` (the scorer ignores `env/` anyway).

---

## 1. Why this comes first

- Only one test map ships (`two_rooms.wad`). Without varied maps the agent overfits and scores ~0 on unseen layouts.
- Eval tiers (from `docs/EVALUATION.md`): T1 Combat, T2 Multi-Room Exploration, T3 Keycard → Locked Door → Exit, T4 Full Gauntlet. Tick caps are 2,100–4,200 (525–1,050 decisions). The generator should produce one family of maps per tier.
- Training-time tricks are explicitly allowed (auto-open doors, key-free maps, no enemies, curriculum). **A generator with difficulty knobs gives you the curriculum for free**, with no engine hacks. Final eval is vanilla, and generated maps are vanilla-compatible.

## 2. Facts already verified (spike, run against the template's `doom_env`)

A throwaway spike (`spike_mapgen.py`, attached) built a 4-sector map (room → door → room → exit alcove) and ran it through `make_env`. Confirmed:

| Fact | Result |
| --- | --- |
| Map format | **UDMF `TEXTMAP` only** (`MAP01`, `TEXTMAP`, `ENDMAP` lumps), `namespace = "zdoom"`. **No nodes/BSP needed**; the engine builds them. `doom_env.wad.write_wad` already exists. |
| Plain door | linedef `special = 12` (Door_Raise), `arg0=0` (tag 0 = the line's *back* sector), `arg1=16`, `arg2=150`, `playeruse = true`. Ray type reads `2` (door). Opens with a **single** `USE`. |
| Locked door | `special = 13` (Door_LockedRaise), same args + `arg3 = lock`. Ray type reads `3` (locked_door). With `lock=1` the **red keycard (thing type 13)** opens it. Without the key, `USE` does nothing. |
| Key pickup | Thing `13` shows in `visible_entities` as entity type `9` ("key"); `vitals[10]` (`key_red_card`) flips to 1 on pickup. |
| Exit | A small alcove sector whose entrance linedef has `special = 243` (Exit_Normal) + `playercross = true`. `privileged_state()["exit_reached"]` becomes True. |
| Door rule | A second `USE` while the door is moving **reverses** it (this bit my first run). Any scripted test must press `USE` once, then wait. |
| Things | `1` player start, `3004` zombieman (hitscan), `2011` stimpack (both used by the template), `13` red card (verified above). |
| Throughput | ~1,660 steps/s on 1 core with `render_buffers=False`. |

Not yet verified (the coding agent must check with a test that spawns each thing and reads the entity class): blue card `5`, yellow card `6`, lock numbers 2/3, other monster/pickup IDs (imp `3001`, demon `3002`, shotgun guy `9`, shotgun `2001`, clip `2007`, shells `2008`, medikit `2012`, armor `2018`).

## 3. Core data model

Sectors are **clockwise polygons** (y up). A linedef is created per unique edge:

- Edge owned by one sector → one-sided wall (`blocking = true`).
- Edge shared by two sectors (identical vertex pair) → two-sided linedef. If one side is a door sector, **front = the non-door sector, back = the door sector**, so tag-0 specials open the door.
- Rooms are rectangles. Wherever a connector attaches, insert the opening's two end points into the room's boundary polygon, so both sectors share the exact same edge. This avoids T-junctions.
- Door sector: floor 0, ceiling 0 (closed). Corridors and open connectors are ordinary sectors.

The spike's `Map` class is the working seed for this; port it, clean it up and extend it.

## 4. Generator design (`training/mapgen/`)

```
training/mapgen/
  wadbuild.py    # Map/Sector builder + UDMF writer (ported from spike)
  layout.py      # room graph + geometry
  gating.py      # key/lock placement + solvability check
  populate.py    # enemies, ammo, health, player start, exit
  tiers.py       # TierConfig presets and curriculum stages
  preview.py     # top-down PNG via pygame (already a dependency)
  generate.py    # CLI: batch → packed WADs + manifest.json
tests/test_mapgen.py
```

### 4.1 Layout
1. Grid of `R × C` cells (cell ≈ 512 units). Each cell may hold a room, a rectangle jittered inside it (sizes 192–448, snapped to 32).
2. Pick the room set, then build connectivity: random spanning tree (guarantees connected), plus `extra_loops` random extra edges between adjacent rooms (loops teach backtracking), and leave some leaf rooms as dead ends (T2/T3 "dead-end traps").
3. Each graph edge = a **connector sector** across the gap between two adjacent rooms: width 64, length = the gap (64–192). Connector kind ∈ {open, door, locked door}.
4. Start room and exit room: exit = far end of the graph (maximize graph distance from start). Exit alcove is a 64×64 sector off a wall of the exit room.
5. Stretch: pillars/obstacles (extra one-sided loops, no sector needed), small floor-height steps (≤ 24 units walkable), irregular rooms.

Sector count stays ≈ 2 × rooms, far below ViZDoom's 2,048-sector limit.

### 4.2 Gating (T3/T4)
Standard lock-and-key placement on the room graph:
1. Compute the start→exit path.
2. Choose a connector on that path to lock with color *k*.
3. Put key *k* in a room reachable from the start **without** crossing that lock (preferably a dead-end branch → forces exploration + remembering the locked door).
4. For T4 repeat with chained locks (key 2 only reachable after lock 1).
5. **Solvability validator:** BFS over `(room, keys_held)` states must reach the exit. This is training tooling, so graph search is fine here. Reject and regenerate if unsolvable.

### 4.3 Population
- Player start in the start room, facing the interior. Things must lie strictly inside sector polygons (point-in-polygon check, ≥ 32 units from walls) and not within 128 units of the start (except T1 optional).
- Enemies from a per-tier pool, placed by a budget. Mix hitscan (3004), melee (3002), projectile (3001) for T1/T4 so the encoder sees all enemy types.
- Ammo/health guarantees: a clip near the start; health/ammo pickups scaled by `resource_scarcity` (T4 scarce).
- T4 ambush (stretch): linedef-triggered monster closets. Do it last.

### 4.4 Tier presets (proposal; calibrate after the first agent runs)

| Tier | Rooms | Structure | Locks | Enemies | Tick cap |
| --- | --- | --- | --- | --- | --- |
| T1 Combat | 1–3 | arena, exit open | 0 | 3–8 mixed | 2100 |
| T2 Explore | 5–9 | loops + dead ends, plain doors | 0 | 0–4 | ~3000 |
| T3 Progression | 6–10 | 1 key → 1 locked door → exit | 1 | 2–6 | ~3500 |
| T4 Gauntlet | 9–14 | chained locks, scarce resources, ambush | 2–3 | 8–15 | 4200 |

Calibrate caps empirically: measure the player's real top speed, then require the optimal path to use ≤ ~40% of the cap so a decent agent has slack to explore.

### 4.5 Curriculum stages = just parameters
`TierConfig` flags: `rooms`, `loops`, `door_mode` (`open|plain|locked`), `locks`, `enemies`, `scarcity`, `exit_visible_from_start`, `cap_ticks`.

| Stage | Config | What the agent learns |
| --- | --- | --- |
| S0 | one room, exit visible | move toward and reach the exit |
| S1 | 2–3 rooms, open connectors | navigation, turning |
| S2 | plain doors | `USE` edge-trigger, wait for the door |
| S3 | 5–9 rooms, loops/dead ends | exploration, memory |
| S4 | + key and lock | long-horizon progression |
| S5 | + enemies | combat while navigating |
| S6 | T4 full | everything |

## 5. Dataset & training integration

- **Packed WADs:** CLI writes `MAP01..MAP32` per WAD to avoid thousands of files. Example: `python -m training.mapgen.generate --tier 3 --n 500 --seed 0 --out maps/generated/`.
- **`manifest.json`:** per map: tier, seed, rooms, locks, solution path length, sector count, recommended `episode_ticks`, key/door/exit coordinates (the agent never sees these; training-only for reward shaping and eval metrics).
- **Deterministic:** same `(tier, seed, version)` → byte-identical WAD (test via SHA-1).
- **Splits:** train seeds `0–9,999`, validation `10,000+`. Add a *shifted* hold-out set (bigger rooms, narrower connectors, different enemy mix) to estimate zero-shot generalization honestly.
- **Real-map pool:** ViZDoom bundles `freedoom2.wad`. The template's own tests load Freedoom MAP01, so mix a few as stress tests (watch the 2,048-sector warning; large levels may degrade rays).
- **Env loading cost:** `make_env` creates a `DoomGame` per call. Measure init time; if it is slow, keep a pool of environments per worker and re-use each across several maps/episodes.
- **Reward shaping hooks:** `privileged_state()` gives `x, y, kills, keys, exit_reached`, so a visited-cell grid (exploration), key/door/exit bonuses and potential-based progress shaping all work in training. They are unavailable at eval, so the policy must learn from the observation only.

## 6. Phases and acceptance criteria

| Phase | Work | Done when |
| --- | --- | --- |
| **0 (~30–45 min)** | Port the spike into `wadbuild.py`; add primitive tests: door opens with one `USE`, lock needs key, exit triggers | tests pass in CI/locally |
| **1 (~1.5 h)** | `layout.py` + `generate.py` + `preview.py` for T1/T2 (S0–S3) | 200 seeds load in `doom_env` with no errors; previews look sane; sectors < 200 each |
| **2 (~1.5 h)** | `gating.py` + T3 (keys, locks) + solver | 200 seeds per tier all graph-solvable; key never behind its own lock |
| **3 (~1–2 h)** | Enemies, resources, T4, calibrated caps, hold-out set | scripted/random baselines give expected tier ordering; manifest complete |
| **4** | Hook into the training env: map sampler, curriculum scheduler, val set | a training run samples fresh maps per episode |

**Minimum useful milestone = end of Phase 1.** Training can begin on S0–S3 while Phases 2–3 are built in parallel.

### Tests (`tests/test_mapgen.py`)
- Primitives (from spike): plain door, locked door + correct key, locked door without key stays shut, exit trigger.
- Per tier × 200 seeds: loads; `sector_count < 2048`; every thing inside a sector polygon; start ≠ exit room; graph solvable; no duplicate vertices at distinct IDs.
- Determinism: same seed → identical SHA-1.
- Sanity: a uniform-random agent's exit rate is low on T3/T4 and nonzero on S0 (checks the curriculum actually ramps).

## 7. Risks

| Risk | Mitigation |
| --- | --- |
| T-junction / mismatched shared edges → broken geometry | Insert opening vertices into room polygons; assert every two-sided edge has exactly two owners |
| Things spawn in walls or on door sectors | Point-in-polygon + wall clearance check; never place in door sectors |
| Door fiddling: agent spams `USE` and the door bounces | Real Doom behavior and the same at eval; don't fix it in the maps. The policy has to learn it |
| Other lock numbers / thing IDs wrong | Verify each in a test before using |
| Distribution gap between generated maps and hidden eval maps | The hidden maps are unseen. Mix generator styles, add the shifted hold-out set, include a few Freedoom maps; don't over-tune to my guesses in §4.4 |
| Time | Phase 1 alone unlocks training; everything past it is incremental |

## 8. Rules check
Generator, curriculum and reward shaping are allowed training-time scaffolding (`docs/RULES.md` §2). The **agent** must still choose actions from trained network parameters only. Keep every solver or BFS helper inside `training/` and out of `submission/`.
