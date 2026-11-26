"""Evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6.

Every action comes from the trained DoomNet (brain/model.py). Per step the agent:
  1. integrates its own last action + observed velocity + ray scans into a dead-reckoned pose
     (physics replay corrected by the learned OdoNet),
  2. writes rays / entities into its allocentric EgoMap and reads two ego-centric crops,
  3. runs DoomNet (ray conv + entity attention + vitals + map CNN -> GRU) and samples an action.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"
sys.path.insert(0, str(SUBMISSION_DIR))

from brain.memory import OdoNet  # noqa: E402
from brain.model import DoomNet, Perception, comps_to_buttons, flatten_obs, sample, stack_obs  # noqa: E402

POLICY = WEIGHTS_DIR / "policy.pt"
ODONET = WEIGHTS_DIR / "odonet.pt"


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        self.observation_space = observation_space
        self.action_space = action_space
        # The network is small (~1.5M params, ~6 ms/step on one CPU core); running on the CPU avoids
        # host<->GPU copies every step and stays far below the 40 ms budget.
        self.device = torch.device("cpu")
        torch.set_num_threads(1)
        self.net = DoomNet().eval()
        odonet = None
        ck = torch.load(POLICY, map_location="cpu") if POLICY.exists() else None
        if ck is not None:
            self.net.load_state_dict(ck["model"])
            if ck.get("odonet") is not None:
                odonet = OdoNet()
                odonet.load_state_dict(ck["odonet"])
        elif ODONET.exists():
            odonet = OdoNet()
            odonet.load_state_dict(torch.load(ODONET, map_location="cpu"))
        else:
            print("[DoomAgent] warning: no trained weights found in", WEIGHTS_DIR, file=sys.stderr)
        if odonet is not None:
            odonet.eval()
        self.perception = Perception(1, self.device, odonet)
        self.greedy = os.environ.get("DOOM_AGENT_GREEDY", "0") == "1"
        self.gen = torch.Generator().manual_seed(0)
        self.reset()

    def reset(self) -> None:
        self.h = self.net.initial_state(1, self.device)
        self.prev_comps = torch.zeros(1, 6, dtype=torch.long)
        self.last_buttons = None
        self.perception.reset()
        self.perception.prev_rays = None

    @torch.no_grad()
    def act(self, observation: dict) -> np.ndarray:
        obs = stack_obs([flatten_obs(observation)], self.device)
        fresh = torch.tensor([self.last_buttons is None])
        maps = self.perception.observe(obs, self.last_buttons, fresh)
        logits, _, self.h = self.net.step(obs, maps, self.prev_comps, self.h)
        comps = sample(logits, generator=self.gen, greedy=self.greedy)
        buttons = comps_to_buttons(comps)
        self.prev_comps = comps
        self.last_buttons = buttons
        return buttons[0].numpy().astype(np.int8)
