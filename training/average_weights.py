"""Average the weights of several checkpoints from one training run (stochastic weight averaging).

Checkpoints taken a few hundred updates apart at a small learning rate lie in the same basin; their
mean sits nearer its centre and usually generalises better than any single one (Izmailov et al. 2018,
"Averaging Weights Leads to Wider Optima and Better Generalization"; Nikishin et al. 2018, "Improving
Stability in Deep Reinforcement Learning with Weight Averaging"). The result is still one trained
network; always confirm it with select_best.py on held-out maps before installing it.

    uv run python training/average_weights.py training/runs/v4/eval_02{6,7,8,9}00.pt training/runs/v4/eval_03000.pt \\
        --out training/runs/v4/avg_2600-3000.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def average(paths: list[str]) -> dict:
    sds = [torch.load(p, map_location="cpu", weights_only=False)["model"] for p in paths]
    out = {}
    for k, v in sds[0].items():
        if v.is_floating_point():
            out[k] = (sum(sd[k].float() for sd in sds) / len(sds)).to(v.dtype)
        else:  # integer buffers (none in DoomNet today) are taken from the first checkpoint
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--odonet", default=str(Path(__file__).resolve().parents[1] / "submission/weights/odonet.pt"),
                    help="odometry net to bundle so the file can be installed as submission/weights/policy.pt")
    a = ap.parse_args()
    ck = {"model": average(a.ckpts), "averaged_from": [str(p) for p in a.ckpts]}
    if a.odonet and Path(a.odonet).exists():
        ck["odonet"] = torch.load(a.odonet, map_location="cpu")
    torch.save(ck, a.out)
    print(f"averaged {len(a.ckpts)} checkpoints -> {a.out}")


if __name__ == "__main__":
    main()
