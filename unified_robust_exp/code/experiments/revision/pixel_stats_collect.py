"""Raw-pixel-statistics detector: a model-free control for the fallback gate.

For each sensor and image: per-channel mean, standard deviation and gradient
energy (mean |Laplacian|) in raw sensor units.  The deviation score is the
largest absolute z-score of these statistics against clean TRAINING images.
Corruptions use exactly the batch order, batch size and per-batch seeds of
gate_collect.py, so image i here is image i there.

    python experiments/revision/pixel_stats_collect.py
Writes revision_tgrs/gate_analysis/pixel_stats/{val,test}/<condition>.npz with
keys dev_o, dev_s (one score per image; larger = more anomalous).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CODE_ROOT = Path(__file__).resolve().parents[2]
REPO = CODE_ROOT.parents[1]
sys.path.insert(0, str(CODE_ROOT))

from ever.core.config import import_config  # noqa: E402
from experiments.revision.evaluate import Corruptor, loader  # noqa: E402
from experiments.revision.gate_collect import conditions  # noqa: E402

LAPLACE = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)


def stats(raw: torch.Tensor) -> torch.Tensor:
    """[B,C,H,W] raw -> [B, 3C] (mean, std, gradient energy per channel)."""
    b, c = raw.shape[:2]
    lap = F.conv2d(raw.reshape(b * c, 1, *raw.shape[2:]), LAPLACE.to(raw.device), padding=1)
    grad = lap.abs().reshape(b, c, -1).mean(-1)
    flat = raw.reshape(b, c, -1)
    return torch.cat([flat.mean(-1), flat.std(-1), grad], dim=1)


@torch.inference_mode()
def image_stats(dl, corrupt, kind, severity):
    out = {"o": [], "s": []}
    mean, std = corrupt.mean, corrupt.std
    for batch_index, (x, _) in enumerate(dl):
        generator = torch.Generator().manual_seed(10_000 + batch_index)
        x = corrupt(x.cuda(), kind, severity, generator)
        raw = x * std.to(x.device) + mean.to(x.device)
        out["s"].append(stats(raw[:, :1]).cpu())
        out["o"].append(stats(raw[:, 1:4]).cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def main():
    config = import_config("revision.whu_moddrop")  # WHU data block (identical for all methods)
    corrupt = Corruptor(config)
    root = REPO / "revision_tgrs" / "gate_analysis" / "pixel_stats"
    ref = image_stats(loader(config, "mmr", 4, 1024), corrupt, "none", 0)
    mu = {k: v.mean(0) for k, v in ref.items()}
    sd = {k: v.std(0).clamp_min(1e-6) for k, v in ref.items()}
    for split, name in (("val", "test"), ("test", "heldout_test")):  # data['test'] is validation
        dl = loader(config, name, 4)
        (root / split).mkdir(parents=True, exist_ok=True)
        for cond, kind, severity in conditions():
            s = image_stats(dl, corrupt, kind, severity)
            dev = {k: ((s[k] - mu[k]) / sd[k]).abs().max(1).values.numpy() for k in s}
            np.savez_compressed(root / split / f"{cond}.npz", dev_o=dev["o"], dev_s=dev["s"])
            print(f"[{split}] {cond}", flush=True)


if __name__ == "__main__":
    main()
