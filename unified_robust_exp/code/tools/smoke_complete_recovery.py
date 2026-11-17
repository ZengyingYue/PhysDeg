"""Deterministic forward/backward smoke test for complete-recovery configs."""
from __future__ import annotations

import argparse

import torch
from ever.core.builder import make_model
from ever.core.config import import_config

from module.baseline.CoDER import CoDER  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--missing", choices=("rgb", "sar"), required=True
    )
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    cfg = import_config(args.config)
    model = make_model(cfg["model"]).cuda().train()
    missing = [1, 0] if args.missing == "rgb" else [0, 1]
    model._sample_missing = lambda: missing

    size = args.size
    batch_size = args.batch_size
    x = torch.randn(batch_size, 4, size, size, device="cuda")
    target = {
        "cls": torch.randint(
            0,
            int(cfg["model"]["params"]["num_classes"]),
            (batch_size, size, size),
            device="cuda",
        )
    }
    with torch.autocast("cuda", dtype=torch.bfloat16):
        losses = model(x, target)
        total = sum(value for key, value in losses.items() if key != "mem")
    total.backward()

    direction = (
        "complete_sar_to_rgb"
        if args.missing == "rgb"
        else "complete_rgb_to_sar"
    )
    grad_params = sum(
        1
        for name, parameter in model.named_parameters()
        if name.startswith(direction) and parameter.grad is not None
    )
    if grad_params == 0:
        raise RuntimeError(f"no gradients reached {direction}")

    model.zero_grad(set_to_none=True)
    model.eval()
    model.infer_mode = "sar" if args.missing == "rgb" else "rgb"
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        probability = model(x)
    expected = (
        batch_size,
        int(cfg["model"]["params"]["num_classes"]),
        size,
        size,
    )
    if tuple(probability.shape) != expected:
        raise RuntimeError(
            f"unexpected output shape {tuple(probability.shape)} != {expected}"
        )
    if not torch.isfinite(probability).all():
        raise RuntimeError("non-finite complete-recovery prediction")
    normalisation_error = float(
        (probability.sum(1) - 1.0).abs().max()
    )
    print(
        f"PASS config={args.config} missing={args.missing} "
        f"batch={batch_size} size={size} "
        f"loss={float(total.detach()):.6f} "
        f"recon={float(losses['recon_loss'].detach()):.6f} "
        f"grad_params={grad_params} prob_norm_err={normalisation_error:.3e}"
    )


if __name__ == "__main__":
    main()
