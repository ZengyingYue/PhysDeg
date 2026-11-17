"""One-batch forward/backward smoke test for every core-ablation variant."""
from __future__ import annotations

import copy

import torch
from ever.core.builder import make_model
from ever.core.config import import_config

from module.baseline.CoDER import CoDER  # noqa: F401


CONFIGS = {
    "core": "baseline.CoDER_whu_rebound_core",
    "fullrec": "baseline.CoDER_whu_rebound_fullrec",
    "no_recovery": "baseline.CoDER_whu_rebound_no_recovery",
    "fixed_rec": "baseline.CoDER_whu_rebound_fixed_rec",
    "no_obs_rel": "baseline.CoDER_whu_rebound_no_obs_rel",
    "no_rec_rel": "baseline.CoDER_whu_rebound_no_rec_rel",
    "no_dempster": "baseline.CoDER_whu_rebound_no_dempster",
    "flow": "baseline.CoDER_whu_rebound_flow",
    "flow_no_gate": "baseline.CoDER_whu_rebound_flow_no_gate",
}


def main():
    torch.manual_seed(7)
    for tag, name in CONFIGS.items():
        cfg = copy.deepcopy(import_config(name))
        params = cfg["model"]["params"]
        params["encoder"]["pretrained"] = False
        model = make_model(cfg["model"]).cuda()
        model.train()
        # Deterministically exercise RGB-missing recovery instead of relying on
        # the random modality sampler in a one-batch test.
        model._sample_missing = lambda: [1, 0]
        image = torch.randn(1, 4, 64, 64, device="cuda")
        target = {"cls": torch.randint(0, 7, (1, 64, 64), device="cuda")}
        losses = model(image, target)
        train_losses = [
            value for key, value in losses.items()
            if key != "mem" and torch.is_tensor(value) and value.requires_grad
        ]
        assert train_losses, f"{tag}: no differentiable losses"
        total = sum(train_losses)
        assert torch.isfinite(total), f"{tag}: non-finite loss"
        total.backward()

        beta_obs = float(model._compute_beta(False).detach())
        beta_rec = float(model._compute_beta(True).detach())
        if tag == "fullrec":
            assert abs(beta_rec - 1.0) < 1e-6, beta_rec
        if tag == "no_rec_rel":
            assert abs(beta_rec - beta_obs) < 1e-6, (beta_obs, beta_rec)

        model.eval()
        model.infer_mode = "sar"
        with torch.no_grad():
            probability = model(image, target)
        assert probability.shape == (1, 7, 64, 64), probability.shape
        assert torch.isfinite(probability).all()
        print(
            f"[OK] {tag:12s} loss={float(total.detach()):.4f} "
            f"beta_obs={beta_obs:.3f} beta_rec={beta_rec:.3f} "
            f"readout={model.missing_readout}",
            flush=True,
        )
        del model, image, target, losses, total, probability
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

