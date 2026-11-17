"""Collect per-image statistics for fallback-criterion analysis (CoDER models).

For every image and condition this stores
  * the confusion matrix of each read-out route (fused / SAR head / optical head),
  * label-free scores a deployed gate may use: beta_o, beta_s, mean max-probability
    and mean entropy of each head, and argmax disagreement between heads.
Gates are then evaluated offline (gate_analysis.py) by summing the selected
route's confusion matrices, so no criterion needs another forward pass.

    python experiments/revision/gate_collect.py --config revision.whu_rebound \
        --run-dir ../runs/revision_v1/whu_rebound/seed_2333 --seed 2333
Uses Final.pth (reliability reference fitted by evaluate.py).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
REPO = CODE_ROOT.parents[1]
sys.path.insert(0, str(CODE_ROOT))

from ever.core.config import import_config  # noqa: E402
from experiments.revision.evaluate import OPTICAL, SAR, Corruptor, load_model, loader  # noqa: E402
from module.baseline.CoDER import CoDER  # noqa: E402
from module.reliability.covariance_deviation import CovarianceDeviationReliability  # noqa: E402


# --------------------------------------------------------------------------- #
# plug-in gate for models without a built-in reliability score
# --------------------------------------------------------------------------- #
def plain_features(model, x):
    """First two pyramid levels of each sensor's own encoder (256/512 ch)."""
    sar, rgb = model._split(x)
    if hasattr(model, "en_sar"):          # ModDrop / PAD
        return {"sar": model.en_sar(sar)[:2], "rgb": model.en_rgb(rgb)[:2]}
    return {"sar": model.spec_sar(sar)[:2], "rgb": model.spec_rgb(rgb)[:2]}  # ShaSpec


class PluginGate:
    """Covariance-deviation reliability fitted on clean training images of a
    baseline model; same statistic and hyper-parameters as ReBound's."""

    def __init__(self, model, config, train_limit=1024):
        self.model = model
        self.ref = CovarianceDeviationReliability([256, 512], diag_weight=0.25).cuda()
        dl = loader(config, "mmr", 4, train_limit)
        with torch.inference_mode():
            for x, _ in dl:
                f = plain_features(model, x.cuda())
                for m in ("sar", "rgb"):
                    self.ref.update(f[m], m)
            self.ref.finalize_statistics()
            scores = {m: ([], []) for m in ("sar", "rgb")}
            for x, _ in dl:
                f = plain_features(model, x.cuda())
                for m in ("sar", "rgb"):
                    scores[m][0].append(torch.stack(self.ref.deviation_scores(f[m], m), 1).cpu())
                    scores[m][1].append(torch.stack(self.ref.diagonal_scores(f[m], m), 1).cpu())
            for m in ("sar", "rgb"):
                self.ref.set_score_calibration(list(torch.cat(scores[m][0]).T), m)
                self.ref.set_diag_calibration(list(torch.cat(scores[m][1]).T), m)

    @torch.inference_mode()
    def __call__(self, x, target):
        model = self.model
        probs = {}
        for route, mode in (("fused", "fuse"), ("sar", "sar"), ("opt", "rgb")):
            model.infer_mode = mode
            probs[route] = model(x, target)
        f = plain_features(model, x)
        betas = {m: self.ref(f[m], m).reshape(-1) for m in ("sar", "rgb")}
        return probs, betas


def conditions():
    out = [("clean", "none", 0)]
    for kind in OPTICAL:
        out += [(f"optical_{kind}@{s}", kind, s) for s in range(1, 6)]
    for kind in SAR:
        out += [(f"sar_{kind}@{s}", kind, s) for s in range(1, 6)]
    return out


def per_image_confusion(pred, y, k):
    valid = y != -1
    enc = torch.where(valid, y * k + pred, torch.full_like(y, k * k))
    counts = torch.stack([torch.bincount(e.flatten(), minlength=k * k + 1)[: k * k] for e in enc])
    return counts.view(-1, k, k).cpu().numpy().astype(np.int32)


def head_scores(prob):
    msp = prob.max(dim=1).values.flatten(1).mean(1)
    ent = -(prob.clamp_min(1e-12).log() * prob).sum(1).flatten(1).mean(1)
    return msp, ent


@torch.inference_mode()
def collect(model, dl, corrupt, kind, severity, k, plugin=None):
    model.infer_mode = "fuse"
    rows = {key: [] for key in ("beta_o", "beta_s", "msp_f", "msp_s", "msp_o", "ent_f",
                                "ent_s", "ent_o", "dis_fs", "dis_fo", "dis_so")}
    conf = {"fused": [], "sar": [], "opt": []}
    for batch_index, (x, target) in enumerate(dl):
        generator = torch.Generator().manual_seed(10_000 + batch_index)
        x = corrupt(x.cuda(), kind, severity, generator)
        y = target["cls"].cuda().long()
        if plugin is None:
            model(x, target)
            probs = {"fused": model.last_prob_fused, "sar": model.last_prob_sar_head,
                     "opt": model.last_prob_rgb_head}
            betas = {"rgb": model.last_beta_rgb, "sar": model.last_beta_sar}
        else:
            probs, betas = plugin(x, target)
        preds = {r: p.argmax(1) for r, p in probs.items()}
        b = x.shape[0]
        for route, pred in preds.items():
            conf[route].append(per_image_confusion(pred, y, k))
        for route, tag in (("fused", "f"), ("sar", "s"), ("opt", "o")):
            msp, ent = head_scores(probs[route].float())
            rows[f"msp_{tag}"].append(msp.cpu())
            rows[f"ent_{tag}"].append(ent.cpu())
        rows["dis_fs"].append((preds["fused"] != preds["sar"]).flatten(1).float().mean(1).cpu())
        rows["dis_fo"].append((preds["fused"] != preds["opt"]).flatten(1).float().mean(1).cpu())
        rows["dis_so"].append((preds["sar"] != preds["opt"]).flatten(1).float().mean(1).cpu())
        for key, m in (("beta_o", "rgb"), ("beta_s", "sar")):
            v = betas[m].float().reshape(-1)
            rows[key].append((v.expand(b) if v.numel() == 1 else v).cpu())
    out = {key: torch.cat(v).numpy() for key, v in rows.items()}
    out.update({f"conf_{r}": np.concatenate(v) for r, v in conf.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--splits", default="val,test")
    args = ap.parse_args()
    config = import_config(args.config)
    k = int(config["model"]["params"]["num_classes"])
    is_coder = (Path(args.run_dir) / "Final.pth").exists()
    model = load_model(config, Path(args.run_dir) / ("Final.pth" if is_coder else "Best.pth"))
    plugin = None if isinstance(model, CoDER) else PluginGate(model, config)
    corrupt = Corruptor(config)
    out_root = REPO / "revision_tgrs" / "gate_analysis" / args.config.split(".")[-1] / f"seed_{args.seed}"
    for split in args.splits.split(","):
        dl = loader(config, "test" if split == "val" else "heldout_test", 4)  # data['test'] = val
        (out_root / split).mkdir(parents=True, exist_ok=True)
        for name, kind, severity in conditions():
            path = out_root / split / f"{name}.npz"
            if path.exists():
                continue
            start = time.time()
            np.savez_compressed(path, **collect(model, dl, corrupt, kind, severity, k, plugin))
            print(f"[{split}] {name} ({time.time() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
