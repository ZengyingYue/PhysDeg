"""Plan B, diagnostic D0 (evaluation only).

Question: does a CROSS-MODAL per-pixel signal locate real clouds while staying
quiet on clean images from new regions, where out-of-distribution (OOD) pixel
statistics misfire?

Signals (larger = optical less trustworthy), per pixel:
  pixstat   max |z| of local (15x15) optical mean / std against clean training
            images (the OOD approach used by the v4 mask)
  disagree  total-variation distance between the optical-only and the
            SAR-only class distributions of the same model, 0.5 * sum |p_o - p_s|
            (cross-modal approach: a new region keeps the sensors consistent,
            a cloud does not)
Threshold per signal = 99th percentile over clean SEN12MS-CR validation pixels.
Reported (validation data only, no test split):
  auroc_cloud   pixel AUROC, cloud vs clear pixels, SEN12MS-CR validation cloudy images
  trigger_*     fraction of pixels above threshold on: SEN12MS-CR val clear,
                SEN12MS-CR val cloudy (cloud / clear pixels), DFC2020 validation (clean,
                other regions, 10 m labels)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
REPO = CODE_ROOT.parents[1]
sys.path.insert(0, str(CODE_ROOT))

from ever.core.config import import_config  # noqa: E402
from data.Sen12msCR import Sen12msCR  # noqa: E402
from experiments.revision.evaluate import load_model  # noqa: E402
from configs.revision.common import DATASETS, DFC_HR_ROOT, S12CR_ROOT, eval_transforms  # noqa: E402
from module.baseline.CoDER import CoDER  # noqa: E402
from module.reliability import pixel_stats as pxs  # noqa: E402

SIGNALS = ("pixstat", "disagree")


def batches(ds, bs=8):
    for i in range(0, len(ds), bs):
        items = [ds[j] for j in range(i, min(i + bs, len(ds)))]
        x = torch.stack([torch.as_tensor(it[0]) for it in items])
        yield i, x, {"cls": torch.stack([torch.as_tensor(it[1]["cls"]) for it in items])}


@torch.inference_mode()
def head_probs(model, x, t):
    """(optical-only, SAR-only) class distributions of one model."""
    if isinstance(model, CoDER):
        model.infer_mode = "fuse"
        model(x, t)
        return model.last_prob_rgb_head.float(), model.last_prob_sar_head.float()
    model.infer_mode = "rgb"   # optical only
    p_o = model(x, t).float()
    model.infer_mode = "sar"   # SAR only
    return p_o, model(x, t).float()


class Signals:
    def __init__(self, model, ns):
        self.model, self.ns = model, ns
        self.mu = self.sd = None

    def fit(self, ds, n=512):
        acc = []
        for i, x, _ in batches(torch.utils.data.Subset(ds, np.linspace(0, len(ds) - 1, n).astype(int).tolist())):
            acc.append(pxs.local_stats(x.cuda()[:, self.ns:]).transpose(0, 1).flatten(1))
        s = torch.cat(acc, 1).double()
        self.mu, self.sd = s.mean(1).float(), s.std(1).clamp_min(1e-6).float()

    @torch.inference_mode()
    def __call__(self, x, t):
        x = x.cuda()
        p_o, p_s = head_probs(self.model, x, t)
        return {"pixstat": pxs.max_abs_z(pxs.local_stats(x[:, self.ns:]), self.mu, self.sd),
                "disagree": 0.5 * (p_o - p_s).abs().sum(1)}


def auroc(pos, neg):
    s = np.concatenate([pos, neg])
    r = np.empty(len(s))
    r[s.argsort(kind="mergesort")] = np.arange(1, len(s) + 1)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    config = import_config(args.config)
    run = Path(args.run_dir)
    model = load_model(config, run / ("Final.pth" if (run / "Final.pth").exists() else "Best.pth"))
    model.reliability_fallback = False if isinstance(model, CoDER) else None
    ds_cfg = DATASETS["s12cr"]
    tf = eval_transforms(ds_cfg)
    sig = Signals(model, ds_cfg["sar_channels"])
    sig.fit(Sen12msCR(S12CR_ROOT, "train", "clear", tf))
    rng = np.random.default_rng(0)
    val_clear = Sen12msCR(S12CR_ROOT, "val", "clear", tf)
    val_cloudy = Sen12msCR(S12CR_ROOT, "val", "cloudy", tf)
    dfc = Sen12msCR(DFC_HR_ROOT, "validation", "clear", tf)

    clean = {s: [] for s in SIGNALS}
    for _, x, t in batches(val_clear):
        for s, v in sig(x, t).items():
            v = v.flatten().cpu().numpy()
            clean[s].append(rng.choice(v, 4000, replace=False))
    tau = {s: float(np.quantile(np.concatenate(v), 0.99)) for s, v in clean.items()}

    pos = {s: [] for s in SIGNALS}
    neg = {s: [] for s in SIGNALS}
    trig_cloud = {s: [0, 0] for s in SIGNALS}
    trig_clear_in_cloudy = {s: [0, 0] for s in SIGNALS}
    for i, x, t in batches(val_cloudy):
        cloud = np.stack([val_cloudy.cloud_mask(i + j) for j in range(x.shape[0])]).astype(bool)
        for s, v in sig(x, t).items():
            v = v.cpu().numpy()
            c, n = v[cloud], v[~cloud]
            if c.size:
                pos[s].append(rng.choice(c, min(2000, c.size), replace=False))
                trig_cloud[s][0] += int((c > tau[s]).sum()); trig_cloud[s][1] += c.size
            if n.size:
                neg[s].append(rng.choice(n, min(2000, n.size), replace=False))
                trig_clear_in_cloudy[s][0] += int((n > tau[s]).sum()); trig_clear_in_cloudy[s][1] += n.size
    trig_dfc = {s: [0, 0] for s in SIGNALS}
    for _, x, t in batches(dfc):
        for s, v in sig(x, t).items():
            trig_dfc[s][0] += int((v > tau[s]).sum()); trig_dfc[s][1] += v.numel()

    result = {"config": args.config, "run_dir": str(run), "tau_q99_clean_val": tau, "signals": {}}
    for s in SIGNALS:
        result["signals"][s] = {
            "auroc_cloud_val": auroc(np.concatenate(pos[s]), np.concatenate(neg[s])),
            "trigger_cloud_pixels_val": trig_cloud[s][0] / max(trig_cloud[s][1], 1),
            "trigger_clear_pixels_in_cloudy_val": trig_clear_in_cloudy[s][0] / max(trig_clear_in_cloudy[s][1], 1),
            "trigger_dfc_clean": trig_dfc[s][0] / max(trig_dfc[s][1], 1),
        }
    out = REPO / "revision_tgrs" / "planB" / "D0" / f"{args.config.split('.')[-1]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result["signals"], indent=1))


if __name__ == "__main__":
    main()
