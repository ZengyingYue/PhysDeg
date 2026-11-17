"""Feasibility probe for PIXEL-level fallback routing (CoDER models).

Per-pixel optical-reliability signals (larger = less reliable), all label-free
and calibrated on clean images only:
  res     cross-modal shared-space residual: mean_c z(c_o - c_s)^2 at the
          finest shared level, z-scored with clean-training statistics.  The
          alignment loss makes c_o predictable from c_s on clean data, so a
          large residual flags optical content SAR cannot account for.
  conf    1 - max probability of the optical head.
  pixdev  local raw-pixel statistics (15x15 mean / std per band) z-scored
          against clean training images (model-free control).
Routing: a pixel takes the SAR head's label where signal > tau (the 99th
percentile of that signal over clean VALIDATION pixels), else the fused
label.  'oracle' takes the SAR label wherever only the SAR head is correct.
Also reports pixel AUROC of each signal against the true cloud mask.

    python experiments/revision/pixel_route_probe.py --config revision.whu_rebound \
        --run-dir ../runs/revision_v1/whu_rebound/seed_2333 --seed 2333
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CODE_ROOT = Path(__file__).resolve().parents[2]
REPO = CODE_ROOT.parents[1]
sys.path.insert(0, str(CODE_ROOT))

from ever.core.config import import_config  # noqa: E402
from experiments.revision.evaluate import Corruptor, load_model, loader  # noqa: E402

SIGNALS = ("res", "conf", "pixdev")
TEST_CONDITIONS = [("clean", "none", 0)] + [(f"optical_cloud@{s}", "cloud", s) for s in range(1, 6)] + [
    (f"optical_{k}@{s}", k, s) for k in ("haze", "noise", "blur", "dark") for s in (3, 5)]


def miou(conf):
    inter = np.diag(conf).astype(float)
    union = conf.sum(0) + conf.sum(1) - inter
    ok = union > 0
    return float((inter[ok] / union[ok]).mean())


def local_stats(raw_rgb):
    mean = F.avg_pool2d(raw_rgb, 15, 1, 7, count_include_pad=False)
    sq = F.avg_pool2d(raw_rgb ** 2, 15, 1, 7, count_include_pad=False)
    return torch.cat([mean, (sq - mean ** 2).clamp_min(0).sqrt()], 1)  # [B,6,H,W]


class Probe:
    def __init__(self, model, corrupt):
        self.model, self.corrupt = model, corrupt
        self.ref = {}

    @torch.inference_mode()
    def signals(self, x, target):
        m = self.model
        m.infer_mode = "fuse"
        m(x, target)
        sar, rgb = x[:, :1], x[:, 1:4]
        c_s = m._encode_sar(sar)[0][0]
        c_o = m._encode_rgb(rgb)[0][0]
        diff = c_o - c_s
        raw = x * self.corrupt.std.to(x.device) + self.corrupt.mean.to(x.device)
        ls = local_stats(raw[:, 1:4])
        out = {"diff": diff, "ls": ls,
               "conf": 1 - m.last_prob_rgb_head.max(1).values,
               "p_f": m.last_prob_fused, "p_s": m.last_prob_sar_head}
        if self.ref:
            z = (diff - self.ref["d_mu"]) / self.ref["d_sd"]
            res = z.pow(2).mean(1, keepdim=True)
            out["res"] = F.interpolate(res, size=x.shape[-2:], mode="bilinear",
                                       align_corners=False).squeeze(1)
            zl = ((ls - self.ref["l_mu"]) / self.ref["l_sd"]).abs().max(1).values
            out["pixdev"] = zl
        return out

    def fit_reference(self, dl):
        d_sum = d_sq = l_sum = l_sq = None
        n_d = n_l = 0
        for x, t in dl:
            s = self.signals(x.cuda(), t)
            d = s["diff"].transpose(0, 1).flatten(1).double()
            l = s["ls"].transpose(0, 1).flatten(1).double()
            d_sum = d.sum(1) if d_sum is None else d_sum + d.sum(1)
            d_sq = d.pow(2).sum(1) if d_sq is None else d_sq + d.pow(2).sum(1)
            l_sum = l.sum(1) if l_sum is None else l_sum + l.sum(1)
            l_sq = l.pow(2).sum(1) if l_sq is None else l_sq + l.pow(2).sum(1)
            n_d += d.shape[1]
            n_l += l.shape[1]
        view = (1, -1, 1, 1)
        d_mu = d_sum / n_d
        l_mu = l_sum / n_l
        self.ref = {"d_mu": d_mu.float().view(view), "d_sd": (d_sq / n_d - d_mu ** 2).clamp_min(1e-8).sqrt().float().view(view),
                    "l_mu": l_mu.float().view(view), "l_sd": (l_sq / n_l - l_mu ** 2).clamp_min(1e-8).sqrt().float().view(view)}


def auroc(pos, neg):
    scores = np.concatenate([pos, neg])
    order = scores.argsort()
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.inference_mode()
def run_condition(probe, dl, kind, severity, k, tau, rng):
    conf = {n: np.zeros((k, k), np.int64) for n in ("fused", "sar", "oracle", *SIGNALS)}
    samples = {n: ([], []) for n in SIGNALS}
    for batch_index, (x, target) in enumerate(dl):
        g = torch.Generator().manual_seed(10_000 + batch_index)
        probe.corrupt.last_mask = None
        x = probe.corrupt(x.cuda(), kind, severity, g)
        mask = probe.corrupt.last_mask
        s = probe.signals(x, target)
        y = target["cls"].cuda().long()
        valid = y != -1
        pf, ps = s["p_f"].argmax(1), s["p_s"].argmax(1)
        preds = {"fused": pf, "sar": ps,
                 "oracle": torch.where((ps == y) & (pf != y), ps, pf)}
        for n in SIGNALS:
            preds[n] = torch.where(s[n] > tau[n], ps, pf)
            if mask is not None:
                cloud = (mask.squeeze(1) > 0.5)
                for bucket, sel in ((0, cloud), (1, ~cloud)):
                    v = s[n][sel].flatten()
                    if v.numel():
                        idx = torch.randint(0, v.numel(), (min(2000, v.numel()),), generator=rng)
                        samples[n][bucket].append(v.cpu()[idx].numpy())
        for n, pred in preds.items():
            enc = y[valid] * k + pred[valid]
            conf[n] += torch.bincount(enc, minlength=k * k).reshape(k, k).cpu().numpy()
    out = {n: miou(c) for n, c in conf.items()}
    for n in SIGNALS:
        if samples[n][0] and samples[n][1]:
            out[f"auroc_{n}"] = auroc(np.concatenate(samples[n][0]), np.concatenate(samples[n][1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    config = import_config(args.config)
    k = int(config["model"]["params"]["num_classes"])
    model = load_model(config, Path(args.run_dir) / "Final.pth")
    probe = Probe(model, Corruptor(config))
    probe.fit_reference(loader(config, "mmr", 4, 256))
    # tau: 99th percentile of each signal over clean validation pixels
    vals = {n: [] for n in SIGNALS}
    rng = torch.Generator().manual_seed(0)
    for x, t in loader(config, "test", 4):
        s = probe.signals(x.cuda(), t)
        for n in SIGNALS:
            v = s[n].flatten()
            vals[n].append(v[torch.randint(0, v.numel(), (20000,), generator=rng).to(v.device)].cpu())
    tau = {n: float(torch.quantile(torch.cat(v), 0.99)) for n, v in vals.items()}
    print("tau", tau, flush=True)
    result = {"tau": tau, "conditions": {}}
    dl = loader(config, "heldout_test", 4)
    out = REPO / "revision_tgrs" / "gate_analysis" / args.config.split(".")[-1] / f"seed_{args.seed}" / "pixel_route.json"
    for name, kind, sev in TEST_CONDITIONS:
        r = run_condition(probe, dl, kind, sev, k, tau, rng)
        result["conditions"][name] = r
        out.write_text(json.dumps(result, indent=1))
        print(name, {a: round(100 * b, 2) if not a.startswith("auroc") else round(b, 3) for a, b in r.items()},
              flush=True)


if __name__ == "__main__":
    main()
