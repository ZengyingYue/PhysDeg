"""Protocol-v1 evaluation: post-hoc reliability calibration, then one test pass.

    python experiments/revision/evaluate.py --config revision.whu_rebound \
        --run-dir ../runs/revision_v1/whu_rebound/seed_2333 --seed 2333

Order of operations (the test split is read only in step 3):

1. Load ``Best.pth`` (selected on the validation split during training).
2. CoDER with covariance-deviation reliability only: fit the clean reference
   statistics on training patches, then set the fallback thresholds tau to
   the ``--tau-quantile`` quantile of per-image beta on CLEAN validation
   images.  No corrupted image is used.  The finalised state (reference
   buffers + tau) is saved to ``Final.pth``.
3. Evaluate every condition on ``data['heldout_test']`` and write
   ``revision_tgrs/results/<config>/seed_<seed>.json``.

Confidence for the calibration / risk-coverage metrics of the deployed
prediction is the maximum predictive probability for every method.  CoDER's
evidential scores are reported separately under ``evidential``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ever.core.builder import make_dataloader, make_model
from ever.core.config import import_config

CODE_ROOT = Path(__file__).resolve().parents[2]
REPO = CODE_ROOT.parents[1]
sys.path.insert(0, str(CODE_ROOT))

from module.baseline.CoDER import CoDER  # noqa: E402,F401 - registration
from module.baseline.MetaRS import MetaRS  # noqa: E402,F401 - registration
import module.baseline.revision_baselines  # noqa: E402,F401 - registration
import module.baseline.stars  # noqa: E402,F401 - registration (STARS, 2026-11-08)
from module.reliability import phys_degrade as phys  # noqa: E402

BINS = 2000
ECE_BINS = 15
CWR_THRESHOLD = 0.3

# Corruptions are applied in raw sensor units (before normalisation).
OPTICAL = {
    "haze": [0.10, 0.25, 0.40, 0.55, 0.70],   # airlight blend weight, airlight = 230 DN
    "noise": [5.0, 10.0, 20.0, 35.0, 50.0],   # Gaussian sigma in DN
    "blur": [0.5, 1.0, 2.0, 3.0, 4.0],        # Gaussian blur sigma in pixels
    # Held-out types (not used when designing the fallback criterion):
    "cloud": [0.10, 0.25, 0.50, 0.75, 0.95],  # fraction of the image under opaque cloud
    "dark": [0.70, 0.50, 0.30, 0.15, 0.05],   # illumination gain (low light)
    # Local types held out from ReBound-v4's cloud-like training occlusions:
    "stripe": [0.10, 0.25, 0.50, 0.75, 0.95],      # fraction of rows lost (bands of dead lines)
    "localnoise": [0.10, 0.25, 0.50, 0.75, 0.95],  # fraction covered by sigma=60 DN noise boxes
}
SAR = {
    "speckle": [16.0, 8.0, 4.0, 2.0, 1.0],    # multiplicative Gamma speckle, equivalent looks
}
# Physical suite (--suite phys; revision_tgrs/PHYS_PLAN.md), 8-bit imagery except coreg.
PHYS = {
    "phys_haze": [0.15, 0.35, 0.6, 0.9, 1.3],          # optical depth tau0 (alpha 1.3, airlight 230)
    "phys_cloudshadow": [0.1, 0.2, 0.35, 0.5, 0.7],    # cloud cover; shadow cast 64 px away
    "sar_calib": [1.0, 2.0, 3.0, 4.5, 6.0],            # |calibration offset| in dB
    "sar_noisefloor": [10.0, 20.0, 35.0, 50.0, 70.0],  # thermal-noise amplitude in DN
    "sar_rfi": [0.01, 0.025, 0.05, 0.1, 0.2],          # fraction of range lines with RFI (200 DN)
    "coreg": [1.0, 2.0, 4.0, 8.0, 16.0],               # SAR shift w.r.t. optical/labels, pixels
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 23), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
class ScoreStats:
    """Histogram of a confidence score split by correct / wrong pixels."""

    def __init__(self):
        self.correct = np.zeros(BINS, dtype=np.int64)
        self.wrong = np.zeros(BINS, dtype=np.int64)

    def update(self, score: torch.Tensor, correct: torch.Tensor) -> None:
        index = torch.clamp((score.float() * BINS).long(), 0, BINS - 1)
        self.correct += torch.bincount(index[correct], minlength=BINS).cpu().numpy()
        self.wrong += torch.bincount(index[~correct], minlength=BINS).cpu().numpy()

    def summary(self) -> dict:
        # Risk-coverage: accept pixels from the highest confidence down.
        c, w = self.correct[::-1], self.wrong[::-1]
        n = c + w
        cum_n, cum_w = np.cumsum(n), np.cumsum(w)
        total = max(int(cum_n[-1]), 1)
        occupied = cum_n > 0
        aurc = float(np.sum(cum_w[occupied] / cum_n[occupied] * n[occupied] / total))
        k80 = int(np.searchsorted(cum_n, 0.8 * total))
        sel80 = float(1.0 - cum_w[k80] / max(cum_n[k80], 1))
        # Error-detection AUROC: P(score_wrong < score_correct), ties count 1/2.
        pos, neg = self.wrong.astype(np.float64), self.correct.astype(np.float64)
        below = np.cumsum(neg) - neg
        auroc = float(np.sum(pos * (neg.sum() - below - neg)) + 0.5 * np.sum(pos * neg))
        auroc /= max(pos.sum() * neg.sum(), 1.0)
        # AUROC above is P(score_correct > score_wrong): error ranking quality.
        return {"aurc": aurc, "sel_acc_at_80": sel80, "error_auroc": auroc}


class Metrics:
    def __init__(self, classes: int):
        self.k = classes
        self.confusion = np.zeros((classes, classes), dtype=np.int64)
        self.ece_n = np.zeros(ECE_BINS)
        self.ece_conf = np.zeros(ECE_BINS)
        self.ece_acc = np.zeros(ECE_BINS)
        self.cwr_wrong = np.zeros(classes, dtype=np.int64)
        self.msp = ScoreStats()
        self.evidential = {}
        self.nll = 0.0
        self.brier = 0.0
        self.pixels = 0

    def update(self, prob: torch.Tensor, target: torch.Tensor, extra_scores=None) -> None:
        valid = target != -1
        y = target[valid].long()
        p = prob.permute(0, 2, 3, 1)[valid].float()
        if not bool(torch.isfinite(p).all()):
            raise FloatingPointError("non-finite probability")
        conf, pred = p.max(dim=1)
        correct = pred.eq(y)
        self.confusion += torch.bincount(y * self.k + pred, minlength=self.k ** 2) \
            .reshape(self.k, self.k).cpu().numpy()
        index = torch.clamp((conf * ECE_BINS).long(), 0, ECE_BINS - 1)
        self.ece_n += torch.bincount(index, minlength=ECE_BINS).cpu().numpy()
        self.ece_conf += torch.bincount(index, weights=conf.double(), minlength=ECE_BINS).cpu().numpy()
        self.ece_acc += torch.bincount(index, weights=correct.double(), minlength=ECE_BINS).cpu().numpy()
        confident_wrong = (~correct) & (conf > CWR_THRESHOLD)
        self.cwr_wrong += torch.bincount(y[confident_wrong], minlength=self.k).cpu().numpy()
        self.msp.update(conf, correct)
        true_p = p.gather(1, y[:, None]).squeeze(1)
        self.nll += float(-torch.log(true_p.clamp_min(1e-12)).sum())
        self.brier += float((p.square().sum(1) - 2 * true_p + 1).sum())
        self.pixels += int(y.numel())
        for name, score in (extra_scores or {}).items():
            self.evidential.setdefault(name, ScoreStats()).update(
                score.permute(0, 2, 3, 1)[valid].squeeze(1) if score.dim() == 4
                else score[valid], correct)

    def summary(self) -> dict:
        inter = np.diag(self.confusion).astype(np.float64)
        gt = self.confusion.sum(1)
        union = self.confusion.sum(0) + gt - inter
        present = union > 0
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=present)
        occ = self.ece_n > 0
        ece = float(np.sum(self.ece_n[occ] / self.ece_n.sum()
                           * np.abs(self.ece_acc[occ] / self.ece_n[occ]
                                    - self.ece_conf[occ] / self.ece_n[occ])))
        out = {
            "pixels": self.pixels,
            "miou": float(iou[present].mean()),
            "oa": float(inter.sum() / max(self.confusion.sum(), 1)),
            "per_class_iou": iou.tolist(),
            "ece15": ece,
            "nll": self.nll / max(self.pixels, 1),
            "brier": self.brier / max(self.pixels, 1),
            "cwr_at_0.3_per_class": np.divide(
                self.cwr_wrong, gt, out=np.zeros(self.k), where=gt > 0).tolist(),
            "msp": self.msp.summary(),
            "confusion": self.confusion.tolist(),
        }
        if self.evidential:
            out["evidential"] = {k: v.summary() for k, v in self.evidential.items()}
        return out


# --------------------------------------------------------------------------- #
# corruptions
# --------------------------------------------------------------------------- #
class Corruptor:
    """Apply a raw-unit corruption to a normalised [SAR..., R, G, B] batch.

    Severities are defined for 8-bit optical imagery (WHU-OPT-SAR); datasets
    with other radiometry (SEN12MS-CR) are evaluated on real clouds only.
    """

    def __init__(self, config: dict):
        tf = config["data"]["heldout_test"]["params"]["transforms"]
        norm = next(t for t in tf.transforms if type(t).__name__ == "Normalize")
        self.mean = torch.tensor(norm.mean, dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.tensor(norm.std, dtype=torch.float32).view(1, -1, 1, 1)
        self.ns = self.mean.shape[1] - 3  # SAR channels precede R, G, B

    def __call__(self, x, kind, severity, generator):
        if kind == "none":
            return x
        if kind == "coreg":  # geometric, unit-free: shift the normalised SAR channels
            angle = 2 * math.pi * float(torch.rand(1, generator=generator))
            r = PHYS[kind][severity - 1]
            sar = phys.apply_coreg(x[:, :self.ns], r * math.sin(angle), r * math.cos(angle))
            return torch.cat([sar, x[:, self.ns:]], dim=1)
        mean, std = self.mean.to(x.device), self.std.to(x.device)
        raw = x * std + mean
        if kind in PHYS:
            value = PHYS[kind][severity - 1]
            sar, rgb = raw[:, :self.ns], raw[:, self.ns:self.ns + 3]
            if kind == "phys_haze":
                field = phys.fractal_field(rgb.shape[0], *rgb.shape[2:], x.device, 3.0, generator)
                rgb = phys.apply_spectral_haze(rgb, value, 1.3, 230.0, field)
            elif kind == "phys_cloudshadow":
                angle = 2 * math.pi * float(torch.rand(1, generator=generator))
                rgb, self.last_mask = phys.apply_cloud_shadow(
                    rgb, value, (64 * math.sin(angle), 64 * math.cos(angle)), 0.3, 2.0, 1.0,
                    240.0, generator)
            elif kind == "sar_calib":
                sign = 1.0 if float(torch.rand(1, generator=generator)) < 0.5 else -1.0
                sar = phys.apply_calib(sar, sign * value, 0.0, "amp")
            elif kind == "sar_noisefloor":
                sar = phys.apply_noisefloor(sar, value, "amp", generator=generator)
            elif kind == "sar_rfi":
                sar = phys.apply_rfi(sar, value, 200.0, "amp", generator=generator)
            raw = torch.cat([sar.clamp(0, 255), rgb.clamp(0, 255)], dim=1)
            return (raw - mean) / std
        if kind in OPTICAL:
            value = OPTICAL[kind][severity - 1]
            rgb = raw[:, self.ns:self.ns + 3]
            if kind == "haze":
                rgb = rgb * (1 - value) + 230.0 * value
            elif kind == "noise":
                noise = torch.randn(rgb.shape, generator=generator, device="cpu").to(x.device)
                rgb = rgb + noise * value
            elif kind == "cloud":
                b, _, h, w = rgb.shape
                field = torch.randn((b, 1, max(h // 32, 2), max(w // 32, 2)), generator=generator)
                field = F.interpolate(field.to(x.device), size=(h, w), mode="bicubic",
                                      align_corners=False)
                flat = field.flatten(1)
                thr = torch.quantile(flat, 1.0 - value, dim=1).view(b, 1, 1, 1)
                soft = 0.1 * flat.std(dim=1).view(b, 1, 1, 1)
                mask = ((field - thr) / soft + 0.5).clamp(0, 1)
                self.last_mask = mask  # ground-truth cloud opacity (localization probes)
                rgb = rgb * (1 - mask) + 240.0 * mask
            elif kind == "dark":
                rgb = rgb * value
            elif kind == "stripe":
                b, _, h, w = rgb.shape
                band = 8  # rows per dead band
                n_bands = h // band
                lost = torch.rand((b, n_bands), generator=generator) < value
                rows = lost.repeat_interleave(band, dim=1)[:, :h].to(x.device)
                mask = rows.view(b, 1, h, 1).expand(b, 1, h, w).float()
                self.last_mask = mask
                rgb = rgb * (1 - mask)
            elif kind == "localnoise":
                b, _, h, w = rgb.shape
                cell = 64  # boxes on a 64-pixel grid
                grid = torch.rand((b, 1, h // cell, w // cell), generator=generator) < value
                mask = F.interpolate(grid.float(), size=(h, w), mode="nearest").to(x.device)
                self.last_mask = mask
                noise = torch.randn(rgb.shape, generator=generator).to(x.device) * 60.0
                rgb = rgb + noise * mask
            else:
                radius = max(1, int(math.ceil(3 * value)))
                axis = torch.arange(-radius, radius + 1, dtype=torch.float32, device=x.device)
                kernel = torch.exp(-axis ** 2 / (2 * value ** 2))
                kernel = kernel / kernel.sum()
                k2 = (kernel[:, None] * kernel[None, :]).expand(3, 1, -1, -1)
                rgb = F.conv2d(F.pad(rgb, [radius] * 4, mode="reflect"), k2, groups=3)
            raw = torch.cat([raw[:, :self.ns], rgb.clamp(0, 255)], dim=1)
        elif kind in SAR:
            looks = SAR[kind][severity - 1]
            shape = torch.full(tuple(raw[:, :self.ns].shape), looks)
            speckle = torch._standard_gamma(shape, generator=generator).div(looks).to(x.device)
            raw = torch.cat([(raw[:, :self.ns] * speckle).clamp(0, 255), raw[:, self.ns:]], dim=1)
        else:
            raise ValueError(kind)
        return (raw - mean) / std


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def loader(config: dict, split: str, batch_size: int, limit: int = 0):
    block = copy.deepcopy(config["data"][split])
    params = block["params"]
    for key, value in dict(distributed=False, training=False, batch_size=batch_size,
                           num_workers=2).items():
        params[key] = value
    if "cache_ram" in params:
        params["cache_ram"] = False
    if split in ("mmr", "train"):
        params["transforms"] = copy.deepcopy(config["data"]["heldout_test"]["params"]["transforms"])
    dl = make_dataloader(block)
    if limit and limit < len(dl.dataset):
        # Evenly spaced so city-sorted datasets (EarthMiss) are not biased.
        picks = np.linspace(0, len(dl.dataset) - 1, limit).round().astype(int).tolist()
        subset = torch.utils.data.Subset(dl.dataset, picks)
        dl = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)
    return dl


def load_model(config: dict, checkpoint: Path):
    model = make_model(config["model"]).cuda().eval()
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model", payload)
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


# --------------------------------------------------------------------------- #
# post-hoc reliability calibration (CoDER only)
# --------------------------------------------------------------------------- #
def needs_reliability(model) -> bool:
    return (isinstance(model, CoDER)
            and getattr(model, "observed_cov_reliability", None) is not None)


@torch.inference_mode()
def whitening_features(model, x):
    sar = x[:, : model.config.sar_in_channels]
    rgb = x[:, model.config.sar_in_channels:]
    return {"sar": model._encode_sar(sar)[3], "rgb": model._encode_rgb(rgb)[3]}


@torch.inference_mode()
def calibrate_pixel_stats(model, config, train_limit, tau_quantile):
    """Model-free image / local statistics (ReBound-v4): moments from clean
    training images, thresholds from clean validation images."""
    if not hasattr(model, "rel_img_mu_o"):
        return None
    from module.reliability import pixel_stats as pxs
    img = {"s": [], "o": []}
    loc_sum = loc_sq = None
    n = 0
    for x, _ in loader(config, "mmr", 4, train_limit):
        x = x.cuda()
        ns = model.config.sar_in_channels
        img["s"].append(pxs.image_stats(x[:, :ns]).cpu())
        img["o"].append(pxs.image_stats(x[:, ns:]).cpu())
        ls = pxs.local_stats(x[:, ns:]).transpose(0, 1).flatten(1).double()
        loc_sum = ls.sum(1) if loc_sum is None else loc_sum + ls.sum(1)
        loc_sq = (ls ** 2).sum(1) if loc_sq is None else loc_sq + (ls ** 2).sum(1)
        n += ls.shape[1]
    for key, m in (("s", "s"), ("o", "o")):
        v = torch.cat(img[key])
        getattr(model, f"rel_img_mu_{m}").copy_(v.mean(0))
        getattr(model, f"rel_img_sd_{m}").copy_(v.std(0).clamp_min(1e-6))
    mu = loc_sum / n
    model.rel_loc_mu.copy_(mu.float())
    model.rel_loc_sd.copy_((loc_sq / n - mu ** 2).clamp_min(1e-12).sqrt().float())
    dev = {"s": [], "o": []}
    loc = []
    g = torch.Generator(device="cpu").manual_seed(0)
    for x, _ in loader(config, "test", 4):  # clean validation images
        x = x.cuda()
        ns = model.config.sar_in_channels
        dev_s, dev_o = model._image_deviation(x[:, :ns], x[:, ns:])
        dev["s"].append(dev_s.cpu())
        dev["o"].append(dev_o.cpu())
        d = pxs.max_abs_z(pxs.local_stats(x[:, ns:]), model.rel_loc_mu, model.rel_loc_sd).flatten()
        loc.append(d[torch.randint(0, d.numel(), (20000,), generator=g).to(d.device)].cpu())
    q = 1.0 - tau_quantile
    img_tau = {m: float(torch.quantile(torch.cat(v), q)) for m, v in dev.items()}
    loc_tau = float(torch.quantile(torch.cat(loc), q))
    model.rel_img_tau.copy_(torch.tensor([img_tau["s"], img_tau["o"]]))
    model.rel_loc_tau.fill_(loc_tau)
    return {"img_tau": img_tau, "loc_tau": loc_tau, "train_images": len(torch.cat(img["s"]))}


@torch.inference_mode()
def calibrate_reliability(model, config, train_limit, tau_quantile, log):
    ref = model.observed_cov_reliability
    ref.reset_calibration()
    train_dl = loader(config, "mmr", 4, train_limit)
    for x, _ in train_dl:
        feats = whitening_features(model, x.cuda())
        for m in ("sar", "rgb"):
            ref.update(feats[m], m)
    ref.finalize_statistics()
    scores = {m: ([], []) for m in ("sar", "rgb")}
    for x, _ in train_dl:
        feats = whitening_features(model, x.cuda())
        for m in ("sar", "rgb"):
            scores[m][0].append(torch.stack(ref.deviation_scores(feats[m], m), 1).cpu())
            scores[m][1].append(torch.stack(ref.diagonal_scores(feats[m], m), 1).cpu())
    for m in ("sar", "rgb"):
        off, diag = torch.cat(scores[m][0]), torch.cat(scores[m][1])
        ref.set_score_calibration(list(off.T), m)
        ref.set_diag_calibration(list(diag.T), m)
    betas = {"sar": [], "rgb": []}
    for x, _ in loader(config, "test", 4):  # data['test'] is the validation split
        feats = whitening_features(model, x.cuda())
        for m in ("sar", "rgb"):
            betas[m].append(ref(feats[m], m).flatten().cpu())
    tau = {m: float(torch.quantile(torch.cat(b), tau_quantile)) for m, b in betas.items()}
    model.fallback_tau.copy_(torch.tensor([tau["sar"], tau["rgb"]]))
    pixel_info = calibrate_pixel_stats(model, config, train_limit, tau_quantile)
    # Head-confidence thresholds (clean validation, same quantile) for the
    # 'hybrid' / 'pix_or_conf' fallback rules.
    msp = {"sar": [], "rgb": []}
    model.infer_mode = "fuse"
    for x, target in loader(config, "test", 4):
        model(x.cuda(), target)
        for m, attr in (("sar", "last_prob_sar_head"), ("rgb", "last_prob_rgb_head")):
            msp[m].append(getattr(model, attr).float().max(dim=1).values.flatten(1).mean(1).cpu())
    msp_tau = {m: float(torch.quantile(torch.cat(v), tau_quantile)) for m, v in msp.items()}
    model.fallback_msp_tau.copy_(torch.tensor([msp_tau["sar"], msp_tau["rgb"]]))
    info = {
        "train_patches": len(train_dl.dataset),
        "val_images": int(torch.cat(betas["rgb"]).numel()),
        "tau_quantile": tau_quantile,
        "tau": tau,
        "msp_tau": msp_tau,
        "pixel_stats": pixel_info,
        "fallback_rule": getattr(model, "fallback_rule", "beta"),
        "val_beta_quantiles": {m: {q: float(torch.quantile(torch.cat(b), q))
                                   for q in (0.01, 0.05, 0.5)} for m, b in betas.items()},
    }
    log(f"reliability calibrated: {info}")
    return info


# --------------------------------------------------------------------------- #
# temperature scaling fitted on the validation split
# --------------------------------------------------------------------------- #
def fit_temperature(model, config, mode, pixels_per_image=512):
    """Single temperature on log-probabilities, minimising validation NLL."""
    logp, label = _validation_pixels(model, config, mode, pixels_per_image)
    logp, label = logp.clone(), label.clone().long()  # leave inference mode; int8 EarthMiss masks
    log_t = torch.zeros(1, requires_grad=True)
    optim = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        optim.zero_grad()
        loss = F.cross_entropy(logp / log_t.exp(), label)
        loss.backward()
        return loss

    optim.step(closure)
    return float(log_t.exp()), int(label.numel())


@torch.inference_mode()
def _validation_pixels(model, config, mode, pixels_per_image):
    model.infer_mode = mode
    logps, labels = [], []
    for batch_index, (x, target) in enumerate(loader(config, "test", 4)):  # validation split
        prob = model(x.cuda(), target)
        y = target["cls"].cuda()
        g = torch.Generator(device="cpu").manual_seed(batch_index)
        for b in range(x.shape[0]):
            valid = (y[b] != -1).flatten().nonzero().squeeze(1)
            if valid.numel() == 0:
                continue
            pick = valid[torch.randperm(valid.numel(), generator=g)[:pixels_per_image].to(valid.device)]
            lp = torch.log(prob[b].flatten(1).T[pick].float().clamp_min(1e-12))
            logps.append(lp.cpu())
            labels.append(y[b].flatten()[pick].cpu())
    return torch.cat(logps), torch.cat(labels)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def conditions(is_coder: bool, suite: str = "std"):
    if suite == "phys":
        return [("clean", "fuse", "none", 0)] + [
            (f"{kind}@{s}", "fuse", kind, s) for kind in PHYS for s in range(1, 6)]
    out = [("clean", "fuse", "none", 0), ("missing_optical", "sar", "none", 0),
           ("missing_sar", "rgb", "none", 0)]
    for kind in OPTICAL:
        out += [(f"optical_{kind}@{s}", "fuse", kind, s) for s in range(1, 6)]
    for kind in SAR:
        out += [(f"sar_{kind}@{s}", "fuse", kind, s) for s in range(1, 6)]
    return out


@torch.inference_mode()
def evaluate_condition(model, dl, corrupt, mode, kind, severity, classes, is_coder,
                       temperature=None):
    model.infer_mode = mode
    gated, fused, scaled = Metrics(classes), Metrics(classes), Metrics(classes)
    routes = np.zeros(3, dtype=np.int64)
    betas = {"sar": [], "rgb": []}
    for batch_index, (x, target) in enumerate(dl):
        generator = torch.Generator().manual_seed(10_000 + batch_index)
        x = corrupt(x.cuda(), kind, severity, generator)
        y = target["cls"].cuda()
        prob = model(x, target)
        extra = None
        if is_coder:
            u = model.last_uncertainty
            extra = {"one_minus_vacuity": (1.0 - u),
                     "evidential_max_prob": model.last_evi_prob.max(dim=1).values}
            for m, attr in (("sar", "last_beta_sar"), ("rgb", "last_beta_rgb")):
                value = getattr(model, attr, None)
                if value is not None:
                    betas[m].append(value.reshape(-1).expand(x.shape[0]).float().cpu()
                                    if value.numel() == 1 else value.reshape(-1).float().cpu())
        gated.update(prob, y, extra)
        if temperature is not None:
            scaled.update(torch.softmax(torch.log(prob.float().clamp_min(1e-12)) / temperature, 1), y)
        if mode == "fuse" and is_coder:
            routes += np.bincount(model.last_route.cpu().numpy(), minlength=3)
            fused.update(model.last_prob_fused, y)
    result = gated.summary()
    if temperature is not None:
        ts = scaled.summary()
        result["temperature_scaled"] = {"temperature": temperature, "ece15": ts["ece15"],
                                        "nll": ts["nll"], "msp": ts["msp"]}
    if mode == "fuse" and is_coder:
        result["always_fuse"] = fused.summary()
        result["route_fraction"] = dict(zip(("fused", "sar_head", "optical_head"),
                                            (routes / max(routes.sum(), 1)).tolist()))
    for m, values in betas.items():
        if values:
            v = torch.cat(values)
            result[f"beta_{m}"] = {"mean": float(v.mean()), "q05": float(torch.quantile(v, 0.05))}
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--checkpoint", default="Best.pth")
    ap.add_argument("--train-limit", type=int, default=1024,
                    help="training patches used for the reliability reference")
    ap.add_argument("--tau-quantile", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--only", default="", help="comma-separated condition names (debug)")
    ap.add_argument("--out", default="")
    ap.add_argument("--test-key", default="heldout_test",
                    help="data block to evaluate (e.g. dfc_hr_validation for s12cr models)")
    ap.add_argument("--final-name", default="Final.pth",
                    help="where the calibrated state is saved inside --run-dir "
                         "(use a distinct name for inference-only ablation configs)")
    ap.add_argument("--suite", choices=("std", "phys"), default="std",
                    help="'phys': physical degradations (results/<tag>_phys/)")
    args = ap.parse_args()

    torch.manual_seed(0)
    run_dir = Path(args.run_dir).resolve()
    tag = args.config.split(".")[-1] + ("" if args.test_key == "heldout_test" else f"__{args.test_key}")
    tag += "_phys" if args.suite == "phys" else ""
    out = Path(args.out) if args.out else (
        REPO / "revision_tgrs" / "results" / tag / f"seed_{args.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".log")

    def log(msg):
        line = f"[{time.strftime('%F %T')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as handle:
            handle.write(line + "\n")

    config = import_config(args.config)
    classes = int(config["model"]["params"]["num_classes"])
    ckpt = run_dir / args.checkpoint
    model = load_model(config, ckpt)
    is_coder = isinstance(model, CoDER)
    record = {
        "config": args.config, "seed": args.seed, "run_dir": str(run_dir),
        "checkpoint": str(ckpt), "checkpoint_sha256": sha256(ckpt),
        "evaluator_sha256": sha256(Path(__file__)),
        "best_checkpoint": json.loads((run_dir / "best_checkpoint.json").read_text())
        if (run_dir / "best_checkpoint.json").exists() else None,
        "split_manifest": json.loads((REPO / "revision_tgrs" / "splits" / "manifest.json").read_text()),
        "corruptions": {"optical": OPTICAL, "sar": SAR} if args.suite == "std" else {"phys": PHYS},
        "conditions": {},
    }
    if needs_reliability(model):
        record["reliability"] = calibrate_reliability(
            model, config, args.train_limit, args.tau_quantile, log)
        final = run_dir / args.final_name
        torch.save({"model": model.state_dict()}, final)
        record["final_checkpoint_sha256"] = sha256(final)

    test_dl = loader(config, args.test_key, args.batch_size)
    record["test_key"] = args.test_key
    record["test_images"] = len(test_dl.dataset)
    corrupt = Corruptor(config)
    # Temperatures are fitted on the validation split, one per deployment mode.
    record["temperature"] = {}
    for mode in ("fuse", "sar", "rgb"):
        t, n = fit_temperature(model, config, mode)
        record["temperature"][mode] = {"T": t, "val_pixels": n}
        log(f"temperature[{mode}] = {t:.4f} (val pixels {n})")
    only = set(filter(None, args.only.split(",")))
    for name, mode, kind, severity in conditions(is_coder, args.suite):
        if only and name not in only:
            continue
        start = time.time()
        result = evaluate_condition(model, test_dl, corrupt, mode, kind, severity, classes, is_coder,
                                    record["temperature"][mode]["T"])
        record["conditions"][name] = result
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=1) + "\n")
        tmp.replace(out)
        extra = ""
        if "always_fuse" in result:
            extra = (f" always_fuse={result['always_fuse']['miou']:.4f}"
                     f" routes={result['route_fraction']}")
        log(f"{name}: mIoU={result['miou']:.4f} ECE={result['ece15']:.4f} "
            f"AURC={result['msp']['aurc']:.4f}{extra} ({time.time() - start:.0f}s)")
    out.with_suffix(".done").write_text(time.strftime("%F %T") + "\n")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
