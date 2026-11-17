"""WHU robustness eval for a CoDER checkpoint.

Graded RGB degradation (blur / noise / haze) + modality-missing, reporting
mIoU/OA. Only the RGB channels (1-3) are corrupted; SAR (channel 0) is left
clean -- so the sweep probes whether the model falls back onto the reliable SAR
stream as RGB degrades (graceful degradation) instead of collapsing below the
SAR-only floor.

Naming: infer_mode='sar' uses SAR only  => the RGB-missing scenario.
        infer_mode='rgb' uses RGB only  => the SAR-missing scenario.

Self-contained; depends only on torch (gaussian blur is a manual depthwise conv,
no torchvision). Eval runs the fp32 master weights (no autocast), so a bf16-
trained checkpoint is evaluated identically to an fp32 one.
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ever.core.config import import_config
from ever.core.builder import make_model
from module.baseline.CoDER import CoDER  # noqa: F401 — registers 'CoDER'


def build_test_loader(cfg, bs):
    """Build the test DataLoader from any of the 3 datasets' config blocks.

    WHU uses WhuM3 (root-based split); EarthMiss & Potsdam use the EarthM3
    layout (image_dir/{SAR,RGB} + masks). Both yield (ms[4,H,W], dict(cls=...)),
    with channel 0 = the reliable modality (SAR or nDSM) and 1-3 = optical.
    """
    tp = cfg["data"]["test"]["params"]
    if "root" in tp:                       # WHU-OPT-SAR
        from data.WHU import WhuM3
        ds = WhuM3(tp["root"], "test", tp["patch"], tp["seed"], tp["train_frac"],
                   tp["transforms"], cache_ram=True)
    else:                                   # EarthMiss / Potsdam
        from data.EarthMiss import EarthM3
        ds = EarthM3(tp["image_dir"], tp["mask_dir"], tp["transforms"], tp["sensors"])
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=8, pin_memory=True)

# Severity tables (index 0 = clean). Values are in the model's *normalized*
# input space (RGB ~ unit std after Normalize), so they are directly comparable
# across channels.
BLUR_SIGMA = [0.0, 0.6, 1.0, 1.6, 2.4, 3.5]
NOISE_STD  = [0.0, 0.05, 0.10, 0.20, 0.35, 0.55]
HAZE_ALPHA = [0.0, 0.15, 0.30, 0.50, 0.70, 0.90]
# "airlight" ~ a near-white pixel in normalized space ((~230-mean)/std per ch).
AIRLIGHT = torch.tensor([5.7, 5.5, 5.6]).view(1, 3, 1, 1)


def _gauss_kernel(sigma, device, dtype):
    k = max(3, int(2 * round(3 * sigma) + 1))
    ax = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kern = (g[:, None] * g[None, :]).to(device, dtype)
    return kern.view(1, 1, k, k), k


def degrade_rgb(ms, kind, sev):
    """Corrupt RGB channels (1-3) of ms [B,4,H,W]; SAR (ch 0) untouched."""
    if kind == "clean" or sev == 0:
        return ms
    out = ms.clone()
    rgb = out[:, 1:4]
    if kind == "blur":
        kern, k = _gauss_kernel(BLUR_SIGMA[sev], rgb.device, rgb.dtype)
        kern = kern.expand(3, 1, k, k)
        rgb = F.conv2d(rgb, kern, padding=k // 2, groups=3)
    elif kind == "noise":
        rgb = rgb + torch.randn_like(rgb) * NOISE_STD[sev]
    elif kind == "haze":
        a = HAZE_ALPHA[sev]
        rgb = rgb * (1 - a) + AIRLIGHT.to(rgb.device, rgb.dtype) * a
    else:
        raise ValueError(kind)
    out[:, 1:4] = rgb
    return out


def confusion(pred, gt, n, ignore=-1):
    m = gt != ignore
    p, g = pred[m].astype(np.int64), gt[m].astype(np.int64)
    return np.bincount(g * n + p, minlength=n * n).reshape(n, n)


@torch.no_grad()
def run(model, loader, mode, kind, sev, device, K):
    model.eval()
    model.infer_mode = mode
    cm = np.zeros((K, K), dtype=np.int64)
    for ms, y in loader:
        ms = degrade_rgb(ms, kind, sev).to(device)
        prob = model(ms, y)
        pred = prob.argmax(1).cpu().numpy()
        cm += confusion(pred, y["cls"].numpy(), K)
    inter = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - inter
    iou = inter / np.maximum(union, 1)
    f1 = 2 * inter / np.maximum(cm.sum(0) + cm.sum(1), 1)   # per-class Dice/F1
    tot = max(cm.sum(), 1)
    oa = inter.sum() / tot
    pe = (cm.sum(0) * cm.sum(1)).sum() / (tot ** 2)          # Cohen's kappa
    kappa = (oa - pe) / max(1 - pe, 1e-9)
    return float(iou.mean()), float(oa), float(f1.mean()), float(kappa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="baseline.CoDER_whu_moar_ml_r1")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--kinds", default="blur,noise,haze")
    ap.add_argument("--degrade_mode", default="fuse",
                    help="readout for the degradation sweep: 'fuse' (softmax) or "
                         "'full' (evidential Dempster α/S)")
    ap.add_argument("--quick", action="store_true",
                    help="only clean + sar + blur@3 (harness smoke)")
    args = ap.parse_args()
    torch.manual_seed(42)  # reproducible noise

    cfg = import_config(args.config)
    K = int(cfg["model"]["params"]["num_classes"])
    loader = build_test_loader(cfg, args.bs)

    model = make_model(cfg["model"]).cuda()
    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck["model"] if "model" in ck else ck
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"loaded {args.ckpt}: K={K} missing={len(miss)} unexpected={len(unexp)}", flush=True)

    # reference points: clean fuse + the two single-modality floors
    for tag, mode in [("clean (fuse)", "fuse"),
                      ("RGB-missing (SAR-only)", "sar"),
                      ("SAR-missing (RGB-only)", "rgb")]:
        miou, oa, mf1, kappa = run(model, loader, mode, "clean", 0, "cuda", K)
        print(f"=== {tag:24s} mIoU={miou:.4f}  OA={oa:.4f}  mF1={mf1:.4f}  Kappa={kappa:.4f} ===", flush=True)

    if args.quick:
        miou, oa, _, _ = run(model, loader, "fuse", "blur", 3, "cuda", K)
        print(f"=== fuse blur@3 (quick)      mIoU={miou:.4f}  OA={oa:.4f} ===", flush=True)
        return

    # graded RGB degradation. degrade_mode selects the readout: 'fuse' (softmax
    # accuracy path) or 'full' (evidential Dempster readout — reliability path).
    dm = args.degrade_mode
    if dm != "fuse":
        miou, oa, _, _ = run(model, loader, dm, "clean", 0, "cuda", K)
        print(f"=== clean ({dm})            mIoU={miou:.4f}  OA={oa:.4f} ===", flush=True)
    for kind in args.kinds.split(","):
        for sev in range(1, 6):
            miou, oa, _, _ = run(model, loader, dm, kind, sev, "cuda", K)
            print(f"=== {dm} {kind}@{sev}            mIoU={miou:.4f}  OA={oa:.4f} ===", flush=True)


if __name__ == "__main__":
    main()
