"""WHU/EarthMiss/Potsdam calibration eval for a CoDER checkpoint.

For every condition we run the model once and score TWO readouts fairly — each on
its OWN prediction and its OWN max-probability confidence (no head mismatch):

  * soft = softmax of the fused head            (the accuracy path; model output)
  * evi  = evidential Dempster prob  α/S         (model.last_evi_prob)

Metrics per readout (pixel-level, ignore excluded):
  acc   – pixel accuracy of that readout's argmax.
  ECE   – Expected Calibration Error of its max-prob confidence (15 bins).
  AURC  – Area Under the Risk-Coverage curve (×10³): rank by confidence, keep the
          most-confident first; lower = confidence better ranks errors.
  Sel   – selective accuracy keeping the C-fraction most-confident pixels.
  meanU – mean evidential uncertainty u=K/S (descriptive; should rise w/ degradation).

Histograms (200 bins) keep the whole sweep O(bins) in memory. Eval is fp32.
"""
import argparse
import numpy as np
import torch

from ever.core.config import import_config
from ever.core.builder import make_model
from module.baseline.CoDER import CoDER  # noqa: F401 — registers 'CoDER'
from tools.eval_robust import degrade_rgb, build_test_loader  # noqa: F401

NB_ECE = 15
NB_RC = 200
COVERAGES = (1.0, 0.9, 0.8, 0.6)


class Calib:
    """Single confidence signal in (0,1] vs binary correctness (streaming)."""

    def __init__(self):
        self.ec_c = np.zeros(NB_ECE)   # Σ confidence
        self.ec_a = np.zeros(NB_ECE)   # Σ correct
        self.ec_n = np.zeros(NB_ECE)
        self.rc_n = np.zeros(NB_RC)
        self.rc_e = np.zeros(NB_RC)    # errors, binned by confidence

    def update(self, conf, correct):
        bi = np.clip((conf * NB_ECE).astype(np.int64), 0, NB_ECE - 1)
        self.ec_c += np.bincount(bi, weights=conf, minlength=NB_ECE)
        self.ec_a += np.bincount(bi, weights=correct, minlength=NB_ECE)
        self.ec_n += np.bincount(bi, minlength=NB_ECE)
        ci = np.clip((conf * NB_RC).astype(np.int64), 0, NB_RC - 1)
        self.rc_n += np.bincount(ci, minlength=NB_RC)
        self.rc_e += np.bincount(ci, weights=1.0 - correct, minlength=NB_RC)

    def acc(self):
        return float(self.ec_a.sum() / max(self.ec_n.sum(), 1))

    def ece(self):
        N = max(self.ec_n.sum(), 1)
        m = self.ec_n > 0
        return float(np.sum(self.ec_n[m] / N * np.abs(self.ec_a[m] / self.ec_n[m]
                                                      - self.ec_c[m] / self.ec_n[m])))

    def aurc(self):
        # keep most-confident first => iterate bins high->low confidence
        order = np.arange(NB_RC)[::-1]
        cn, ce = np.cumsum(self.rc_n[order]), np.cumsum(self.rc_e[order])
        N = cn[-1] if cn[-1] > 0 else 1
        m = cn > 0
        return float(np.sum((ce[m] / cn[m]) * (self.rc_n[order][m] / N)))

    def sel(self, cov):
        order = np.arange(NB_RC)[::-1]
        cn, ce = np.cumsum(self.rc_n[order]), np.cumsum(self.rc_e[order])
        N = cn[-1] if cn[-1] > 0 else 1
        idx = min(np.searchsorted(cn, int(cov * N)), NB_RC - 1)
        return 1.0 - ce[idx] / max(cn[idx], 1)


@torch.no_grad()
def run(model, loader, mode, kind, sev, device, ignore=-1):
    model.eval()
    model.infer_mode = mode
    soft, evi = Calib(), Calib()
    u_sum, n = 0.0, 0
    for ms, y in loader:
        ms = degrade_rgb(ms, kind, sev).to(device)
        prob = model(ms, y)                       # softmax fused (accuracy path)
        evp = model.last_evi_prob                 # evidential α/S
        u = model.last_uncertainty
        gt = y["cls"].to(device)
        v = gt != ignore
        ps, cs = prob.argmax(1), prob.max(1).values
        pe, ce = evp.argmax(1), evp.max(1).values
        soft.update(cs[v].float().cpu().numpy(), (ps == gt)[v].float().cpu().numpy())
        evi.update(ce[v].float().cpu().numpy(), (pe == gt)[v].float().cpu().numpy())
        u_sum += float(u[v].sum()); n += int(v.sum())
    return soft, evi, u_sum / max(n, 1)


def report(tag, soft, evi, meanu):
    sels = lambda c: "/".join(f"{c.sel(cv):.3f}" for cv in COVERAGES)
    covs = "/".join(str(int(cv * 100)) for cv in COVERAGES)
    print(f"=== {tag:24s} | SOFT acc={soft.acc():.4f} ECE={soft.ece():.4f} "
          f"AURC={soft.aurc()*1e3:6.2f} sel@{covs}={sels(soft)} "
          f"| EVI acc={evi.acc():.4f} ECE={evi.ece():.4f} AURC={evi.aurc()*1e3:6.2f} "
          f"sel={sels(evi)} | meanU={meanu:.4f} ===", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="baseline.CoDER_whu_moar_ml_r1")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--kinds", default="blur,noise,haze")
    ap.add_argument("--sevs", default="1,3,5")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(42)

    cfg = import_config(args.config)
    loader = build_test_loader(cfg, args.bs)
    model = make_model(cfg["model"]).cuda()
    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck["model"] if "model" in ck else ck
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"loaded {args.ckpt}: missing={len(miss)} unexpected={len(unexp)}", flush=True)

    report("clean (fuse)", *run(model, loader, "fuse", "clean", 0, "cuda"))
    report("RGB-missing (SAR)", *run(model, loader, "sar", "clean", 0, "cuda"))
    report("SAR-missing (RGB)", *run(model, loader, "rgb", "clean", 0, "cuda"))
    if args.quick:
        report("fuse blur@3", *run(model, loader, "fuse", "blur", 3, "cuda"))
        return
    for kind in args.kinds.split(","):
        for sev in [int(s) for s in args.sevs.split(",")]:
            report(f"fuse {kind}@{sev}", *run(model, loader, "fuse", kind, sev, "cuda"))


if __name__ == "__main__":
    main()
