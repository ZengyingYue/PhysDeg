"""Post-hoc temperature scaling of CoDER's evidential confidence — the cheap test
for whether the evidential uncertainty can be calibrated (QUESTIONS Q9).

Evidential Dirichlet alpha = model.last_alpha. A single temperature T sharpens it:
    alpha_T = (alpha - 1) * T + 1 ;  p_T = alpha_T / sum ;  conf = max_k p_T .
T>1 shrinks u and fixes underconfidence WITHOUT changing the prediction (argmax is
T-invariant ⇒ accuracy unchanged). We fit ONE T on a held-out clean-fuse split
(minimize ECE), then report evidential ECE/AURC/Sel at T=1 (raw) vs T* (calibrated)
across clean/missing/degraded conditions. Fit and report use disjoint batches.

Answers: can u be made calibrated post-hoc (ECE → ~softmax 0.061), and does a
clean-fit T still hold under degradation?
"""
import argparse
import numpy as np
import torch

from ever.core.config import import_config
from ever.core.builder import make_model
from module.baseline.CoDER import CoDER  # noqa: F401 — registers 'CoDER'
from tools.eval_robust import degrade_rgb, build_test_loader
from tools.eval_calib import Calib, NB_ECE, COVERAGES

FIT_MOD = 3   # batches with i%3==0 fit T; the rest report (disjoint)


def ece_np(conf, correct, nb=NB_ECE):
    bi = np.clip((conf * nb).astype(np.int64), 0, nb - 1)
    c = np.bincount(bi, weights=conf, minlength=nb)
    a = np.bincount(bi, weights=correct, minlength=nb)
    n = np.bincount(bi, minlength=nb)
    N, m = max(n.sum(), 1), n > 0
    return float(np.sum(n[m] / N * np.abs(a[m] / n[m] - c[m] / n[m])))


def conf_at_T_np(alpha, T):
    aT = (alpha - 1.0) * T + 1.0
    return (aT / aT.sum(-1, keepdims=True)).max(-1)


@torch.no_grad()
def collect_fit(model, loader, device, sub=60000):
    model.eval(); model.infer_mode = "fuse"
    A, C = [], []
    for i, (ms, y) in enumerate(loader):
        if i % FIT_MOD != 0:
            continue
        ms = ms.to(device); model(ms, y)
        al = model.last_alpha
        gt = y["cls"].to(device); v = gt != -1
        correct = (al.argmax(1) == gt)
        al = al.permute(0, 2, 3, 1)[v]                       # [P,K]
        idx = torch.randperm(al.shape[0], device=device)[:sub]
        A.append(al[idx].float().cpu().numpy())
        C.append(correct[v][idx].float().cpu().numpy())
    return np.concatenate(A), np.concatenate(C)


def fit_T(A, C):
    best = (1e9, 1.0)
    for T in np.geomspace(0.5, 60, 40):
        e = ece_np(conf_at_T_np(A, T), C)
        if e < best[0]:
            best = (e, float(T))
    return best


@torch.no_grad()
def run(model, loader, mode, kind, sev, T, device):
    model.eval(); model.infer_mode = mode
    cal = Calib()
    for i, (ms, y) in enumerate(loader):
        if i % FIT_MOD == 0:
            continue                                          # report on fit's complement
        ms = degrade_rgb(ms, kind, sev).to(device); model(ms, y)
        al = model.last_alpha
        gt = y["cls"].to(device); v = gt != -1
        correct = (al.argmax(1) == gt)
        aT = (al - 1.0) * T + 1.0
        conf = (aT / aT.sum(1, keepdim=True)).max(1).values
        cal.update(conf[v].float().cpu().numpy(), correct[v].float().cpu().numpy())
    return cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="baseline.CoDER_whu_moar_ml_r1")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--tag", default="s4")
    args = ap.parse_args()
    torch.manual_seed(42)

    cfg = import_config(args.config)
    loader = build_test_loader(cfg, args.bs)
    model = make_model(cfg["model"]).cuda()
    ck = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)
    print(f"[{args.tag}] loaded {args.ckpt}", flush=True)

    A, C = collect_fit(model, loader, "cuda")
    e1 = ece_np(conf_at_T_np(A, 1.0), C)
    eopt, T = fit_T(A, C)
    print(f"[{args.tag}] fit on {A.shape[0]} px (acc={C.mean():.4f}): "
          f"ECE(T=1)={e1:.4f} -> ECE(T*={T:.2f})={eopt:.4f}", flush=True)

    covs = "/".join(str(int(c * 100)) for c in COVERAGES)
    for mode, kind, sev, name in [("fuse", "clean", 0, "clean"), ("sar", "clean", 0, "RGB-miss"),
                                  ("rgb", "clean", 0, "SAR-miss"), ("fuse", "blur", 3, "blur@3"),
                                  ("fuse", "haze", 5, "haze@5")]:
        c1 = run(model, loader, mode, kind, sev, 1.0, "cuda")
        cT = run(model, loader, mode, kind, sev, T, "cuda")
        print(f"[{args.tag}] {name:9s} | T=1: ECE={c1.ece():.4f} AURC={c1.aurc()*1e3:6.2f} "
              f"sel@{covs}={'/'.join(f'{c1.sel(cv):.3f}' for cv in COVERAGES)} "
              f"| T*: ECE={cT.ece():.4f} AURC={cT.aurc()*1e3:6.2f} "
              f"sel={'/'.join(f'{cT.sel(cv):.3f}' for cv in COVERAGES)}", flush=True)


if __name__ == "__main__":
    main()
