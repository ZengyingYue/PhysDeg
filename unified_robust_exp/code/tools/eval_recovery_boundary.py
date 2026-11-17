"""ReBound recoverable-boundary analysis (feature space) — direct evidence for the
hallucination claim, without depending on prediction confidence.

On RGB-present data we extract, at a content-pyramid level:
  S = SAR content features        (model._encode_sar)
  R = REAL RGB content features   (model._encode_rgb)
  Z = RECOVERED RGB content       (z_hat, restored from SAR by the flow recovery)

For each RGB feature direction we measure (i) how SAR-predictable it is --- the
linear R^2 of predicting R from S --- and (ii) the recovery error |R - Z|. The
recoverable-information boundary predicts: directions SAR can predict (high R^2 =
SHARED) are recovered accurately (low error); directions SAR cannot predict (low
R^2 = optical-SPECIFIC) are fabricated (high error). Information-theoretically this
is near-forced: SAR does not carry the optical-specific signal, so no recovery can
reconstruct it. Outputs aggregate numbers + a vector-PDF scatter (R^2 vs error).
"""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ever.core.config import import_config
from ever.core.builder import make_model
from module.baseline.CoDER import CoDER  # noqa: F401
from tools.eval_robust import build_test_loader


@torch.no_grad()
def collect(model, loader, level, n_batches, sub, device="cuda"):
    model.eval()
    sc = int(model.config.sar_in_channels); rc = int(model.config.rgb_in_channels)
    S, R, Z = [], [], []
    for i, (ms, y) in enumerate(loader):
        if i >= n_batches:
            break
        ms = ms.to(device)
        sar = ms[:, :sc]; rgb = ms[:, sc:sc + rc]
        sar_content, _, sar_fuse, _ = model._encode_sar(sar)
        rgb_content, _, _, _ = model._encode_rgb(rgb)
        beta_rec = model.cov_reliability.compute(model.cov_matrix_layer, recovered=True)
        rec = model._recover_content(model.decoder_rgb, sar_fuse, sar_content, beta_rec)
        zhat = rec["z_hat"]
        if i == 0:
            print("content pyramid levels (channels): "
                  + ", ".join(f"{li}:{t.shape[1]}@{t.shape[2]}x{t.shape[3]}"
                              for li, t in enumerate(rgb_content)), flush=True)
        s, r, z = sar_content[level], rgb_content[level], zhat[level]
        s = s.permute(0, 2, 3, 1).reshape(-1, s.shape[1])
        r = r.permute(0, 2, 3, 1).reshape(-1, r.shape[1])
        z = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
        idx = torch.randperm(s.shape[0], device=device)[:sub]
        S.append(s[idx].float().cpu().numpy())
        R.append(r[idx].float().cpu().numpy())
        Z.append(z[idx].float().cpu().numpy())
    return np.concatenate(S), np.concatenate(R), np.concatenate(Z)


def analyze(S, R, Z):
    Sc = S - S.mean(0); Rc = R - R.mean(0)
    A, *_ = np.linalg.lstsq(Sc, Rc, rcond=None)        # Cs x Cr : linear S->R
    res = Rc - Sc @ A
    r2 = np.clip(1 - (res ** 2).sum(0) / ((Rc ** 2).sum(0) + 1e-8), 0, 1)  # per-dim SAR-predictability
    err = np.abs(R - Z).mean(0) / (R.std(0) + 1e-6)    # per-dim recovery error (norm.)
    return r2, err


def _invsqrt(M):
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 1e-8, None)
    return V @ np.diag(w ** -0.5) @ V.T


def cca_boundary(S, R, Z, reg=1e-2):
    """Confound-free boundary test via CCA. Canonical variates are unit-variance,
    so recovery error per direction is energy-normalized by construction --- it
    cannot be inflated by low-variance directions (the worry about the naive metric).
    Returns rho (canonical corr = SAR-predictability) and recovery error + fidelity
    of the RECOVERED features projected onto the SAME RGB canonical basis."""
    mS, mR = S.mean(0), R.mean(0)
    Sc, Rc, Zc = S - mS, R - mR, Z - mR
    N = S.shape[0]
    CSS = Sc.T @ Sc / N + reg * np.eye(S.shape[1])
    CRR = Rc.T @ Rc / N + reg * np.eye(R.shape[1])
    CSR = Sc.T @ Rc / N
    WS, WR = _invsqrt(CSS), _invsqrt(CRR)
    U, D, Vt = np.linalg.svd(WS @ CSR @ WR)            # D = canonical correlations
    rho = np.clip(D, 0, 1)
    Brgb = WR @ Vt.T                                    # RGB canonical directions (orig space)
    Rcv, Zcv = Rc @ Brgb, Zc @ Brgb                     # variates (Rcv ~ unit-variance)
    err = np.abs(Rcv - Zcv).mean(0) / (Rcv.std(0) + 1e-6)
    fid = np.array([np.corrcoef(Rcv[:, j], Zcv[:, j])[0, 1] for j in range(Rcv.shape[1])])
    return rho, err, np.nan_to_num(fid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="baseline.CoDER_whu_moar_ml_r1")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--level", type=int, default=2, help="content-pyramid level index")
    ap.add_argument("--n_batches", type=int, default=20)
    ap.add_argument("--sub", type=int, default=6000, help="pixels sampled per batch")
    ap.add_argument("--out", default="recovery_boundary.pdf")
    args = ap.parse_args()
    torch.manual_seed(42)

    cfg = import_config(args.config)
    loader = build_test_loader(cfg, args.bs)
    model = make_model(cfg["model"]).cuda()
    ck = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)

    S, R, Z = collect(model, loader, args.level, args.n_batches, args.sub)
    print(f"=== RECOVERY BOUNDARY (level {args.level}, {S.shape[0]} px, {R.shape[1]} RGB dims) ===", flush=True)

    # (1) naive per-dim metric (energy-confounded)
    r2, err = analyze(S, R, Z)
    print(f"  [naive] corr(R^2, recovery-error) = {np.corrcoef(r2, err)[0,1]:+.3f}", flush=True)

    # (2) CCA confound-free: unit-variance canonical directions
    rho, errc, fidc = cca_boundary(S, R, Z)
    cpe = np.corrcoef(rho, errc)[0, 1]
    cpf = np.corrcoef(rho, fidc)[0, 1]
    med = np.median(rho)
    sh, sp = rho >= med, rho < med
    print(f"  [CCA]   corr(rho, recovery-error)    = {cpe:+.3f}   (expect NEGATIVE)", flush=True)
    print(f"  [CCA]   corr(rho, recovery-fidelity) = {cpf:+.3f}   (expect POSITIVE)", flush=True)
    print(f"  [CCA]   error  SHARED (rho>={med:.2f}): {errc[sh].mean():.4f}", flush=True)
    print(f"  [CCA]   error  SPECIFIC (rho<{med:.2f}): {errc[sp].mean():.4f}", flush=True)
    print(f"  [CCA]   specific/shared error ratio = {errc[sp].mean()/max(errc[sh].mean(),1e-6):.2f}x  "
          f"=> {'BOUNDARY HOLDS (confound-free)' if cpe < -0.3 else 'WEAK/ARTIFACT (was energy confound)'}", flush=True)

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.7))
    o = np.argsort(r2)
    a1.scatter(r2[r2 >= np.median(r2)], err[r2 >= np.median(r2)], s=8, c="#2E8B57", alpha=0.5)
    a1.scatter(r2[r2 < np.median(r2)], err[r2 < np.median(r2)], s=8, c="#C0392B", alpha=0.5)
    a1.plot(r2[o], np.poly1d(np.polyfit(r2, err, 1))(r2[o]), "k--", lw=1.2)
    a1.set_xlabel("SAR-predictability ($R^2$)"); a1.set_ylabel("recovery error (norm.)")
    a1.set_title(f"naive (per-dim)  r={np.corrcoef(r2,err)[0,1]:+.2f}", fontsize=8.5)
    oc = np.argsort(rho)
    a2.scatter(rho[sh], fidc[sh], s=8, c="#2E8B57", alpha=0.5, label="SAR-shared")
    a2.scatter(rho[sp], fidc[sp], s=8, c="#C0392B", alpha=0.5, label="optical-specific")
    a2.plot(rho[oc], np.poly1d(np.polyfit(rho, fidc, 1))(rho[oc]), "k--", lw=1.2)
    a2.set_xlabel("SAR canonical correlation $\\rho$"); a2.set_ylabel("recovery fidelity")
    a2.set_title(f"CCA (confound-free)  r={cpf:+.2f}", fontsize=8.5)
    a2.legend(fontsize=7.5, frameon=False, loc="lower right")
    for a in (a1, a2):
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.suptitle("Recovery reconstructs the SAR-shared subspace but fabricates the optical-specific one",
                 fontsize=9, y=1.03)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
