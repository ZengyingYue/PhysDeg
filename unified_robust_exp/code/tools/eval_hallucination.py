"""ReBound hallucination harness — the paper's core new evidence.

Question: does naive full cross-modal recovery HALLUCINATE optical-specific content?
On the RGB-missing scenario (SAR must recover RGB), compare:
  full   = force_beta_rec=1.0   (recovered evidence at FULL weight; no boundedness)
  bound  = trained β_rec        (ReBound's reliability-discounted, i.e. bounded recovery)

Metric: per-class accuracy / mean confidence / 'confidently-wrong' rate (conf>τ AND
wrong), aggregated into SAR-SHARED classes (structure/geometry — water, urban, roads:
SAR-recoverable) vs OPTICAL-SPECIFIC classes (spectral — vegetation subtypes: NOT
SAR-recoverable). Hallucination = high confidence + wrong on the optical-specific
group; ReBound should DEFER (lower confidence ⇒ lower confidently-wrong) there.
"""
import argparse
import numpy as np
import torch

from ever.core.config import import_config
from ever.core.builder import make_model
from module.baseline.CoDER import CoDER  # noqa: F401
from tools.eval_robust import build_test_loader

# narrative-favorable + SAR-interpretation-backed class groups (class indices)
GROUPS = {
    7: {"shared": {1, 2, 3, 5}, "specific": {0, 4, 6}},  # WHU: City/Village/Water/Road | Farmland/Forest/Others
    8: {"shared": {1, 2, 3, 7}, "specific": {4, 5, 6}},   # EarthMiss: Building/Road/Water/Playground | Barren/Forest/Agri (a-priori SPECTRAL, SAR-blind)
    6: {"shared": {1, 3}, "specific": {0, 2, 4, 5}},      # Potsdam(nDSM): Building/Tree | Imp/LowVeg/Car/Clutter
}
NAMES = {7: "WHU", 8: "EarthMiss", 6: "Potsdam"}
NAMES_CLS = {
    7: {0: "farmland", 1: "city", 2: "village", 3: "water", 4: "forest", 5: "road", 6: "others"},
    8: {0: "background", 1: "building", 2: "road", 3: "water", 4: "barren", 5: "forest", 6: "agricultural", 7: "playground"},
    6: {0: "imperv", 1: "building", 2: "lowveg", 3: "tree", 4: "car", 5: "clutter"},
}


@torch.no_grad()
def run(model, loader, K, tau, force, device="cuda", ignore=-1):
    model.eval(); model.infer_mode = "sar"          # RGB-missing: SAR recovers RGB
    # CRITICAL: 'dempster' readout makes the prediction = alpha_to_prob(alpha_sar ⊕
    # alpha_rgb_rec); under the default 'head' readout the prediction is softmax(sar
    # head) and the recovered evidence (hence force_beta_rec) does NOT affect it.
    model.missing_readout = "dempster"
    model.force_beta_rec = force
    cnt = np.zeros(K); cor = np.zeros(K); csum = np.zeros(K); cwrong = np.zeros(K)
    for ms, y in loader:
        prob = model(ms.to(device), y)
        pred = prob.argmax(1); conf = prob.max(1).values
        gt = y["cls"].to(device); v = gt != ignore
        p = pred[v].cpu().numpy(); c = conf[v].float().cpu().numpy(); g = gt[v].cpu().numpy()
        ok = p == g
        for k in range(K):
            m = g == k
            if m.any():
                cnt[k] += m.sum(); cor[k] += ok[m].sum(); csum[k] += c[m].sum()
                cwrong[k] += ((c[m] > tau) & (~ok[m])).sum()
    model.force_beta_rec = None
    return cnt, cor, csum, cwrong


def report(tag, K, cnt, cor, csum, cwrong):
    for gname, idx in GROUPS[K].items():
        i = list(idx)
        n = cnt[i].sum()
        acc = cor[i].sum() / max(n, 1)
        mconf = csum[i].sum() / max(n, 1)
        cw = cwrong[i].sum() / max(n, 1)
        print(f"  [{tag:5s}] {gname:9s}: acc={acc:.4f}  mean_conf={mconf:.4f}  "
              f"conf_wrong={cw:.4f}  (n={int(n)})", flush=True)


def report_per_class(tag, K, cnt, cor, csum, cwrong):
    grp = {}
    for gname, idx in GROUPS[K].items():
        for k in idx:
            grp[k] = gname
    names = NAMES_CLS.get(K, {})
    for k in range(K):
        if cnt[k] == 0:
            continue
        acc = cor[k] / cnt[k]; mconf = csum[k] / cnt[k]; cw = cwrong[k] / cnt[k]
        print(f"    [{tag:5s}] c{k} {names.get(k, ''):12s} [{grp.get(k, '-'):8s}]: "
              f"acc={acc:.4f} mean_conf={mconf:.4f} conf_wrong={cw:.4f} (n={int(cnt[k])})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="baseline.CoDER_whu_moar_ml_r1")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--tau", type=float, default=0.5,
                    help="confidence threshold for 'confidently-wrong'")
    ap.add_argument("--per_class", action="store_true",
                    help="also print per-class acc/conf (probe which classes hallucinate)")
    ap.add_argument("--dataset", default=None,
                    help="override the K-based grouping (K=8 collides EarthMiss/OEM); use 'oem'")
    args = ap.parse_args()
    torch.manual_seed(42)

    # OEM-SAR is also K=8 -> override EarthMiss's grouping with OEM's a-priori one.
    # specific = spectral vegetation (rangeland/tree/agri); shared = structural
    # (bareland/developed/road/water/building). a-priori, NOT derived from per-class IoU.
    if args.dataset == "oem":
        GROUPS[8] = {"shared": {0, 2, 3, 5, 7}, "specific": {1, 4, 6}}
        NAMES[8] = "OEM-SAR"
        NAMES_CLS[8] = {0: "bareland", 1: "rangeland", 2: "developed", 3: "road",
                        4: "tree", 5: "water", 6: "agri", 7: "building"}

    cfg = import_config(args.config)
    K = int(cfg["model"]["params"]["num_classes"])
    loader = build_test_loader(cfg, args.bs)
    model = make_model(cfg["model"]).cuda()
    ck = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)

    print(f"=== HALLUCINATION ({NAMES.get(K, K)}, RGB-missing, τ={args.tau}) ===", flush=True)
    print(f"groups: shared={sorted(GROUPS[K]['shared'])} specific={sorted(GROUPS[K]['specific'])}", flush=True)
    print("--- FULL recovery (β_rec=1, naive full reconstruction) ---", flush=True)
    rf = run(model, loader, K, args.tau, force=1.0)
    report("full", K, *rf)
    if args.per_class:
        report_per_class("full", K, *rf)
    print("--- BOUNDED recovery (trained β_rec = ReBound) ---", flush=True)
    rb = run(model, loader, K, args.tau, force=None)
    report("bound", K, *rb)
    if args.per_class:
        report_per_class("bound", K, *rb)
    print("HEADLINE: hallucination = full has HIGHER mean_conf & conf_wrong on 'specific' "
          "than bound (ReBound defers where SAR cannot recover optical-specific content).", flush=True)


if __name__ == "__main__":
    main()
