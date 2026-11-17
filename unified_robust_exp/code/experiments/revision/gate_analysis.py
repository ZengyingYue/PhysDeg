"""Offline evaluation of fallback criteria from gate_collect.py outputs.

Every threshold is a quantile of the score on CLEAN VALIDATION images of the
same model (target false-fallback rate q on clean data).  The criterion is
chosen on the validation split's corrupted images; the test split is only
reported.  'oracle' picks, per image, the route with the most correct pixels
(uses labels; an upper bound, never deployable).

    python experiments/revision/gate_analysis.py --run whu_rebound/seed_2333
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
ROOT = REPO / "revision_tgrs" / "gate_analysis"
ROUTES = ("fused", "sar", "opt")
DESIGN = ("haze", "noise", "blur", "speckle")
HELD_OUT = ("cloud", "dark")


def load(run: str, split: str) -> dict:
    data = {p.stem: dict(np.load(p)) for p in sorted((ROOT / run / split).glob("*.npz"))}
    # Model-free raw-pixel statistics (pixel_stats_collect.py), aligned by image.
    pix = ROOT / "pixel_stats" / split
    for cond, d in data.items():
        f = pix / f"{cond}.npz"
        if f.exists():
            extra = dict(np.load(f))
            if len(extra["dev_o"]) == len(d["beta_o"]):
                d.update(extra)
    return data


def miou(conf: np.ndarray) -> float:
    inter = np.diag(conf).astype(float)
    union = conf.sum(0) + conf.sum(1) - inter
    present = union > 0
    return float((inter[present] / union[present]).mean())


def thresholds(clean: dict, q: float) -> dict:
    """Low-side thresholds (score below -> unreliable) and high-side ones."""
    low = {k: float(np.quantile(clean[k], q)) for k in ("beta_o", "beta_s", "msp_o", "msp_s")}
    high = {k: float(np.quantile(clean[k], 1 - q))
            for k in ("gain_s", "gain_o", "dev_o", "dev_s") if k in clean}
    return {**low, **high}


def add_derived(d: dict) -> dict:
    d = dict(d)
    d["gain_s"] = d["msp_s"] - d["msp_f"]
    d["gain_o"] = d["msp_o"] - d["msp_f"]
    return d


def route(d: dict, t: dict, rule: str) -> np.ndarray:
    n = len(d["beta_o"])
    r = np.zeros(n, dtype=int)  # 0 fused, 1 SAR head, 2 optical head
    if rule == "always_fuse":
        return r
    if rule == "oracle":
        correct = np.stack([np.trace(d[f"conf_{x}"], axis1=1, axis2=2) for x in ROUTES], 1)
        return correct.argmax(1)
    bad = {
        "beta": (d["beta_o"] < t["beta_o"], d["beta_s"] < t["beta_s"]),
        "conf": (d["msp_o"] < t["msp_o"], d["msp_s"] < t["msp_s"]),
        "gain": (d["gain_s"] > t["gain_s"], d["gain_o"] > t["gain_o"]),
    }
    if "dev_o" in d:
        bad["pix"] = (d["dev_o"] > t["dev_o"], d["dev_s"] > t["dev_s"])
    if rule in bad:
        bad_o, bad_s = bad[rule]
    elif rule == "beta_and_conf":
        bad_o = bad["beta"][0] & bad["conf"][0]
        bad_s = bad["beta"][1] & bad["conf"][1]
    elif rule == "hybrid":
        # Per-sensor detector: covariance reliability for optical, head
        # confidence for SAR (the whitened SAR covariance is speckle-invariant).
        bad_o, bad_s = bad["beta"][0], bad["conf"][1]
    elif rule == "pix_or_conf":
        bad_o = bad["pix"][0] | bad["conf"][0]
        bad_s = bad["pix"][1] | bad["conf"][1]
    elif rule == "beta_or_conf":
        bad_o = bad["beta"][0] | bad["conf"][0]
        bad_s = bad["beta"][1] | bad["conf"][1]
    else:
        raise ValueError(rule)
    r[bad_o & ~bad_s] = 1
    r[bad_s & ~bad_o] = 2
    return r


RULES = ("always_fuse", "beta", "conf", "gain", "beta_and_conf", "beta_or_conf", "hybrid",
         "pix", "pix_or_conf", "oracle")


def evaluate(data: dict, t: dict) -> dict:
    out = {}
    has_pix = all("dev_o" in d for d in data.values())
    for rule in RULES:
        if rule.startswith("pix") and not has_pix:
            continue
        per = {}
        for cond, d in data.items():
            d = add_derived(d)
            r = route(d, t, rule)
            conf = sum(d[f"conf_{ROUTES[i]}"][r == i].sum(0) for i in range(3))
            per[cond] = {"miou": miou(conf), "sar_route": float((r == 1).mean()),
                         "opt_route": float((r == 2).mean())}
        out[rule] = per
    return out


def summarise(res: dict) -> dict:
    s = {}
    for rule, per in res.items():
        def mean(kinds):
            v = [x["miou"] for c, x in per.items() if any(f"_{k}@" in c for k in kinds)]
            return float(np.mean(v)) if v else float("nan")
        s[rule] = {"clean": per["clean"]["miou"],
                   "clean_fallback": per["clean"]["sar_route"] + per["clean"]["opt_route"],
                   "design_mean": mean(DESIGN), "held_out_mean": mean(HELD_OUT)}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="e.g. whu_rebound/seed_2333")
    ap.add_argument("--q", type=float, default=0.01)
    ap.add_argument("--sweep", action="store_true", help="q sensitivity on validation only")
    args = ap.parse_args()
    val, test = load(args.run, "val"), load(args.run, "test")
    if args.sweep:
        for q in (0.005, 0.01, 0.02, 0.05):
            summ = summarise(evaluate(val, thresholds(add_derived(val["clean"]), q)))
            print(f"q={q}: " + "  ".join(
                f"{r} {100*v['design_mean']:.2f}/{100*v['clean_fallback']:.1f}%"
                for r, v in summ.items() if r in ("beta", "beta_or_conf", "hybrid")))
        return
    t = thresholds(add_derived(val["clean"]), args.q)
    report = {"thresholds": t, "q": args.q,
              "val": evaluate(val, t), "test": evaluate(test, t) if "clean" in test else {}}
    report["val_summary"] = summarise(report["val"])
    if report["test"]:
        report["test_summary"] = summarise(report["test"])
    (ROOT / args.run / "gate_report.json").write_text(json.dumps(report, indent=1))
    for split in ("val", "test"):
        if f"{split}_summary" not in report:
            continue
        print(f"== {split} (mIoU %; clean fallback %)")
        for rule, v in report[f"{split}_summary"].items():
            print(f"  {rule:14s} clean {100*v['clean']:.2f}  fallback {100*v['clean_fallback']:.1f}"
                  f"  design {100*v['design_mean']:.2f}  held-out {100*v['held_out_mean']:.2f}")


if __name__ == "__main__":
    main()
