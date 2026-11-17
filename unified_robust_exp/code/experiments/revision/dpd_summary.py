"""DPD screening summary: M, R_seen, R_held per config from evaluate.py results.

    python experiments/revision/dpd_summary.py --split val whu_moddrop whu_moddrop_dpd ...

``--split val`` reads ``results/<cfg>__test`` (validation), ``--split test``
reads ``results/<cfg>`` (held-out test).  M is the mean of clean, missing
optical and missing SAR mIoU; R_seen / R_held average severities 3-5 of the
corruption types seen / never seen by the DPD training degradations.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[4] / "revision_tgrs" / "results"
SEEN = ("haze", "noise", "blur", "cloud")
HELD = ("dark", "stripe", "localnoise")
PHYS_OPT = ("phys_haze", "phys_cloudshadow")
PHYS_SAR = ("sar_calib", "sar_noisefloor", "sar_rfi")


def scores(cond: dict) -> dict:
    def miou(name):
        r = cond[name]
        return 100 * r.get("always_fuse", r)["miou"]  # deployed fused head, no gate

    def mean_types(types, prefix="optical"):
        vals = [miou(f"{prefix}_{t}@{s}") for t in types for s in (3, 4, 5)
                if f"{prefix}_{t}@{s}" in cond]
        return st.mean(vals) if vals else float("nan")

    out = {k: miou(k) for k in ("clean", "missing_optical", "missing_sar")}
    out["M"] = st.mean(out.values())
    out["R_seen"] = mean_types(SEEN)
    out["R_held"] = mean_types(HELD)
    out["speckle"] = mean_types(("speckle",), "sar")
    return out


def phys_scores(cond: dict) -> dict:
    """Physical suite (``--suite phys`` results): per-kind and grouped means at severities 3-5."""
    def mean_kind(kind):
        return st.mean(100 * cond[f"{kind}@{s}"]["miou"] for s in (3, 4, 5))

    out = {"clean": 100 * cond["clean"]["miou"]}
    out.update({k: mean_kind(k) for k in PHYS_OPT + PHYS_SAR + ("coreg",)})
    out["R_phys_opt"] = st.mean(out[k] for k in PHYS_OPT)
    out["R_phys_sar"] = st.mean(out[k] for k in PHYS_SAR)
    out["C"] = out["coreg"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--phys", action="store_true", help="physical suite (results/<cfg>__test_phys)")
    ap.add_argument("configs", nargs="+")
    args = ap.parse_args()
    keys = ("clean", "missing_optical", "missing_sar", "M", "R_seen", "R_held", "speckle")
    score = scores
    if args.phys:
        keys = ("clean",) + PHYS_OPT + PHYS_SAR + ("R_phys_opt", "R_phys_sar", "C")
        score = phys_scores
    print("config".ljust(28) + "".join(k.replace("phys_", "").replace("sar_", "s_")[:11].rjust(12)
                                       for k in keys) + "  seeds")
    for cfg in args.configs:
        folder = RESULTS / (cfg + ("__test" if args.split == "val" else "") + ("_phys" if args.phys else ""))
        runs = [json.loads(p.with_suffix(".json").read_text())
                for p in sorted(folder.glob("seed_*.done"))]
        if not runs:
            print(cfg.ljust(28) + "  (no finished evaluation)")
            continue
        rows = [score(r["conditions"]) for r in runs]
        cells = []
        for k in keys:
            v = [r[k] for r in rows]
            cells.append(f"{st.mean(v):6.2f}" + (f"±{st.stdev(v):4.2f}" if len(v) > 1 else "     "))
        print(cfg.ljust(28) + "".join(c.rjust(12) for c in cells) + f"  {[r['seed'] for r in runs]}")


if __name__ == "__main__":
    main()
