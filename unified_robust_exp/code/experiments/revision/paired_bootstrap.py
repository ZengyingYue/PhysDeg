"""Paired bootstrap of mIoU differences between two methods (R3: significance, by ROI / city).

Input: results_analysis/perimg*/<config>/seed_<s>.npz from dump_perimage.py.  One resample of the
units is shared by both methods and all seeds, so the difference is paired.  mIoU is recomputed from
the resampled confusion sums.  Seeds are resampled too (hierarchical bootstrap; --fixed-seeds turns
this off): with three seeds the training variance dominates on SEN12MS-CR, and an interval over
images alone would describe the trained models, not the method.

Units: geographic groups (SEN12MS-CR ROI, EarthMiss city, WHU source image).  With fewer than
--min-groups groups a group bootstrap is meaningless, so images are resampled within each group;
such an interval covers image sampling only and does not generalise to new locations, and the
output says so ("unit": "image-within-group").

    python experiments/revision/paired_bootstrap.py --folder perimg_val --a s12cr_moddrop_gp25 \
        --b s12cr_moddrop_degaug25 --metric cloudy
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
SEEDS = (2333, 42, 1234)


def group_of(name: str) -> str:
    if m := re.match(r"(\d+)_p\d+$", name):  # SEN12MS-CR <roi>_p<patch>
        return m.group(1)
    if m := re.match(r"(.+)_\d+_\d+\.tif$", name):  # WHU <source image>_<row>_<col>.tif
        return m.group(1)
    if m := re.match(r"([A-Za-z]+(?:-[A-Za-z]+)?)", name):  # EarthMiss <Country-City>...
        return m.group(1)
    raise ValueError(name)


def miou(conf):
    """conf [..., k, k] -> mean IoU over classes with a non-empty union (as in the evaluators)."""
    inter = np.diagonal(conf, axis1=-2, axis2=-1).astype(float)
    union = conf.sum(-1) + conf.sum(-2) - inter
    ok = union > 0
    return np.where(ok, inter / np.maximum(union, 1), 0).sum(-1) / np.maximum(ok.sum(-1), 1)


def load(folder, name, metrics):
    out = {}
    for s in SEEDS:
        z = np.load(REPO / "revision_tgrs" / "results_analysis" / folder / name / f"seed_{s}.npz")
        out[s] = dict(names=z["names"], cm=[z[f"cm_{m}"].astype(np.int64) for m in metrics])
    return out


def score(sums):
    """sums: list (one per metric) of [..., k, k] -> mean mIoU over the metrics."""
    return np.mean([miou(c) for c in sums], axis=0)


def resample_weights(groups, rng, n, min_groups):
    """[n, images] multiplicities."""
    ids, inv = np.unique(groups, return_inverse=True)
    w = np.zeros((n, len(groups)), np.float32)
    if len(ids) >= min_groups:
        draw = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=n)  # [n, groups]
        return draw[:, inv].astype(np.float32), "group"
    for g in range(len(ids)):
        idx = np.flatnonzero(inv == g)
        w[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=n)
    return w, "image-within-group"


def compare(folder, a, b, metrics, n=10000, min_groups=8, seed=0, fixed_seeds=False):
    ra, rb = load(folder, a, metrics), load(folder, b, metrics)
    names = ra[SEEDS[0]]["names"]
    for r in (ra, rb):
        for s in SEEDS:
            assert (r[s]["names"] == names).all(), "image order differs"
    groups = np.array([group_of(str(x)) for x in names])
    rng = np.random.default_rng(seed)
    w, unit = resample_weights(groups, rng, n, min_groups)
    per_seed = np.zeros((n, len(SEEDS)))
    point = []
    for j, s in enumerate(SEEDS):
        ca, cb = ra[s]["cm"], rb[s]["cm"]
        k = ca[0].shape[-1]
        pa = score([c.sum(0) for c in ca])
        pb = score([c.sum(0) for c in cb])
        point.append(100 * (pa - pb))
        for lo in range(0, n, 500):
            ww = w[lo:lo + 500]
            sa = [(ww @ c.reshape(len(names), -1).astype(np.float32)).reshape(-1, k, k) for c in ca]
            sb = [(ww @ c.reshape(len(names), -1).astype(np.float32)).reshape(-1, k, k) for c in cb]
            per_seed[lo:lo + 500, j] = 100 * (score(sa) - score(sb))
    sw = np.ones((n, len(SEEDS))) if fixed_seeds else rng.multinomial(len(SEEDS), np.full(len(SEEDS), 1 / len(SEEDS)), size=n)
    diffs = (per_seed * sw).sum(1) / sw.sum(1)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"a": a, "b": b, "metrics": metrics, "folder": folder, "unit": unit, "seeds_resampled": not fixed_seeds, "groups": len(set(groups)),
            "images": len(names), "resamples": n, "diff_mean": float(np.mean(point)), "diff_per_seed": [float(x) for x in point],
            "ci95": [float(x) for x in np.percentile(diffs, (2.5, 97.5))], "p_two_sided": float(min(p, 1.0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--metric", nargs="+", required=True, help="cm_ keys; several are averaged (e.g. clean missing_optical missing_sar = M)")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--min-groups", type=int, default=8)
    ap.add_argument("--fixed-seeds", action="store_true", help="condition on the trained models (images only)")
    args = ap.parse_args()
    print(json.dumps(compare(args.folder, args.a, args.b, args.metric, args.resamples, args.min_groups,
                             fixed_seeds=args.fixed_seeds), indent=1))


if __name__ == "__main__":
    main()
