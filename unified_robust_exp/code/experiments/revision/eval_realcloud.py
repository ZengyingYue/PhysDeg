"""Real-cloud evaluation on SEN12MS-CR (winter, held-out ROIs).

For every test location the model sees (a) SAR + cloud-free Sentinel-2,
(b) SAR + the cloudy Sentinel-2 acquisition of the same location, and (c) SAR
only.  Reported:
  * label mIoU against the MODIS-derived labels (coarse, 500 m: secondary),
  * consistency of the cloudy prediction with the model's own cloud-free
    prediction (label-free, primary): pixel agreement and mIoU, overall,
    inside and outside the cloud mask, and per cloud-cover bin.
Calibrated models (Final.pth from evaluate.py) are used when present.

    python experiments/revision/eval_realcloud.py --config revision.s12cr_v4 \
        --run-dir ../runs/revision_v1/s12cr_v4/seed_2333 --seed 2333
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
REPO = CODE_ROOT.parents[1]
sys.path.insert(0, str(CODE_ROOT))

from ever.core.config import import_config  # noqa: E402
from experiments.revision.evaluate import load_model, loader, sha256  # noqa: E402
from module.baseline.CoDER import CoDER  # noqa: E402

BINS = [(0.0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.01)]


def miou(conf):
    inter = np.diag(conf).astype(float)
    union = conf.sum(0) + conf.sum(1) - inter
    ok = union > 0
    return float((inter[ok] / union[ok]).mean()) if ok.any() else float("nan")


def confusion(ref, pred, k, valid=None):
    if valid is not None:
        ref, pred = ref[valid], pred[valid]
    return torch.bincount(ref.flatten() * k + pred.flatten(), minlength=k * k).reshape(k, k).cpu().numpy()


def check_dataset_tag(tag):
    """Results of another season must not land in the winter result paths (S12CR_ROOT selects the data)."""
    root = os.environ.get("S12CR_ROOT", "").rstrip("/")
    other = bool(root) and "winter" not in root
    if (other and (not tag or not root.endswith(tag))) or (tag and not other):
        raise SystemExit(f"--dataset-tag '{tag}' does not match S12CR_ROOT '{root}'")


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--labels", choices=("modis", "wc"), default="modis",
                    help="'wc': 10 m ESA WorldCover labels (label_wc.npy, tools/prep_worldcover_s12cr.py)")
    ap.add_argument("--checkpoint", default="", help="file in --run-dir (default: Final.pth if present, else Best.pth)")
    ap.add_argument("--no-fallback", action="store_true",
                    help="disable the image-level fallback (plain / masked fusion only)")
    ap.add_argument("--split", choices=("test", "val"), default="test",
                    help="'val': validation ROIs (method selection; results_realcloud/<tag>__val)")
    ap.add_argument("--dataset-tag", default="", help="spring|summer: with S12CR_ROOT, separate result dir")
    args = ap.parse_args()
    check_dataset_tag(args.dataset_tag)
    config = import_config(args.config)
    k = int(config["model"]["params"]["num_classes"])
    run = Path(args.run_dir)
    ckpt = run / args.checkpoint if args.checkpoint else \
        run / "Final.pth" if (run / "Final.pth").exists() else run / "Best.pth"
    model = load_model(config, ckpt)
    is_coder = isinstance(model, CoDER)
    if args.no_fallback and is_coder:
        model.reliability_fallback = False
    keys = ("heldout_test", "heldout_cloudy") if args.split == "test" else ("val_clear", "val_cloudy")
    clear_dl = loader(config, keys[0], 8)
    cloudy_dl = loader(config, keys[1], 8)
    ds = clear_dl.dataset
    wc = np.load(os.path.join(ds.dir, "label_wc.npy"), mmap_mode="r") if args.labels == "wc" else None

    acc = {name: np.zeros((k, k), np.int64) for name in ("label_clear", "label_cloudy", "label_sar")}
    cons = {name: np.zeros((k, k), np.int64) for name in ("all", "in_cloud", "out_cloud", "sar_all")}
    agree = {name: [0, 0] for name in ("all", "in_cloud", "out_cloud", "sar_all")}
    per_bin = {i: {"cons": np.zeros((k, k), np.int64), "label": np.zeros((k, k), np.int64),
                   "label_clear": np.zeros((k, k), np.int64), "n": 0, "fallback": 0} for i in range(len(BINS))}
    routes = np.zeros(3, np.int64)
    index = 0
    for (xc, tc), (xk, _) in zip(clear_dl, cloudy_dl):
        b = xc.shape[0]
        y = tc["cls"].cuda().long()
        if wc is not None:  # same row order as ids.txt; 255 = ignore
            y = torch.from_numpy(np.asarray(wc[index:index + b]).astype(np.int64)).cuda()
            y[y == 255] = -1
        valid = y != -1
        cloud = torch.from_numpy(np.stack([ds.cloud_mask(index + j) for j in range(b)])).cuda().bool()
        index += b
        model.infer_mode = "fuse"
        p_clear = model(xc.cuda(), tc).argmax(1)
        p_cloudy = model(xk.cuda(), tc).argmax(1)
        route = model.last_route.cpu().numpy() if is_coder and model.last_route is not None else np.zeros(b, int)
        routes += np.bincount(route, minlength=3)
        model.infer_mode = "sar"
        p_sar = model(xc.cuda(), tc).argmax(1)
        acc["label_clear"] += confusion(y, p_clear, k, valid)
        acc["label_cloudy"] += confusion(y, p_cloudy, k, valid)
        acc["label_sar"] += confusion(y, p_sar, k, valid)
        for name, sel, pred in (("all", None, p_cloudy), ("in_cloud", cloud, p_cloudy),
                                ("out_cloud", ~cloud, p_cloudy), ("sar_all", None, p_sar)):
            cons[name] += confusion(p_clear, pred, k, sel)
            m = torch.ones_like(cloud) if sel is None else sel
            agree[name][0] += int((pred[m] == p_clear[m]).sum())
            agree[name][1] += int(m.sum())
        cover = cloud.flatten(1).float().mean(1).cpu().numpy()
        for j in range(b):
            i = next(i for i, (lo, hi) in enumerate(BINS) if lo <= cover[j] < hi)
            per_bin[i]["cons"] += confusion(p_clear[j], p_cloudy[j], k)
            per_bin[i]["label"] += confusion(y[j], p_cloudy[j], k, valid[j])
            per_bin[i]["label_clear"] += confusion(y[j], p_clear[j], k, valid[j])
            per_bin[i]["n"] += 1
            per_bin[i]["fallback"] += int(route[j] == 1)
    result = {
        "config": args.config, "seed": args.seed, "labels": args.labels, "no_fallback": args.no_fallback,
        "dataset": args.dataset_tag or "winter",
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256(ckpt), "evaluator_sha256": sha256(Path(__file__)),
        "test_patches": index,
        "label_miou": {n: miou(c) for n, c in acc.items()},
        "consistency_miou": {n: miou(c) for n, c in cons.items()},
        "consistency_agreement": {n: a / max(t, 1) for n, (a, t) in agree.items()},
        "route_fraction_cloudy": (routes / max(routes.sum(), 1)).tolist(),
        "per_cloud_bin": {f"{lo:.2f}-{hi:.2f}": {"patches": v["n"], "consistency_miou": miou(v["cons"]),
                                                   "label_miou_cloudy": miou(v["label"]),
                                                   "label_miou_clear": miou(v["label_clear"]),
                                                   "sar_fallback": v["fallback"] / max(v["n"], 1)}
                          for (lo, hi), v in zip(BINS, per_bin.values())},
    }
    tag = args.config.split(".")[-1] + ("_nofallback" if args.no_fallback else "")
    tag += "_wc" if args.labels == "wc" else ""
    tag += f"_{args.dataset_tag}" if args.dataset_tag else ""
    tag += "__val" if args.split == "val" else ""
    out = REPO / "revision_tgrs" / "results_realcloud" / tag / f"seed_{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k2: result[k2] for k2 in ("label_miou", "consistency_miou",
                                                "consistency_agreement", "per_cloud_bin")}, indent=1))


if __name__ == "__main__":
    main()
