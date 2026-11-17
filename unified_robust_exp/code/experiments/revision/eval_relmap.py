"""Does the physically supervised reliability head find real clouds?

For reliability-supervised models (ModalityDropoutFusionRel) on SEN12MS-CR:
the head was trained only on synthetic physical degradations; here its map
r_hat on the real cloudy acquisitions is compared with the dataset's real
cloud mask (label-free with respect to the segmentation task).

Reported (validation ROIs by default):
  * pixel AUROC of (1 - r_hat) for the real cloud mask (cloudy images),
  * mean r_hat inside / outside clouds (cloudy images) and on the clear images,
  * Spearman correlation of image-mean (1 - r_hat) with cloud cover.

    python experiments/revision/eval_relmap.py --config revision.s12cr_moddroprel_physopt25 \
        --run-dir ../runs/revision_v1/s12cr_moddroprel_physopt25/seed_2333 --seed 2333
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CODE_ROOT = Path(__file__).resolve().parents[2]
REPO = CODE_ROOT.parents[1]
sys.path.insert(0, str(CODE_ROOT))

from ever.core.config import import_config  # noqa: E402
from experiments.revision.eval_realcloud import check_dataset_tag  # noqa: E402
from experiments.revision.evaluate import load_model, loader, sha256  # noqa: E402

NBINS = 1000


def auroc(pos_hist, neg_hist):
    """AUROC from score histograms (higher score = positive)."""
    pos, neg = pos_hist[::-1].astype(float), neg_hist[::-1].astype(float)  # descending score
    tp, fp = np.cumsum(pos), np.cumsum(neg)
    if tp[-1] == 0 or fp[-1] == 0:
        return float("nan")
    tpr = np.concatenate([[0], tp / tp[-1]])
    fpr = np.concatenate([[0], fp / fp[-1]])
    return float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--checkpoint", default="", help="file in --run-dir (default: Final.pth if present, else Best.pth)")
    ap.add_argument("--split", choices=("test", "val"), default="val")
    ap.add_argument("--dataset-tag", default="", help="spring|summer: with S12CR_ROOT, separate result dir")
    args = ap.parse_args()
    check_dataset_tag(args.dataset_tag)
    config = import_config(args.config)
    run = Path(args.run_dir)
    ckpt = run / args.checkpoint if args.checkpoint else \
        run / "Final.pth" if (run / "Final.pth").exists() else run / "Best.pth"
    model = load_model(config, ckpt)
    if not hasattr(model, "last_reliability"):
        raise SystemExit(f"{args.config}: no reliability head")
    model.infer_mode = "fuse"
    keys = ("heldout_test", "heldout_cloudy") if args.split == "test" else ("val_clear", "val_cloudy")
    clear_dl, cloudy_dl = loader(config, keys[0], 8), loader(config, keys[1], 8)
    ds = clear_dl.dataset

    edges = np.linspace(0, 1, NBINS + 1)
    pos, neg = np.zeros(NBINS, np.int64), np.zeros(NBINS, np.int64)
    sums = {"in_cloud": [0.0, 0], "out_cloud": [0.0, 0], "clear": [0.0, 0]}
    img_unrel, img_cover = [], []
    index = 0
    for (xc, tc), (xk, _) in zip(clear_dl, cloudy_dl):
        b = xc.shape[0]
        cloud = torch.from_numpy(np.stack([ds.cloud_mask(index + j) for j in range(b)])).cuda().bool()
        index += b
        model(xc.cuda(), tc)
        r_clear = F.interpolate(model.last_reliability.float(), size=cloud.shape[-2:],
                                mode="bilinear", align_corners=False)[:, 0]
        model(xk.cuda(), tc)
        r = F.interpolate(model.last_reliability.float(), size=cloud.shape[-2:],
                          mode="bilinear", align_corners=False)[:, 0]
        u = (1 - r).clamp(0, 1).cpu().numpy()
        c = cloud.cpu().numpy()
        pos += np.histogram(u[c], edges)[0]
        neg += np.histogram(u[~c], edges)[0]
        sums["in_cloud"][0] += float(r[cloud].sum()); sums["in_cloud"][1] += int(cloud.sum())
        sums["out_cloud"][0] += float(r[~cloud].sum()); sums["out_cloud"][1] += int((~cloud).sum())
        sums["clear"][0] += float(r_clear.sum()); sums["clear"][1] += r_clear.numel()
        img_unrel += u.reshape(b, -1).mean(1).tolist()
        img_cover += c.reshape(b, -1).mean(1).tolist()
    result = {
        "config": args.config, "seed": args.seed, "split": args.split, "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256(ckpt), "evaluator_sha256": sha256(Path(__file__)),
        "patches": index,
        "pixel_auroc_cloud": auroc(pos, neg),
        "mean_rel": {k: s / max(n, 1) for k, (s, n) in sums.items()},
        "spearman_unrel_vs_cover": spearman(np.array(img_unrel), np.array(img_cover)),
    }
    tag = args.config.split(".")[-1] + (f"_{args.dataset_tag}" if args.dataset_tag else "") + ("__val" if args.split == "val" else "")
    out = REPO / "revision_tgrs" / "results_relmap" / tag / f"seed_{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
