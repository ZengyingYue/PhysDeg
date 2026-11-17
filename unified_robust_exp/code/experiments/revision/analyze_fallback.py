"""Analysis (validation only): how much do real clouds cost, and what can decision-level
SAR fallback recover?  SEN12MS-CR, any ModalityDropoutFusion-family model.

For every validation location the model predicts with (clear optical + SAR), (cloudy
optical + SAR) and (SAR only).  Decision rules combining the cloudy-fused and SAR-only
class probabilities are scored by label mIoU (MODIS labels, overall and per cloud-cover
bin) and by consistency with the model's own clear prediction (overall / in cloud):

  fuse, sar                  the two single predictions
  avg                        mean of the two probability maps (label-free)
  conf                       per pixel, the more confident of the two (label-free)
  mask / mask_dil            SAR inside the real cloud mask (dilated by 7 px) -- needs a cloud mask
  img50                      SAR for images with cloud cover > 50 % (needs a cloud mask)
  rel_blend / rel_thr / rel_img   reliability head r_hat (Rel models only): r*fuse + (1-r)*sar,
                             SAR where r < 0.5, SAR for images with mean r < 0.5
  oracle_img                 per image the better of fuse / sar by label accuracy (upper bound)

    python experiments/revision/analyze_fallback.py --config revision.s12cr_moddrop_gp25 \
        --run-dir ../runs/revision_v1/s12cr_moddrop_gp25/seed_2333 --seed 2333
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CODE_ROOT = Path(__file__).resolve().parents[2]
REPO = CODE_ROOT.parents[1]
sys.path.insert(0, str(CODE_ROOT))

from ever.core.config import import_config  # noqa: E402
from experiments.revision.evaluate import load_model, loader, sha256  # noqa: E402
from experiments.revision.eval_realcloud import BINS, check_dataset_tag, confusion, miou  # noqa: E402


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--labels", choices=("modis", "wc"), default="modis",
                    help="'wc': 10 m ESA WorldCover labels (label_wc.npy, tools/prep_worldcover_s12cr.py)")
    ap.add_argument("--checkpoint", default="", help="file in --run-dir (default: Final.pth if present, else Best.pth)")
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N batches (no output file)")
    ap.add_argument("--dataset-tag", default="", help="spring|summer: with S12CR_ROOT, separate result dir")
    args = ap.parse_args()
    check_dataset_tag(args.dataset_tag)
    config = import_config(args.config)
    k = int(config["model"]["params"]["num_classes"])
    run = Path(args.run_dir)
    ckpt = run / args.checkpoint if args.checkpoint else \
        run / "Final.pth" if (run / "Final.pth").exists() else run / "Best.pth"
    model = load_model(config, ckpt)
    has_rel = hasattr(model, "last_reliability")
    clear_dl, cloudy_dl = loader(config, "val_clear", 8), loader(config, "val_cloudy", 8)
    ds = clear_dl.dataset
    wc = np.load(os.path.join(ds.dir, "label_wc.npy"), mmap_mode="r") if args.labels == "wc" else None

    names = ["fuse", "sar", "avg", "conf", "mask", "mask_dil", "img50", "oracle_img"]
    if has_rel:
        names += ["rel_blend", "rel_thr", "rel_img"] + [f"rel_{m}{t}" for m in ("thr", "img") for t in (70, 80, 90, 95)]
    lab = {n: np.zeros((k, k), np.int64) for n in names + ["clear", "clear_avg", "clear_conf"] + (["clear_relblend"] + [f"clear_rel{m}{t}" for m in ("thr", "img") for t in (70, 80, 90, 95)] if has_rel else [])}
    lab_bin = {n: [np.zeros((k, k), np.int64) for _ in BINS] for n in names + ["clear"]}
    con = {n: np.zeros((k, k), np.int64) for n in names}
    con_in = {n: np.zeros((k, k), np.int64) for n in names}
    chosen_sar = {n: [0, 0] for n in names}
    index = 0
    import time
    t0 = time.time()
    for step, ((xc, tc), (xk, _)) in enumerate(zip(clear_dl, cloudy_dl)):
        if args.limit and step >= args.limit:
            print(f"{step} batches in {time.time() - t0:.1f}s", flush=True)
            return
        if step % 50 == 0:
            print(f"batch {step} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        b = xc.shape[0]
        y = tc["cls"].cuda().long()
        if wc is not None:  # same row order as ids.txt; 255 = ignore
            y = torch.from_numpy(np.asarray(wc[index:index + b]).astype(np.int64)).cuda()
            y[y == 255] = -1
        valid = y != -1
        cloud = torch.from_numpy(np.stack([ds.cloud_mask(index + j) for j in range(b)])).cuda().bool()
        index += b
        model.infer_mode = "fuse"
        p_clear = model(xc.cuda(), tc).float()
        p_fuse = model(xk.cuda(), tc).float()
        r = None
        if has_rel:
            r = F.interpolate(model.last_reliability.float(), size=y.shape[-2:], mode="bilinear",
                              align_corners=False).clamp(0, 1)
        model.infer_mode = "sar"
        p_sar = model(xk.cuda(), tc).float()
        ref = p_clear.argmax(1)
        a_f, a_s = p_fuse.argmax(1), p_sar.argmax(1)
        cover = cloud.flatten(1).float().mean(1)
        dil = F.max_pool2d(cloud[:, None].float(), 7, 1, 3)[:, 0].bool()
        acc_f = ((a_f == y) & valid).flatten(1).sum(1)
        acc_s = ((a_s == y) & valid).flatten(1).sum(1)
        use = {  # bool map: True = take the SAR-only prediction
            "fuse": torch.zeros_like(cloud), "sar": torch.ones_like(cloud),
            "conf": p_sar.amax(1) > p_fuse.amax(1),
            "mask": cloud, "mask_dil": dil,
            "img50": (cover > 0.5)[:, None, None].expand_as(cloud),
            "oracle_img": (acc_s > acc_f)[:, None, None].expand_as(cloud),
        }
        preds = {n: torch.where(m, a_s, a_f) for n, m in use.items()}
        preds["avg"] = (p_fuse + p_sar).argmax(1)
        if has_rel:
            preds["rel_blend"] = (r * p_fuse + (1 - r) * p_sar).argmax(1)
            use["rel_thr"] = r[:, 0] < 0.5
            use["rel_img"] = (r.flatten(1).mean(1) < 0.5)[:, None, None].expand_as(cloud)
            preds["rel_thr"] = torch.where(use["rel_thr"], a_s, a_f)
            preds["rel_img"] = torch.where(use["rel_img"], a_s, a_f)
            for t in (70, 80, 90, 95):  # thresholds on r_hat (pixel) / image-mean r_hat
                use[f"rel_thr{t}"] = r[:, 0] < t / 100
                preds[f"rel_thr{t}"] = torch.where(use[f"rel_thr{t}"], a_s, a_f)
                use[f"rel_img{t}"] = (r.flatten(1).mean(1) < t / 100)[:, None, None].expand_as(cloud)
                preds[f"rel_img{t}"] = torch.where(use[f"rel_img{t}"], a_s, a_f)
        for n in names:
            p = preds[n]
            lab[n] += confusion(y, p, k, valid)
            con[n] += confusion(ref, p, k)
            con_in[n] += confusion(ref, p, k, cloud)
            if n in use:
                chosen_sar[n][0] += int(use[n].sum()); chosen_sar[n][1] += use[n].numel()
        lab["clear"] += confusion(y, ref, k, valid)
        # the same combination rules on the clear acquisition (cost when there is no cloud)
        lab["clear_avg"] += confusion(y, (p_clear + p_sar).argmax(1), k, valid)
        lab["clear_conf"] += confusion(y, torch.where(p_sar.amax(1) > p_clear.amax(1), a_s, ref), k, valid)
        if has_rel:
            model.infer_mode = "fuse"
            model(xc.cuda(), tc)
            r_c = F.interpolate(model.last_reliability.float(), size=y.shape[-2:], mode="bilinear",
                                align_corners=False).clamp(0, 1)
            lab["clear_relblend"] += confusion(y, (r_c * p_clear + (1 - r_c) * p_sar).argmax(1), k, valid)
            for t in (70, 80, 90, 95):
                lab[f"clear_relthr{t}"] += confusion(y, torch.where(r_c[:, 0] < t / 100, a_s, ref), k, valid)
                m_img = (r_c.flatten(1).mean(1) < t / 100)[:, None, None].expand_as(ref)
                lab[f"clear_relimg{t}"] += confusion(y, torch.where(m_img, a_s, ref), k, valid)
        edges = torch.tensor([hi for _, hi in BINS[:-1]], device=cover.device)
        bin_px = torch.bucketize(cover, edges, right=True)[:, None, None].expand_as(y)[valid]
        nb = len(BINS)
        for n, p in list(preds.items()) + [("clear", ref)]:
            idx = (bin_px * k + y[valid]) * k + p[valid]
            cm = torch.bincount(idx, minlength=nb * k * k).reshape(nb, k, k).cpu().numpy()
            for i in range(nb):
                lab_bin[n][i] += cm[i]
    bins = [f"{lo:.2f}-{hi:.2f}" for lo, hi in BINS]
    result = {
        "config": args.config, "seed": args.seed, "labels": args.labels, "checkpoint": str(ckpt),
        "dataset": args.dataset_tag or "winter",
        "checkpoint_sha256": sha256(ckpt), "evaluator_sha256": sha256(Path(__file__)),
        "patches": index,
        "label_miou": {n: miou(c) for n, c in lab.items()},
        "label_miou_bin": {n: dict(zip(bins, map(miou, cs))) for n, cs in lab_bin.items()},
        "consistency_miou": {n: miou(c) for n, c in con.items()},
        "consistency_miou_in_cloud": {n: miou(c) for n, c in con_in.items()},
        "sar_pixel_fraction": {n: a / max(t, 1) for n, (a, t) in chosen_sar.items()},
    }
    out = REPO / "revision_tgrs" / "results_analysis" / (("fallback" if args.labels == "modis" else "fallback_wc") + (f"_{args.dataset_tag}" if args.dataset_tag else "")) / args.config.split(".")[-1] / f"seed_{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({n: [round(100 * result["label_miou"][n], 1), round(100 * result["consistency_miou"][n], 1),
                          round(100 * result["consistency_miou_in_cloud"][n], 1)] for n in names}))


if __name__ == "__main__":
    main()
