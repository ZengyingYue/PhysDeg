"""Tile OpenEarthMap-SAR (DFC25 Track 1) into the EarthM3 layout.

Input  (/mnt/e/experiment/missing_modality/datasets/OpenEarthMap-SAR/train):
  rgb_images/TrainArea_NNN.tif   1024x1024x3 uint8   optical
  sar_images/TrainArea_NNN.tif   1024x1024   uint8   SAR (grayscale)
  labels/TrainArea_NNN.tif       1024x1024   uint8   0=ignore, 1..8 = 8 classes

Output (datasets/oem_sar_proc/patches/{train,heldout}):
  images/SAR/<id>.tif   SAR,  1ch uint8   (the reliable modality, channel 0)
  images/RGB/<id>.tif   RGB,  3ch uint8
  masks/<id>.tif        1ch uint8, values 0..8  (EarthM3 does mask-1 ->
                        0 -> -1 ignore, 1..8 -> 0..7 == 8 classes)

Labels are stored AS-IS (already integer-coded 0..8) -- unlike Potsdam, which
needed an RGB-colormap decode + class+1.

Split: the official val/test are SAR-only (the missing-modality benchmark), so
they carry no RGB for the degradation ladder. We carve a seed-42, SOURCE-IMAGE
-level 90/10 split out of the labelled train set -> train + heldout, both with
real RGB+SAR+labels. heldout is the eval set for clean / RGB-ladder / SAR-only.

Tiling 1024 -> 512 yields 4 patches per source image (2x2). Also accumulates
per-channel mean/std over the TRAIN patches for metadata/OEM_SAR.py.
"""
import argparse
import glob
import os
import numpy as np
from skimage.io import imread
import tifffile

RAW = "/mnt/e/experiment/missing_modality/datasets/OpenEarthMap-SAR/train"
OUT = "/home/yue/experiments/earthmiss_baseline/unified_robust_exp/datasets/oem_sar_proc/patches"
HELD_FRAC = 0.10
SPLIT_SEED = 42


def starts(full, tile):
    s = list(range(0, full - tile + 1, tile))
    if s and s[-1] != full - tile:
        s.append(full - tile)
    return s or [0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="process only N source imgs (smoke)")
    args = ap.parse_args()

    rgb = {os.path.basename(f): f for f in glob.glob(f"{RAW}/rgb_images/*.tif")}
    sar = {os.path.basename(f): f for f in glob.glob(f"{RAW}/sar_images/*.tif")}
    lab = {os.path.basename(f): f for f in glob.glob(f"{RAW}/labels/*.tif")}
    ids = sorted(set(rgb) & set(sar) & set(lab))
    print(f"matched imgs: {len(ids)}  (rgb {len(rgb)} sar {len(sar)} lab {len(lab)})", flush=True)
    if args.limit:
        ids = ids[:args.limit]

    # seed-42 source-image-level 90/10 split
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(ids))
    n_held = max(1, round(HELD_FRAC * len(ids)))
    held = {ids[i] for i in perm[:n_held]}
    print(f"split: train={len(ids) - n_held}  heldout={n_held}  (seed {SPLIT_SEED})", flush=True)

    ch_sum = np.zeros(4); ch_sqs = np.zeros(4); ch_n = 0
    counts = {"train": 0, "heldout": 0}
    uniq = set()
    for k, name in enumerate(ids):
        split = "heldout" if name in held else "train"
        for sub in ("images/SAR", "images/RGB", "masks"):
            os.makedirs(f"{OUT}/{split}/{sub}", exist_ok=True)
        R = imread(rgb[name])                       # HxWx3
        S = imread(sar[name])                        # HxW (or HxWxC)
        if S.ndim == 3:
            S = S[..., 0]
        M = imread(lab[name])                        # HxW, 0..8
        if M.ndim == 3:
            M = M[..., 0]
        H = min(R.shape[0], S.shape[0], M.shape[0])
        W = min(R.shape[1], S.shape[1], M.shape[1])
        R, S, M = R[:H, :W], S[:H, :W], M[:H, :W]
        uniq.update(np.unique(M).tolist())
        stem = os.path.splitext(name)[0]
        for r in starts(H, args.tile):
            for c in starts(W, args.tile):
                rr, cc = slice(r, r + args.tile), slice(c, c + args.tile)
                s, rgbp, m = S[rr, cc], R[rr, cc], M[rr, cc]
                pid = f"{stem}_{r}_{c}"
                tifffile.imwrite(f"{OUT}/{split}/images/SAR/{pid}.tif", s)
                tifffile.imwrite(f"{OUT}/{split}/images/RGB/{pid}.tif", rgbp)
                tifffile.imwrite(f"{OUT}/{split}/masks/{pid}.tif", m)
                counts[split] += 1
                if split == "train":
                    px = np.concatenate([s[..., None], rgbp], axis=2).reshape(-1, 4).astype(np.float64)
                    ch_sum += px.sum(0); ch_sqs += (px ** 2).sum(0); ch_n += px.shape[0]
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(ids)} ...", flush=True)

    print(f"patches: train={counts['train']} heldout={counts['heldout']}", flush=True)
    print(f"label values seen: {sorted(uniq)}", flush=True)
    if ch_n:
        mean = ch_sum / ch_n
        std = np.sqrt(np.maximum(ch_sqs / ch_n - mean ** 2, 1e-6))
        print("CHANNEL STATS (SAR, R, G, B):", flush=True)
        print(f"  sar_mean_v = [{mean[0]:.4f}]", flush=True)
        print(f"  sar_std_v  = [{std[0]:.4f}]", flush=True)
        print(f"  rgb_mean_v = [{mean[1]:.4f}, {mean[2]:.4f}, {mean[3]:.4f}]", flush=True)
        print(f"  rgb_std_v  = [{std[1]:.4f}, {std[2]:.4f}, {std[3]:.4f}]", flush=True)


if __name__ == "__main__":
    main()
