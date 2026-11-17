"""Tile ISPRS Potsdam into the EarthM3 layout so EarthM3DALoader can read it.

Input  (datasets/potsdam_proc/raw):
  rgb/2_Ortho_RGB/top_potsdam_<x>_<y>_RGB.tif          6000x6000x3 uint8
  ndsm/1_DSM_normalisation/dsm_potsdam_<xx>_<yy>_normalized_lastools.jpg  6000x6000 uint8
  labels/top_potsdam_<x>_<y>_label.tif                 6000x6000x3 uint8 (RGB-coded)

Output (datasets/potsdam_proc/patches/{train,test}):
  images/SAR/<id>.tif   nDSM, 1ch uint8   (the "SAR-like" reliable modality, channel 0)
  images/RGB/<id>.tif   RGB,  3ch uint8
  masks/<id>.tif        1ch uint8, values 1..6  (EarthM3 does mask-1 -> classes 0..5)

Potsdam 6-class colormap -> stored value (class+1):
  Impervious(255,255,255)=1  Building(0,0,255)=2  LowVeg(0,255,255)=3
  Tree(0,255,0)=4  Car(255,255,0)=5  Clutter(255,0,0)=6
Unmapped colors -> 0 (EarthM3 -> -1 ignore).

Official ISPRS split: 14 test tiles, 24 train tiles.
Also accumulates per-channel mean/std over TRAIN for metadata/Potsdam.py.
"""
import argparse
import glob
import os
import re
import numpy as np
from skimage.io import imread
import tifffile

_DS = "/home/yue/experiments/earthmiss_baseline/unified_robust_exp/datasets/potsdam_proc"
RAW = f"{_DS}/raw"
OUT = f"{_DS}/patches"

COLORMAP = {  # RGB -> stored value (class+1)
    (255, 255, 255): 1,
    (0, 0, 255): 2,
    (0, 255, 255): 3,
    (0, 255, 0): 4,
    (255, 255, 0): 5,
    (255, 0, 0): 6,
}
TEST_TILES = {(2, 13), (2, 14), (3, 13), (3, 14), (4, 13), (4, 14), (4, 15),
              (5, 13), (5, 14), (5, 15), (6, 13), (6, 14), (6, 15), (7, 13)}


def tid(path):
    a, b = re.findall(r'(\d+)_(\d+)', os.path.basename(path))[0]
    return (int(a), int(b))


def decode_label(lab):
    out = np.zeros(lab.shape[:2], dtype=np.uint8)
    for (r, g, b), v in COLORMAP.items():
        out[(lab[..., 0] == r) & (lab[..., 1] == g) & (lab[..., 2] == b)] = v
    return out


def starts(full, tile):
    s = list(range(0, full - tile + 1, tile))
    if s[-1] != full - tile:
        s.append(full - tile)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="process only N tiles (smoke)")
    args = ap.parse_args()

    rgb = {tid(f): f for f in glob.glob(f"{RAW}/rgb/2_Ortho_RGB/*.tif")}
    dsm = {tid(f): f for f in glob.glob(f"{RAW}/ndsm/1_DSM_normalisation/*normalized_lastools.jpg")}
    lab = {tid(f): f for f in glob.glob(f"{RAW}/labels/*_label.tif")}
    ids = sorted(set(rgb) & set(dsm) & set(lab))
    print(f"matched tiles: {len(ids)}  (rgb {len(rgb)} dsm {len(dsm)} lab {len(lab)})", flush=True)
    if args.limit:
        ids = ids[:args.limit]

    ch_sum = np.zeros(4); ch_sqs = np.zeros(4); ch_n = 0
    counts = {"train": 0, "test": 0}
    for tile_id in ids:
        split = "test" if tile_id in TEST_TILES else "train"
        for sub in ("images/SAR", "images/RGB", "masks"):
            os.makedirs(f"{OUT}/{split}/{sub}", exist_ok=True)
        R = imread(rgb[tile_id])                 # HxWx3
        D = imread(dsm[tile_id])                 # HxW (nDSM jpg can be off-by-1 vs RGB!)
        if D.ndim == 3:
            D = D[..., 0]
        M = decode_label(imread(lab[tile_id]))   # HxW in 0..6
        # nDSM jpg / RGB tif / label tif can differ by a pixel -> crop to common size
        H = min(R.shape[0], D.shape[0], M.shape[0])
        W = min(R.shape[1], D.shape[1], M.shape[1])
        R, D, M = R[:H, :W], D[:H, :W], M[:H, :W]
        for r in starts(H, args.tile):
            for c in starts(W, args.tile):
                rr, cc = slice(r, r + args.tile), slice(c, c + args.tile)
                d, rgbp, m = D[rr, cc], R[rr, cc], M[rr, cc]
                name = f"top_{tile_id[0]}_{tile_id[1]}_{r}_{c}"
                tifffile.imwrite(f"{OUT}/{split}/images/SAR/{name}.tif", d)
                tifffile.imwrite(f"{OUT}/{split}/images/RGB/{name}.tif", rgbp)
                tifffile.imwrite(f"{OUT}/{split}/masks/{name}.tif", m)
                counts[split] += 1
                if split == "train":
                    px = np.concatenate([d[..., None], rgbp], axis=2).reshape(-1, 4).astype(np.float64)
                    ch_sum += px.sum(0); ch_sqs += (px ** 2).sum(0); ch_n += px.shape[0]
        print(f"  {tile_id} -> {split}", flush=True)

    print(f"patches: train={counts['train']} test={counts['test']}", flush=True)
    if ch_n:
        mean = ch_sum / ch_n
        std = np.sqrt(np.maximum(ch_sqs / ch_n - mean ** 2, 1e-6))
        print("CHANNEL STATS (nDSM, R, G, B):", flush=True)
        print(f"  sar_mean_v = [{mean[0]:.4f}]", flush=True)
        print(f"  sar_std_v  = [{std[0]:.4f}]", flush=True)
        print(f"  rgb_mean_v = [{mean[1]:.4f}, {mean[2]:.4f}, {mean[3]:.4f}]", flush=True)
        print(f"  rgb_std_v  = [{std[1]:.4f}, {std[2]:.4f}, {std[3]:.4f}]", flush=True)


if __name__ == "__main__":
    main()
