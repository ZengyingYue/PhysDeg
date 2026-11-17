"""Tile the OFFICIAL OpenEarthMap-SAR val split (SAR-only) for the SAR-only floor.

The official val/test are the purpose-built missing-modality benchmark: SAR only,
no RGB. We tile val 1024->512 into the EarthM3 layout with a ZERO RGB placeholder
so the existing 4-channel loader + eval_robust.py run unchanged. infer_mode='sar'
uses only channel 0 (SAR) -- the RGB placeholder is never read (this is exactly
how eval_robust.py's "RGB-missing (SAR-only)" floor is computed on heldout), so
the reported SAR-only mIoU is the genuine official-val number.

Input : /mnt/e/.../OpenEarthMap-SAR/val/{sar_images,labels}/ValArea_NNN.tif
Output: datasets/oem_sar_proc/patches/officialval/{images/SAR,images/RGB,masks}
Labels stored AS-IS (0..8); EarthM3 maps mask-1 -> 0 -> -1 ignore, 1..8 -> 0..7.
"""
import glob
import os
import numpy as np
from skimage.io import imread
import tifffile

VAL = "/mnt/e/experiment/missing_modality/datasets/OpenEarthMap-SAR/val"
OUT = "/home/yue/experiments/earthmiss_baseline/unified_robust_exp/datasets/oem_sar_proc/patches/officialval"
TILE = 512


def starts(full, tile):
    s = list(range(0, full - tile + 1, tile))
    if s and s[-1] != full - tile:
        s.append(full - tile)
    return s or [0]


def main():
    sar = {os.path.basename(f): f for f in glob.glob(f"{VAL}/sar_images/*.tif")}
    lab = {os.path.basename(f): f for f in glob.glob(f"{VAL}/labels/*.tif")}
    ids = sorted(set(sar) & set(lab))
    print(f"matched val imgs: {len(ids)}", flush=True)
    for sub in ("images/SAR", "images/RGB", "masks"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)

    n = 0
    for name in ids:
        S = imread(sar[name])
        if S.ndim == 3:
            S = S[..., 0]
        M = imread(lab[name])
        if M.ndim == 3:
            M = M[..., 0]
        H = min(S.shape[0], M.shape[0]); W = min(S.shape[1], M.shape[1])
        S, M = S[:H, :W], M[:H, :W]
        stem = os.path.splitext(name)[0]
        for r in starts(H, TILE):
            for c in starts(W, TILE):
                rr, cc = slice(r, r + TILE), slice(c, c + TILE)
                s, m = S[rr, cc], M[rr, cc]
                rgb0 = np.zeros((s.shape[0], s.shape[1], 3), dtype=np.uint8)  # RGB placeholder
                pid = f"{stem}_{r}_{c}"
                tifffile.imwrite(f"{OUT}/images/SAR/{pid}.tif", s)
                tifffile.imwrite(f"{OUT}/images/RGB/{pid}.tif", rgb0)
                tifffile.imwrite(f"{OUT}/masks/{pid}.tif", m)
                n += 1
    print(f"officialval patches: {n}", flush=True)


if __name__ == "__main__":
    main()
