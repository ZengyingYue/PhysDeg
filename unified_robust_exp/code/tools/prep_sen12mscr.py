"""Prepare SEN12MS-CR (winter) for real-cloud evaluation.

Sources (verified with the official SHA-512 checksums):
  SEN12MS-CR  ROIs2017_winter_{s1,s2,s2_cloudy}   (Ebel et al., TGRS 2020)
  SEN12MS     ROIs2017_winter_lc                  (MODIS-derived land cover, 500 m)

For every patch present in all four products this writes, per split,
memory-mappable arrays under unified_robust_exp/datasets/sen12mscr_winter/:
  s1.npy      float16 [N,2,256,256]  VV, VH backscatter in dB (NaN -> -30)
  rgb.npy     uint16  [N,3,256,256]  cloud-free Sentinel-2 B4,B3,B2
  rgb_cloudy  uint16  [N,3,256,256]  cloudy Sentinel-2 B4,B3,B2 (same location)
  label.npy   uint8   [N,256,256]    IGBP -> DFC2020 8 classes, 255 = ignore
  cloud.npy   uint8   [N,256,256]    1 where the cloudy image is brighter than the
                                     clear one by > 0.06 reflectance in the visible
                                     mean (analysis only: stratification by cloud)
  ids.txt     "<roi>_p<patch>" per row
Split: whole ROIs, permutation seed 42: 19 train / 3 val / 5 test ROIs.
Also writes stats.json (training-split channel means / stds) and manifest.json.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "datasets_raw"
OUT = REPO / "unified_robust_exp" / "datasets" / "sen12mscr_winter"
RGB = [3, 2, 1]  # Sentinel-2 13-band order B1,B2,B3,B4,... -> B4,B3,B2
# IGBP (SEN12MS lc layer 1) -> DFC2020 simplified classes (as in the DFC2020 baseline
# and GFM-Bench): forest, shrubland, grassland, wetland, cropland, urban, barren, water.
IGBP_TO_DFC = np.array([255, 0, 0, 0, 0, 0, 1, 1, 255, 255, 2, 3, 4, 5, 4, 255, 6, 7], dtype=np.uint8)
CLASSES = ["forest", "shrubland", "grassland", "wetland", "cropland", "urban", "barren", "water"]


def index():
    def ids(pattern):
        out = {}
        for f in glob.glob(str(RAW / pattern)):
            m = re.search(r"_(\d+)_p(\d+)\.tif$", f)
            out[(int(m.group(1)), int(m.group(2)))] = f
        return out
    s1 = ids("SEN12MS-CR/ROIs2017_winter_s1/*/*.tif")
    s2 = ids("SEN12MS-CR/ROIs2017_winter_s2/*/*.tif")
    cl = ids("SEN12MS-CR/ROIs2017_winter_s2_cloudy/*/*.tif")
    lc = ids("SEN12MS_lc/ROIs2017_winter/*/*.tif")
    keys = sorted(set(s1) & set(s2) & set(cl) & set(lc))
    return [(k, s1[k], s2[k], cl[k], lc[k]) for k in keys]


def load(item):
    _, f1, f2, fc, fl = item
    with rasterio.open(f1) as d:
        s1 = np.nan_to_num(d.read().astype(np.float32), nan=-30.0)
    with rasterio.open(f2) as d:
        rgb = d.read([b + 1 for b in RGB]).astype(np.uint16)
    with rasterio.open(fc) as d:
        cloudy = d.read([b + 1 for b in RGB]).astype(np.uint16)
    with rasterio.open(fl) as d:
        igbp = d.read(1)
    label = IGBP_TO_DFC[np.clip(igbp, 0, len(IGBP_TO_DFC) - 1)]
    diff = cloudy.astype(np.float32).mean(0) - rgb.astype(np.float32).mean(0)
    cloud = (diff > 600).astype(np.uint8)  # 0.06 reflectance
    return s1.astype(np.float16), np.minimum(rgb, 10000), np.minimum(cloudy, 10000), label, cloud


def main():
    items = index()
    rois = sorted({k[0] for k, *_ in items})
    perm = np.random.RandomState(42).permutation(len(rois))
    owner = {}
    for i, j in enumerate(perm):
        owner[rois[j]] = "train" if i < 19 else ("val" if i < 22 else "test")
    manifest = {"rois": {s: sorted(r for r in rois if owner[r] == s) for s in ("train", "val", "test")}}
    stats = None
    for split in ("train", "val", "test"):
        part = [it for it in items if owner[it[0][0]] == split]
        d = OUT / split
        d.mkdir(parents=True, exist_ok=True)
        n = len(part)
        arrays = {
            "s1": np.lib.format.open_memmap(d / "s1.npy", "w+", np.float16, (n, 2, 256, 256)),
            "rgb": np.lib.format.open_memmap(d / "rgb.npy", "w+", np.uint16, (n, 3, 256, 256)),
            "rgb_cloudy": np.lib.format.open_memmap(d / "rgb_cloudy.npy", "w+", np.uint16, (n, 3, 256, 256)),
            "label": np.lib.format.open_memmap(d / "label.npy", "w+", np.uint8, (n, 256, 256)),
            "cloud": np.lib.format.open_memmap(d / "cloud.npy", "w+", np.uint8, (n, 256, 256)),
        }
        with ProcessPoolExecutor(12) as pool:
            for i, out in enumerate(pool.map(load, part, chunksize=32)):
                for name, arr in zip(("s1", "rgb", "rgb_cloudy", "label", "cloud"), out):
                    arrays[name][i] = arr
        for a in arrays.values():
            a.flush()
        ids = [f"{k[0]}_p{k[1]}" for k, *_ in part]
        (d / "ids.txt").write_text("\n".join(ids) + "\n")
        cover = arrays["cloud"].reshape(n, -1).mean(1)
        manifest[split] = {
            "patches": n,
            "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "cloud_cover_quantiles": np.quantile(cover, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist(),
            "label_hist": np.bincount(arrays["label"].reshape(-1), minlength=256)[list(range(8)) + [255]].tolist(),
        }
        if split == "train":
            sub = np.linspace(0, n - 1, 2000).astype(int)
            s1 = arrays["s1"][sub].astype(np.float32)
            rgb = arrays["rgb"][sub].astype(np.float32)
            stats = {"sar_mean": s1.mean((0, 2, 3)).tolist(), "sar_std": s1.std((0, 2, 3)).tolist(),
                     "rgb_mean": rgb.mean((0, 2, 3)).tolist(), "rgb_std": rgb.std((0, 2, 3)).tolist()}
        print(split, manifest[split], flush=True)
    manifest["classes"] = CLASSES
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("stats", stats)


if __name__ == "__main__":
    main()
