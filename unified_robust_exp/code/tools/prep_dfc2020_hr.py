"""DFC2020 with the official 10 m labels, as a clean high-quality-label test set
for models trained on SEN12MS-CR (same sensors: Sentinel-1 VV/VH, Sentinel-2).

Imagery: GFM-Bench mirror (96x96 tiles, 9 per patch, 3x3 with stride 80)
stitched back to 256x256 (overlaps verified identical).  Labels: `dfc_*`
(10 m, DFC2020 10-class codes) mapped to the 8 classes used for SEN12MS-CR:
forest, shrubland, grassland, wetland, cropland, urban, barren, water
(savanna and snow -> ignore 255).

Writes unified_robust_exp/datasets/dfc2020_hr/{validation,test}/ with the
array layout of tools/prep_sen12mscr.py (s1.npy, rgb.npy, label.npy, ids.txt).
'validation' = the 986 official validation patches, 'test' = the 5128 patches
of the official test scene.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "datasets_raw" / "DFC2020"
OUT = REPO / "unified_robust_exp" / "datasets" / "dfc2020_hr"
# DFC2020 codes 1..10 -> 8-class scheme (index 0 unused)
DFC_TO_8 = np.array([255, 0, 1, 255, 2, 3, 4, 5, 255, 6, 7], dtype=np.uint8)
RGB = [3, 2, 1]  # 13-band order B01..B12 (with B8A) -> B04, B03, B02


def stitch(paths, channels, dtype):
    out = np.zeros((256, 256, channels), dtype)
    for k, p in enumerate(paths):
        r, c = divmod(k, 3)
        out[80 * r:80 * r + 96, 80 * c:80 * c + 96] = tifffile.imread(p)
    return out


def tile_paths(roi, pid, kind):
    names = [f"ROIs0000_{roi}_{kind}_0_p{pid}_{k}.tiff" for k in range(9)]
    for split in ("train", "val", "test"):
        paths = [SRC / "DFC2020" / split / kind / n for n in names]
        if all(p.exists() for p in paths):
            return paths
    raise FileNotFoundError(f"{roi} p{pid} {kind}")


def main():
    manifest = {}
    for part, roi, folder in (("validation", "validation", "dfc_validation"), ("test", "test", "dfc_0")):
        labels = sorted(glob.glob(str(SRC / "official_labels_10m" / folder / "**" / "*.tif"), recursive=True),
                        key=lambda p: int(p.rsplit("_p", 1)[1][:-4]))
        n = len(labels)
        d = OUT / part
        d.mkdir(parents=True, exist_ok=True)
        s1 = np.lib.format.open_memmap(d / "s1.npy", "w+", np.float16, (n, 2, 256, 256))
        rgb = np.lib.format.open_memmap(d / "rgb.npy", "w+", np.uint16, (n, 3, 256, 256))
        lab = np.lib.format.open_memmap(d / "label.npy", "w+", np.uint8, (n, 256, 256))
        ids = []
        for i, f in enumerate(labels):
            pid = int(f.rsplit("_p", 1)[1][:-4])
            s1[i] = stitch(tile_paths(roi, pid, "s1"), 2, np.float32).transpose(2, 0, 1)
            s2 = stitch(tile_paths(roi, pid, "s2"), 13, np.float32)
            rgb[i] = np.clip(s2[..., RGB], 0, 10000).transpose(2, 0, 1).astype(np.uint16)
            lab[i] = DFC_TO_8[np.clip(tifffile.imread(f), 0, 10)]
            ids.append(f"p{pid}")
        for a in (s1, rgb, lab):
            a.flush()
        (d / "ids.txt").write_text("\n".join(ids) + "\n")
        manifest[part] = {"patches": n, "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                          "label_hist": np.bincount(lab.reshape(-1), minlength=256)[list(range(8)) + [255]].tolist()}
        print(part, manifest[part], flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
