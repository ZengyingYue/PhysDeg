"""10 m ESA WorldCover labels for SEN12MS-CR patches (evaluation only).

The SEN12MS labels used for training are MODIS IGBP at 500 m.  For a sharper
evaluation of real-cloud robustness this re-projects ESA WorldCover 2020 v100
(10 m; Zanaga et al. 2021, CC-BY 4.0) onto every patch's own 10 m UTM grid
(nearest neighbour; patch CRS/transform read from the raw Sentinel-1 GeoTIFF) and
maps it to the same 8 DFC2020 classes:

  10 tree cover -> forest     20 shrubland -> shrubland   30 grassland -> grassland
  40 cropland   -> cropland   50 built-up  -> urban       60 bare / sparse -> barren
  80 water      -> water      90 herbaceous wetland, 95 mangroves -> wetland
  70 snow/ice, 100 moss/lichen, 0 no data -> 255 (ignore; no DFC2020 class)

Writes <dataset>/<split>/label_wc.npy (uint8 [N,256,256], row order = ids.txt) and
label_wc.json (tile list + SHA-256, class histogram, pixel agreement with the MODIS labels).
WorldCover tiles (3x3 degree COGs) are read from --tiles.

    python tools/prep_worldcover_s12cr.py --dataset ../datasets/sen12mscr_winter --splits val test
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject, transform_bounds

REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "datasets_raw" / "SEN12MS-CR"
WC_TO_DFC = np.full(256, 255, np.uint8)
for wc, dfc in {10: 0, 20: 1, 30: 2, 40: 4, 50: 5, 60: 6, 80: 7, 90: 3, 95: 3}.items():
    WC_TO_DFC[wc] = dfc
TILES = None


def tile_name(lat, lon):
    return f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}"


def s1_path(season, pid):
    roi, patch = re.match(r"(\d+)_p(\d+)$", pid).groups()
    return RAW / f"{season}_s1" / f"s1_{roi}" / f"{season}_s1_{roi}_p{patch}.tif"


def label_patch(args):
    season, pid = args
    with rasterio.open(s1_path(season, pid)) as d:
        crs, transform, shape = d.crs, d.transform, d.shape
        w, s, e, n = transform_bounds(crs, "EPSG:4326", *d.bounds)
    dst = np.zeros(shape, np.uint8)  # 0 = WorldCover no data
    names = {tile_name(3 * math.floor(la / 3), 3 * math.floor(lo / 3)) for la in (s, n) for lo in (w, e)}
    used = []
    for name in sorted(names):
        path = TILES / f"ESA_WorldCover_10m_2020_v100_{name}_Map.tif"
        if not path.exists():
            raise FileNotFoundError(f"{pid}: WorldCover tile {name} missing")
        part = np.zeros(shape, np.uint8)
        with rasterio.open(path) as src:
            reproject(rasterio.band(src, 1), part, dst_transform=transform, dst_crs=crs,
                      resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
        dst = np.where(dst == 0, part, dst)
        used.append(name)
    return WC_TO_DFC[dst], used


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    global TILES
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="prepared SEN12MS-CR dir (ids.txt, label.npy per split)")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--season", default="ROIs2017_winter")
    ap.add_argument("--tiles", default="/home/yue/data_nvme/worldcover")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    TILES = Path(args.tiles)
    for split in args.splits:
        d = Path(args.dataset) / split
        ids = (d / "ids.txt").read_text().split()
        out = np.lib.format.open_memmap(d / "label_wc.npy", "w+", np.uint8, (len(ids), 256, 256))
        tiles = set()
        with ProcessPoolExecutor(args.workers) as pool:
            for i, (lab, used) in enumerate(pool.map(label_patch, [(args.season, p) for p in ids], chunksize=16)):
                out[i] = lab
                tiles.update(used)
        out.flush()
        modis = np.load(d / "label.npy", mmap_mode="r")
        both = (out != 255) & (modis != 255)
        info = {
            "source": "ESA WorldCover 2020 v100, 10 m (CC-BY 4.0), nearest-neighbour onto each patch grid",
            "mapping": {"10": 0, "20": 1, "30": 2, "40": 4, "50": 5, "60": 6, "80": 7, "90": 3, "95": 3,
                        "70,100,0": 255},
            "patches": len(ids),
            "tiles": {t: sha256(TILES / f"ESA_WorldCover_10m_2020_v100_{t}_Map.tif") for t in sorted(tiles)},
            "class_hist": np.bincount(np.asarray(out).reshape(-1), minlength=256)[list(range(8)) + [255]].tolist(),
            "ignore_fraction": float((np.asarray(out) == 255).mean()),
            "agreement_with_modis": float((np.asarray(out)[both] == np.asarray(modis)[both]).mean()),
            "label_wc_sha256": sha256(d / "label_wc.npy"),
        }
        (d / "label_wc.json").write_text(json.dumps(info, indent=1))
        print(split, json.dumps({k: v for k, v in info.items() if k != "tiles"}), flush=True)


if __name__ == "__main__":
    main()
