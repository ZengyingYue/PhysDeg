"""Prepare a further SEN12MS-CR season for out-of-season real-cloud evaluation (no training).

Same products, band order, label mapping and cloud threshold as tools/prep_sen12mscr.py
(`load` is reused).  Models are trained on winter only, so this season keeps the winter
normalisation: stats.json is a copy of the winter file.

ROI rule (PHYS_PLAN 8), fixed before any prediction on these seasons:
  * ROIs whose lon/lat box intersects a winter training ROI are excluded (leakage guard).
  * The rest are permuted with seed 42; ROIs go to val until it holds >= --val-patches
    patches (winter val: 2106), all others to test.  Test is read once, after freezing.

Also records the ESA WorldCover tiles these patches would need (nothing is downloaded).

    python tools/prep_sen12mscr_season.py --season spring --out /home/yue/data_nvme/sen12mscr_spring
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_sen12mscr import CLASSES, RAW, load  # noqa: E402
from prep_worldcover_s12cr import tile_name  # noqa: E402

PREFIX = {"spring": "ROIs1158_spring", "summer": "ROIs1868_summer"}
WINTER = "ROIs2017_winter"


def scan(pattern):
    out = {}
    for f in glob.glob(str(RAW / pattern)):
        m = re.search(r"_(\d+)_p(\d+)\.tif$", f)
        out[(int(m.group(1)), int(m.group(2)))] = f
    return out


def index(prefix):
    s1, s2 = scan(f"SEN12MS-CR/{prefix}_s1/*/*.tif"), scan(f"SEN12MS-CR/{prefix}_s2/*/*.tif")
    cl, lc = scan(f"SEN12MS-CR/{prefix}_s2_cloudy/*/*.tif"), scan(f"SEN12MS_lc/{prefix}/*/*.tif")
    keys = sorted(set(s1) & set(s2) & set(cl) & set(lc))
    counts = {"s1": len(s1), "s2": len(s2), "s2_cloudy": len(cl), "lc": len(lc), "complete": len(keys)}
    return [(k, s1[k], s2[k], cl[k], lc[k]) for k in keys], counts


def lonlat(path):
    with rasterio.open(path) as d:
        return transform_bounds(d.crs, "EPSG:4326", *d.bounds)


def roi_boxes(keys_paths, workers):
    """{roi: (w, s, e, n)} and the WorldCover tiles touched by the patches."""
    boxes, tiles = {}, set()
    with ProcessPoolExecutor(workers) as pool:
        for (roi, _), (w, s, e, n) in zip((k for k, _ in keys_paths),
                                          pool.map(lonlat, [p for _, p in keys_paths], chunksize=64)):
            b = boxes.get(roi, (w, s, e, n))
            boxes[roi] = (min(b[0], w), min(b[1], s), max(b[2], e), max(b[3], n))
            for la in (s, n):
                for lo in (w, e):
                    tiles.add(tile_name(3 * math.floor(la / 3), 3 * math.floor(lo / 3)))
    return boxes, tiles


def intersects(a, b):
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True, choices=sorted(PREFIX))
    ap.add_argument("--out", required=True)
    ap.add_argument("--winter", default="/home/yue/data_nvme/sen12mscr_winter")
    ap.add_argument("--val-patches", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split-only", action="store_true", help="write the manifest, no arrays")
    args = ap.parse_args()
    out = Path(args.out)
    winter = json.load(open(Path(args.winter) / "manifest.json"))
    items, counts = index(PREFIX[args.season])
    print("products", counts, flush=True)
    per_roi = defaultdict(int)
    for it in items:
        per_roi[it[0][0]] += 1

    train_rois = set(winter["rois"]["train"])
    wfiles = [(k, p) for k, p in scan(f"SEN12MS-CR/{WINTER}_s1/*/*.tif").items() if k[0] in train_rois]
    wbox, _ = roi_boxes(wfiles, args.workers)
    boxes, tiles = roi_boxes([(it[0], it[1]) for it in items], args.workers)
    excluded = sorted(r for r, b in boxes.items() if any(intersects(b, w) for w in wbox.values()))

    rois = sorted(r for r in boxes if r not in excluded)
    perm = np.random.RandomState(42).permutation(len(rois))
    val, n_val = [], 0
    for j in perm:
        if n_val >= args.val_patches:
            break
        val.append(rois[j])
        n_val += per_roi[rois[j]]
    owner = {r: "val" if r in val else "test" for r in rois}
    have = {Path(p).name for p in glob.glob(str(Path("/home/yue/data_nvme/worldcover") / "*_Map.tif"))}
    need = sorted(tiles)
    manifest = {
        "season": args.season, "prefix": PREFIX[args.season], "products": counts,
        "rois": {s: sorted(r for r in rois if owner[r] == s) for s in ("val", "test")},
        "excluded_overlap_winter_train": excluded,
        "roi_lonlat_box": {str(r): [round(v, 4) for v in b] for r, b in sorted(boxes.items())},
        "worldcover_tiles_needed": need,
        "worldcover_tiles_missing": [t for t in need if f"ESA_WorldCover_10m_2020_v100_{t}_Map.tif" not in have],
        "stats_source": str(Path(args.winter) / "stats.json"), "classes": CLASSES,
    }
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(args.winter) / "stats.json", out / "stats.json")
    if not args.split_only:
        for split in ("val", "test"):
            part = [it for it in items if owner.get(it[0][0]) == split]
            d = out / split
            d.mkdir(parents=True, exist_ok=True)
            n = len(part)
            arrays = {
                "s1": np.lib.format.open_memmap(d / "s1.npy", "w+", np.float16, (n, 2, 256, 256)),
                "rgb": np.lib.format.open_memmap(d / "rgb.npy", "w+", np.uint16, (n, 3, 256, 256)),
                "rgb_cloudy": np.lib.format.open_memmap(d / "rgb_cloudy.npy", "w+", np.uint16, (n, 3, 256, 256)),
                "label": np.lib.format.open_memmap(d / "label.npy", "w+", np.uint8, (n, 256, 256)),
                "cloud": np.lib.format.open_memmap(d / "cloud.npy", "w+", np.uint8, (n, 256, 256)),
            }
            with ProcessPoolExecutor(args.workers) as pool:
                for i, row in enumerate(pool.map(load, part, chunksize=32)):
                    for name, arr in zip(("s1", "rgb", "rgb_cloudy", "label", "cloud"), row):
                        arrays[name][i] = arr
                    if i % 2000 == 0:
                        print(split, i, n, flush=True)
            for a in arrays.values():
                a.flush()
            ids = [f"{k[0]}_p{k[1]}" for k, *_ in part]
            (d / "ids.txt").write_text("\n".join(ids) + "\n")
            cover = np.array([arrays["cloud"][i].mean() for i in range(n)])
            manifest[split] = {
                "patches": n,
                "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                "cloud_cover_quantiles": np.quantile(cover, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist(),
                "label_hist": np.bincount(np.concatenate([arrays["label"][i].ravel() for i in range(0, n, 16)]),
                                          minlength=256)[list(range(8)) + [255]].tolist(),
            }
            print(split, manifest[split], flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("rois", "excluded_overlap_winter_train", "worldcover_tiles_missing")}))


if __name__ == "__main__":
    main()
