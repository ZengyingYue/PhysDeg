"""Decode WHU-OPT-SAR once into uint8 .npy files that loaders memory-map.

Every process then shares the OS page cache instead of holding its own
~10 GB decoded copy, and evaluation no longer decodes GeoTIFF windows.
Values are exactly those read by data/WHU.py (optical bands [3,2,1] -> R,G,B).
"""
import glob
import os
import sys

import numpy as np
import rasterio

root = sys.argv[1]
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
for of in sorted(glob.glob(os.path.join(root, "optical", "*.tif"))):
    stem = os.path.splitext(os.path.basename(of))[0]
    if os.path.exists(os.path.join(out, f"{stem}_lbl.npy")):
        continue
    with rasterio.open(of) as ds:
        rgb = np.ascontiguousarray(ds.read()[[2, 1, 0]])
    with rasterio.open(of.replace("/optical/", "/sar/")) as ds:
        sar = np.ascontiguousarray(ds.read(1))
    with rasterio.open(of.replace("/optical/", "/lbl/")) as ds:
        lbl = np.ascontiguousarray(ds.read(1))
    for name, arr in (("rgb", rgb), ("sar", sar), ("lbl", lbl)):
        np.save(os.path.join(out, f"{stem}_{name}.tmp.npy"), arr)
        os.replace(os.path.join(out, f"{stem}_{name}.tmp.npy"), os.path.join(out, f"{stem}_{name}.npy"))
print("done", len(glob.glob(os.path.join(out, "*_lbl.npy"))))
