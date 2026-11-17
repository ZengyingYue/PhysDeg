"""Materialise the revision protocol-v1 splits and write auditable manifests.

For every dataset the train / val / test membership is written as an ordered
ID list plus a SHA-256 of that list, so every run can prove which samples it
trained on, selected on, and was tested on.

- WHU-OPT-SAR ``pad_v1``: PAD seed-42 test (1400) unchanged; 560 validation
  patches carved from the 5600 train patches with seed 4242.
- WHU-OPT-SAR ``region_v1``: whole source images 70 / 10 / 20 (seed 42).
- EarthMiss: the official city split (7 train / 3 val / 3 test cities).
- OpenEarthMap-SAR: the existing seed-42 source-image heldout set stays the
  test set; a validation set with the same number of source images is carved
  from the train source images (seed 4242).  Materialised as a symlink tree
  ``patches_v1/{train,val,test}`` so the EarthM3 loader needs no change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

REPO = CODE_ROOT.parents[1]
WHU_ROOT = REPO / "datasets_raw" / "whu-opt-sar"
EM_ROOT = REPO / "datasets_raw" / "EarthMiss"
OEM_ROOT = REPO / "unified_robust_exp" / "datasets" / "oem_sar_proc"
OUT = REPO / "revision_tgrs" / "splits"

EM_SPLIT = {
    "train": ["America-Eugene", "America-Louisville", "French-Paris",
              "Morocco-Casablanca", "Nanjing", "Netherlands-Rotterdam", "Singapore"],
    "val": ["Australia-PortHedland", "America-Pake", "Russian-Engels"],
    "test": ["America-NewYork", "Japan-Hakodate", "Peru-Callao"],
}


def digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def write(name: str, split: dict[str, list[str]], note: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    for part, ids in split.items():
        (OUT / f"{name}_{part}.txt").write_text("\n".join(ids) + "\n")
    all_ids = [i for ids in split.values() for i in ids]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError(f"{name}: a sample appears in more than one split")
    return {
        "note": note,
        "counts": {part: len(ids) for part, ids in split.items()},
        "sha256": {part: digest(ids) for part, ids in split.items()},
    }


def whu(split_mode: str) -> dict:
    from data.WHU import split_indices
    kwargs = dict(val_count=560) if split_mode == "patch" else dict(split_mode="region")
    opt_files, patches, idx = split_indices(str(WHU_ROOT), **kwargs)

    def pid(i):
        fi, r, c = patches[i]
        return f"{Path(opt_files[fi]).stem}_{r}_{c}"

    return {part: [pid(i) for i in arr] for part, arr in idx.items()}


def earthmiss() -> dict:
    split = {}
    for part, cities in EM_SPLIT.items():
        ids = []
        for city in cities:
            ids += [f"{city}/{m.stem}" for m in sorted((EM_ROOT / city / "masks").glob("*.tif"))]
        split[part] = ids
    return split


def oem(materialise: bool) -> dict:
    src = OEM_ROOT / "patches"
    train_ids = sorted(p.stem for p in (src / "train" / "masks").glob("*.tif"))
    test_ids = sorted(p.stem for p in (src / "heldout" / "masks").glob("*.tif"))

    def source(pid):  # TrainArea_001_0_512 -> TrainArea_001
        return "_".join(pid.split("_")[:2])

    test_sources = sorted({source(p) for p in test_ids})
    train_sources = sorted({source(p) for p in train_ids})
    perm = np.random.RandomState(4242).permutation(len(train_sources))
    val_sources = {train_sources[i] for i in perm[:len(test_sources)]}
    split = {
        "train": [p for p in train_ids if source(p) not in val_sources],
        "val": [p for p in train_ids if source(p) in val_sources],
        "test": test_ids,
    }
    if materialise:
        dst = OEM_ROOT / "patches_v1"
        origin = {"train": "train", "val": "train", "test": "heldout"}
        for part, ids in split.items():
            for sub in ("images/SAR", "images/RGB", "masks"):
                (dst / part / sub).mkdir(parents=True, exist_ok=True)
                for pid in ids:
                    link = dst / part / sub / f"{pid}.tif"
                    if not link.exists():
                        os.symlink(src / origin[part] / sub / f"{pid}.tif", link)
    return split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-materialise", action="store_true")
    args = ap.parse_args()
    manifest = {
        "whu_pad_v1": write("whu_pad_v1", whu("patch"),
                            "PAD seed-42 1400 test; 560 val carved from train with seed 4242"),
        "whu_region_v1": write("whu_region_v1", whu("region"),
                               "source images 70/10/20, image permutation seed 42"),
        "earthmiss_v1": write("earthmiss_v1", earthmiss(), "official city split"),
        "oem_v1": write("oem_v1", oem(not args.no_materialise),
                        "seed-42 source-image heldout = test; equal-size source-image val "
                        "carved from train with seed 4242"),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v["counts"] for k, v in manifest.items()}, indent=1))


if __name__ == "__main__":
    main()
