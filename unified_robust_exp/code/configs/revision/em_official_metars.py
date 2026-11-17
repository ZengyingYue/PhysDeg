"""Reproduction check (R3-4i): the official MetaRS configuration on EarthMiss.

configs/baseline/MetaRS.py and configs/base/EarthMiss.py are the released EarthMiss/MetaRS
code (github.com/Yi-Heng/EarthMiss; same city split, model, loss, AdamW 1e-4, poly, 15k
iterations, batch 8, RandomCrop 512 + flips/rot90).  Two changes only:
  * data paths point at the local copy of EarthMiss;
  * checkpoint selection uses the validation cities.  The released code evaluates and keeps
    Best.pth on the test cities; here the test cities are 'heldout_test' and are read once,
    after the recipe freeze (protocol-v1).
Target: MetaRS paper (Zhou et al., RSE 2026), SAR-only test mIoU 31.77 (train S+O, test S).
"""
import copy
import os

from configs.baseline.MetaRS import config as _official

_LOCAL = os.environ.get("EM_ROOT", "/media/yue/Data/earthmiss_baseline/datasets_raw/EarthMiss/")
_OFFICIAL_ROOT = "/mnt/e/experiment/missing_modality/datasets/EarthMiss/"


def _relocate(obj):
    if isinstance(obj, dict):
        return {k: _relocate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_relocate(v) for v in obj]
    if isinstance(obj, str) and obj.startswith(_OFFICIAL_ROOT):
        return _LOCAL + obj[len(_OFFICIAL_ROOT):]
    return obj


config = copy.deepcopy(_official)
_d = config["data"]
for _k in ("train", "val", "test", "mmr"):
    _d[_k]["params"] = dict(_relocate(dict(_d[_k]["params"])), transforms=_d[_k]["params"]["transforms"])
config["model"]["params"]["data"]["params"] = dict(
    _relocate(dict(config["model"]["params"]["data"]["params"])),
    transforms=config["model"]["params"]["data"]["params"]["transforms"])
_val = copy.deepcopy(_d["val"])
_val["params"]["distributed"] = True
config["data"] = dict(train=_d["train"], test=_val, heldout_test=_d["test"], mmr=_d["mmr"])
