"""ISPRS Potsdam (C2) — CoDER+MoAR on nDSM(1)+RGB(3), 6 classes.

Mirrors the validated EarthMiss MoAR config (baseline.CoDER_flow_cb_moar:
convnext_base + latent-flow recovery + MoAR/DMFIA fusion, EarthM3DALoader) and
only swaps in the tiled Potsdam data block + num_classes=6. nDSM occupies the
SAR slot (channel 0, the reliable height modality); RGB is channels 1-3 — so the
covariance-reliability / missing-modality / degradation machinery transfers
unchanged (degrade the optical stream, fall back on nDSM).

Patches: tools/prep_potsdam.py -> datasets/potsdam_proc/patches (3456 train /
2016 test, official 24/14 tile split). label_smoothing 0.05 (s4 recipe).
"""
from copy import deepcopy

from configs.baseline.CoDER_flow_cb_moar import config as _base
from configs.base.Potsdam import data as _pdata

config = deepcopy(_base)
config["data"] = _pdata
config["model"]["params"]["data"] = _pdata["mmr"]      # fix the embedded mmr loader
config["model"]["params"]["num_classes"] = 6
config["model"]["params"]["loss"]["class_weights"] = [1.0] * 6
config["model"]["params"]["loss"]["label_smoothing"] = 0.05
