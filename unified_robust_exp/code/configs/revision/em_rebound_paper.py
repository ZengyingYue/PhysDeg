"""Qualitative-figure retraining (Remote Sensing revision, 2026-10): the paper's
WHU ReBound model (baseline.CoDER_whu_rebound_core: Dempster fusion of discounted
evidence, availability-selected softmax heads, MoAR + DMFIA, multilevel head) with
only the dataset block swapped to em, trained under the revision protocol
(validation-split checkpoint selection, same schedule as the baselines)."""
import copy

from configs.baseline.CoDER_whu_rebound_core import config as _paper
from configs.revision.common import DATASETS, assemble, data_block

_ds = DATASETS["em"]
_model = copy.deepcopy(_paper["model"])
_p = _model["params"]
_p["num_classes"] = _ds["num_classes"]
_p["sar_in_channels"] = _ds.get("sar_channels", 1)
_p["infer_mode"] = "fuse"
_p["missing_modality_total_iterations"] = _ds["iters"]
_p["loss"]["class_weights"] = _ds["class_weights"]
_p["data"] = data_block("em")["mmr"]

config = assemble(_model, "em")
