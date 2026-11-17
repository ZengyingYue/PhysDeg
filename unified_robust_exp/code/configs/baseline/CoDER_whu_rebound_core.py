"""WHU core-ablation reference: deterministic shared recovery + bounded evidence.

This is the 0.5728 ReBound r1-aug architecture/recipe.  It preserves the paper's
deployed contract: softmax heads produce segmentation labels and the discounted
evidential/Dempster path produces uncertainty.  All core-ablation configs inherit
from this file and change one factor.
"""
from copy import deepcopy

from configs.baseline.CoDER_whu_moar_ml_r1_aug import config as _base

config = deepcopy(_base)
p = config["model"]["params"]
p["missing_readout"] = "head"
p["recovery_method"] = "det"
p["recovery_discount"] = True
p["observed_reliability_enabled"] = True
p["recovered_reliability_enabled"] = True
p["recovery_uncertainty_enabled"] = True
p["dempster_enabled"] = True
p["zero_recovered_evidence"] = False
