"""True complete feature-recovery baseline for the central recovery table.

Unlike ``CoDER_whu_rebound_fullrec`` (which only gives shared recovered
evidence beta_rec=1), this model predicts the entire missing-modality content
and modality-specific feature pyramids.  The reconstructed features are then
treated as observed inputs by the ordinary MoAR fused segmentation head.
"""
from copy import deepcopy

from configs.baseline.CoDER_whu_rebound_core import config as _base

config = deepcopy(_base)
p = config["model"]["params"]
p["complete_recovery_mode"] = "full_feature"
p["complete_feature_bottleneck_ratio"] = 8
p["recovery_discount"] = False
p["recovered_reliability_enabled"] = False
p["recovery_uncertainty_enabled"] = False
p["zero_recovered_evidence"] = False
