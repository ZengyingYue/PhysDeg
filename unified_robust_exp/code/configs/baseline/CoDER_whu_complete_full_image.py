"""True complete image-recovery baseline for the central recovery table.

The missing raw sensor image is reconstructed from the available modality,
passed through the normal missing-modality encoders, and treated as an observed
input by the ordinary MoAR fused segmentation head.  There is no reliability
discount on the reconstructed branch.
"""
from copy import deepcopy

from configs.baseline.CoDER_whu_rebound_core import config as _base

config = deepcopy(_base)
p = config["model"]["params"]
p["complete_recovery_mode"] = "full_image"
p["complete_image_width"] = 32
p["recovery_discount"] = False
p["recovered_reliability_enabled"] = False
p["recovery_uncertainty_enabled"] = False
p["zero_recovered_evidence"] = False
