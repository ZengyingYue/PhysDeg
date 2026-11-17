"""Core ablation: available-modality evidence only (no recovered evidence)."""
from copy import deepcopy

from configs.baseline.CoDER_whu_rebound_core import config as _base

config = deepcopy(_base)
p = config["model"]["params"]
p["zero_recovered_evidence"] = True
p["loss"]["lam_ce_missing"] = 0.0
p["loss"]["lam_evi_missing"] = 0.0
p["loss"]["lam_recon"] = 0.0

