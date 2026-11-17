"""Core ablation: fixed recovered-evidence discount without observed quality.

beta_obs=1 and beta_rec=gamma_rec=0.3, frozen throughout training.
"""
from copy import deepcopy

from configs.baseline.CoDER_whu_rebound_core import config as _base

config = deepcopy(_base)
p = config["model"]["params"]
p["observed_reliability_enabled"] = False
p["recovered_reliability_enabled"] = True
p["gamma_rec_init"] = 0.3
p["freeze_gamma_rec"] = True

