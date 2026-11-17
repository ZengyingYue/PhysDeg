"""Build clean covariance reference statistics for the 0.5728 ReBound model."""
from copy import deepcopy

from configs.baseline.CoDER_whu_rebound_core import config as _base

config = deepcopy(_base)
p = config["model"]["params"]
p["observed_reliability_mode"] = "covariance_deviation"
p["observed_reliability_stats"] = ""
p["obs_beta_floor"] = 0.1
p["obs_cov_mask_fraction"] = 0.1
p["obs_clean_quantile"] = 0.90
p["obs_scale_quantile"] = 0.99
p["obs_beta_tau"] = 1.0

