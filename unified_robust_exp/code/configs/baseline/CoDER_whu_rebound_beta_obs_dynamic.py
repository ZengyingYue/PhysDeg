"""Input-adaptive beta_obs using a persisted clean covariance reference."""
from copy import deepcopy

from configs.baseline.CoDER_whu_rebound_beta_obs_calibrate import config as _base

config = deepcopy(_base)
config["model"]["params"]["observed_reliability_stats"] = (
    "/home/yue/experiment/earthmiss_baseline/unified_robust_exp/"
    "runs/beta_obs_covariance/reference_r1_aug.pth"
)

