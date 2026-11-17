"""Core ablation: remove observed-modality covariance reliability beta_obs."""
from copy import deepcopy

from configs.baseline.CoDER_whu_rebound_core import config as _base

config = deepcopy(_base)
config["model"]["params"]["observed_reliability_enabled"] = False

