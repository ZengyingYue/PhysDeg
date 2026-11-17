"""Same checkpoint and machinery as dynamic beta_obs, but force beta_obs=1."""
from copy import deepcopy

from configs.baseline.CoDER_whu_rebound_beta_obs_dynamic import config as _base

config = deepcopy(_base)
config["model"]["params"]["observed_reliability_enabled"] = False

