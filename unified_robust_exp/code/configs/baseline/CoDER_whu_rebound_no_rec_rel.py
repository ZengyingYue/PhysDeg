"""Core ablation: recovered evidence gets beta_obs but no extra gamma_rec."""
from copy import deepcopy

from configs.baseline.CoDER_whu_rebound_core import config as _base

config = deepcopy(_base)
config["model"]["params"]["recovered_reliability_enabled"] = False

