"""Core ablation: rectified-flow recovery without its variance gate."""
from copy import deepcopy

from configs.baseline.CoDER_whu_rebound_flow import config as _base

config = deepcopy(_base)
config["model"]["params"]["recovery_uncertainty_enabled"] = False

