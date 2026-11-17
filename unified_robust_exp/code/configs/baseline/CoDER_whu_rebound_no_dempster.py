"""Core ablation: additive evidence pooling without Dempster conflict."""
from copy import deepcopy

from configs.baseline.CoDER_whu_rebound_core import config as _base

config = deepcopy(_base)
config["model"]["params"]["dempster_enabled"] = False
