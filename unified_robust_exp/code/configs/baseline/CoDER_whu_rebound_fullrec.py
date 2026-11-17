"""Core ablation: unbounded full-weight recovered evidence.

Single change from CoDER_whu_rebound_core: beta_rec=1 in both training and eval.
"""
from copy import deepcopy

from configs.baseline.CoDER_whu_rebound_core import config as _base

config = deepcopy(_base)
config["model"]["params"]["recovery_discount"] = False

