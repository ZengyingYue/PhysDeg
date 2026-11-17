"""Core extension: rectified-flow shared recovery with variance gating.

Compared with the deterministic core, this adds only the learned recovery
operator and beta_rec's exp(-sigma^2/tau) gate.  The paired no-gate config below
separates recovery quality from the uncertainty discount.
"""
from copy import deepcopy

from configs.baseline.CoDER_whu_rebound_core import config as _base

config = deepcopy(_base)
p = config["model"]["params"]
p["recovery_method"] = "flow"
p["use_flow_recovery"] = True
p["flow_tau"] = 0.5
p["flow_t_dim"] = 64
p["flow_n_steps"] = 4
p["flow_n_samples"] = 4
p["flow_sigma_detach"] = True
p["flow_spatial_sigma"] = True
p["reliability_source"] = "seed"
p["recovery_uncertainty_enabled"] = True
p["loss"]["lam_flow"] = 1.0
p["loss"]["lam_rel"] = 0.0

