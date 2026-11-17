"""Audit that each ReBound core-ablation config is a small diff from the core.

This catches accidental recipe/backbone/data changes before launching multi-hour
training runs.  It exits non-zero when a variant changes an unapproved key.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping

from ever.core.config import import_config


VARIANTS = {
    "baseline.CoDER_whu_complete_full_feature": {
        "model.params.complete_recovery_mode",
        "model.params.complete_feature_bottleneck_ratio",
        "model.params.recovery_discount",
        "model.params.recovered_reliability_enabled",
        "model.params.recovery_uncertainty_enabled",
    },
    "baseline.CoDER_whu_complete_full_image": {
        "model.params.complete_recovery_mode",
        "model.params.complete_image_width",
        "model.params.recovery_discount",
        "model.params.recovered_reliability_enabled",
        "model.params.recovery_uncertainty_enabled",
    },
    "baseline.CoDER_whu_rebound_fullrec": {
        "model.params.recovery_discount",
    },
    "baseline.CoDER_whu_rebound_no_recovery": {
        "model.params.zero_recovered_evidence",
        "model.params.loss.lam_ce_missing",
        "model.params.loss.lam_evi_missing",
        "model.params.loss.lam_recon",
    },
    "baseline.CoDER_whu_rebound_fixed_rec": {
        "model.params.observed_reliability_enabled",
        "model.params.freeze_gamma_rec",
    },
    "baseline.CoDER_whu_rebound_no_obs_rel": {
        "model.params.observed_reliability_enabled",
    },
    "baseline.CoDER_whu_rebound_no_rec_rel": {
        "model.params.recovered_reliability_enabled",
    },
    "baseline.CoDER_whu_rebound_no_dempster": {
        "model.params.dempster_enabled",
    },
    "baseline.CoDER_whu_rebound_flow": {
        "model.params.recovery_method",
        "model.params.use_flow_recovery",
        "model.params.flow_tau",
        "model.params.flow_t_dim",
        "model.params.flow_n_steps",
        "model.params.flow_n_samples",
        "model.params.flow_sigma_detach",
        "model.params.flow_spatial_sigma",
        "model.params.reliability_source",
        "model.params.loss.lam_flow",
        "model.params.loss.lam_rel",
    },
    "baseline.CoDER_whu_rebound_flow_no_gate": {
        "model.params.recovery_method",
        "model.params.use_flow_recovery",
        "model.params.flow_tau",
        "model.params.flow_t_dim",
        "model.params.flow_n_steps",
        "model.params.flow_n_samples",
        "model.params.flow_sigma_detach",
        "model.params.flow_spatial_sigma",
        "model.params.reliability_source",
        "model.params.recovery_uncertainty_enabled",
        "model.params.loss.lam_flow",
        "model.params.loss.lam_rel",
    },
}


def flatten(value, prefix=""):
    out = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(child, name))
    else:
        out[prefix] = value
    return out


def changed_keys(base, other):
    a, b = flatten(base), flatten(other)
    keys = set(a) | set(b)
    return {key for key in keys if repr(a.get(key)) != repr(b.get(key))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base", default="baseline.CoDER_whu_rebound_core"
    )
    args = parser.parse_args()
    base = import_config(args.base)
    failed = False
    for name, allowed in VARIANTS.items():
        changed = changed_keys(base, import_config(name))
        unexpected = changed - allowed
        missing = allowed - changed
        status = "OK" if not unexpected and not missing else "FAIL"
        print(f"[{status}] {name}")
        print(f"  changed: {sorted(changed)}")
        if unexpected:
            print(f"  unexpected: {sorted(unexpected)}")
        if missing:
            print(f"  expected-but-unchanged: {sorted(missing)}")
        failed |= bool(unexpected or missing)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
