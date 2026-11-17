"""ReBound-v2 and its one-factor variants (recovery ladder, rule and fusion ablations).

``build(dataset, variant)`` returns a complete config.  Every variant differs
from ``rebound`` in exactly the fields listed in ``VARIANTS``.

rebound       shared-only recovery + learned discount + provenance-aware
              ('average') combination + reliability-gated readout, MoAR fused
              path with the Switch gate and a null expert, no DMFIA.
r0            no recovery: no recovered stream, no missing-branch losses.
r1            full-image recovery (pixel-space generator), no discount.
r2            full-feature recovery (shared + specific pyramids), no discount.
r3            shared-only recovery without the discount (beta_rec = 1).
dempster      recovered evidence combined by Dempster's rule (AAAI version).
moar_legacy   MoAR with the legacy renormalised top-1 gate (no router gradient).
fuse_simple   MetaRS SCSE fusion instead of MoAR on the fused path.
v3a           fused head reads shared + specific (projected) features.
v3b           no evidential (EDL) losses.
v3c           v3a + v3b.
v3d           v3c without instance whitening in the shared encoder.
v4            v3c + spatial modality dropout (training) + masked fusion from local
              pixel statistics (inference) + pixel-statistics/confidence gate.
"""
from __future__ import annotations

import copy

from configs.baseline.CoDER_whu import config as _whu_base
from configs.revision.common import DATASETS, assemble, cloud_value_norm, data_block

_NO_EDL = {f"loss.lam_evi_{k}": 0.0 for k in ("sar", "rgb", "fuse", "missing", "fuse_rec")}

_V3C = {"fused_input": "projected", **_NO_EDL}

VARIANTS = {
    "rebound": {},
    "r0": {"zero_recovered_evidence": True,
           "loss.lam_ce_missing": 0.0, "loss.lam_evi_missing": 0.0,
           "loss.lam_recon": 0.0, "loss.lam_evi_fuse_rec": 0.0},
    "r1": {"complete_recovery_mode": "full_image", "complete_image_width": 32,
           "recovery_discount": False, "recovered_reliability_enabled": False,
           "recovery_uncertainty_enabled": False},
    "r2": {"complete_recovery_mode": "full_feature", "complete_feature_bottleneck_ratio": 8,
           "recovery_discount": False, "recovered_reliability_enabled": False,
           "recovery_uncertainty_enabled": False},
    "r3": {"recovery_discount": False},
    "dempster": {"recovered_combination": "dempster"},
    "moar_legacy": {"moar_gate_mode": "renorm", "moar_null_expert": False},
    "fuse_simple": {"fusion_type": "fpnmoe", "use_moe": False},
    # ReBound-v3 screening (2026-09-27): fix the clean-accuracy gap to
    # modality-dropout baselines.  Chosen on WHU validation only.
    "v3a": {"fused_input": "projected"},
    "v3b": _NO_EDL,
    "v3c": {"fused_input": "projected", **_NO_EDL},
    "v3d": {"fused_input": "projected", **_NO_EDL, "encoder.wt_layer": [0] * 7,
            "reliability_features": "content"},
    # ReBound-v4 (2026-09-27): v3c + spatial modality dropout + masked fusion.
    "v4": {**_V3C, "spatial_dropout_prob": 0.5, "spatial_mask_infer": "pix",
           "fallback_rule": "pix_or_conf"},
    # Inference-only ablations (evaluate an existing checkpoint with these configs):
    "v4_nomaskinfer": {**_V3C, "spatial_dropout_prob": 0.5, "spatial_mask_infer": "none",
                       "fallback_rule": "pix_or_conf"},
    "v3c_maskinfer": {**_V3C, "spatial_mask_infer": "pix", "fallback_rule": "pix_or_conf"},
    "v3c_pixgate": {**_V3C, "fallback_rule": "pix_or_conf"},
    # (2026-09-28) plain synthetic-cloud augmentation: the baseline for v4's
    # spatial modality dropout; and v4 + anti-spillover consistency.
    "v3c_cloudaug": {**_V3C, "spatial_dropout_prob": 0.5, "occlusion_mode": "cloud",
                     "spatial_mask_infer": "none", "fallback_rule": "pix_or_conf"},
    "v4s": {**_V3C, "spatial_dropout_prob": 0.5, "spatial_mask_infer": "pix",
            "fallback_rule": "pix_or_conf", "loss.lam_spill": 1.0},
}
# (2026-09-29, plan A) recovery ladder on the v3c structure: R0 no transfer,
# R1 full-image, R2 full-feature, R3 shared transfer without discount.
for _r in ("r0", "r1", "r2", "r3"):
    VARIANTS[_r + "c"] = {**_V3C, **VARIANTS[_r]}


def model_block(dataset: str) -> dict:
    ds = DATASETS[dataset]
    model = copy.deepcopy(_whu_base["model"])
    p = model["params"]
    p.update(
        num_classes=ds["num_classes"],
        sar_in_channels=ds.get("sar_channels", 1),
        infer_mode="fuse",
        missing_modality_total_iterations=ds["iters"],
        # fused path
        fusion_type="moar", moar_dmfia_units=0, moar_num_experts=3, moar_top_k=1,
        moar_gate_mode="switch", moar_null_expert=True,
        fused_head="multilevel", fused_head_dropout=0.1,
        # recovery + evidence
        recovery_method="det", missing_readout="head", recovery_discount=True,
        observed_reliability_enabled=True, observed_reliability_mode="covariance_deviation",
        obs_diag_weight=0.25, recovered_reliability_enabled=True,
        recovery_uncertainty_enabled=True, dempster_enabled=True,
        recovered_combination="average", zero_recovered_evidence=False,
        reliability_fallback=True,
        # chosen on validation corruptions (revision_tgrs/gate_analysis, 2026-09-26)
        fallback_rule="hybrid",
        complete_recovery_mode="none",
        cloud_value_norm=cloud_value_norm(ds),
    )
    p["loss"]["label_smoothing"] = 0.05
    p["loss"]["class_weights"] = ds["class_weights"]
    # MetaRS whitening statistics must come from the v1 training split only.
    p["data"] = data_block(dataset)["mmr"]
    return model


def _apply(model: dict, overrides: dict) -> dict:
    p = model["params"]
    for key, value in overrides.items():
        if key.startswith("loss."):
            p["loss"][key[len("loss."):]] = value
        elif key.startswith("encoder."):
            p["encoder"][key[len("encoder."):]] = value
        else:
            p[key] = value
    return model


def build(dataset: str, variant: str) -> dict:
    model = _apply(model_block(dataset), VARIANTS[variant])
    if variant in ("r1", "r2", "r1c", "r2c"):
        # Two MoAR graphs (clean + recovered) exceed memory without this.
        model["params"]["complete_checkpoint_clean_fusion"] = True
    return assemble(model, dataset)
