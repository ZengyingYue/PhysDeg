"""Baselines under the revision protocol: same data, schedule, selection and
segmentation loss as ReBound (configs/revision/common.py).

moddrop   ModalityDropoutFusion (concat fusion + per-sample modality dropout)
pad       PADFusionR50 (PAD fusion on ResNet-50 + modality dropout)
shaspec   ShaSpec (shared/specific features + missing-feature generation)
metars    MetaRS (in-repo reimplementation; its own mechanism losses, the
          shared segmentation loss)
stars     STARS (authors' release architecture, paper objective; module/baseline/stars.py)
"""
from __future__ import annotations

import copy

from configs.baseline.MetaRS_whu import config as _metars_whu
from configs.revision.common import DATASETS, assemble, cloud_value_norm, data_block

TYPES = {"moddrop": "ModalityDropoutFusion", "pad": "PADFusionR50", "shaspec": "ShaSpec"}


def seg_loss(dataset: str) -> dict:
    return dict(class_weights=DATASETS[dataset]["class_weights"], label_smoothing=0.05,
                ohem_keep_ratio=0.5, lam_dice=0.3)


OCCLUSION = {"moddrop_hole": "hole", "moddrop_cloudaug": "cloud"}


# Degradation-posterior distillation screening (any base: moddrop / pad / shaspec):
#   <base>_degaug  seen-type degradation augmentation, hard labels only
#   <base>_kd      EMA clean-pair teacher distillation, no degradation
#   <base>_dpd     both
# Screening round 2 (after round 1 on WHU val): <base>_dpd5 raises the KD weight to 5,
# <base>_degaug25 halves the degradation probability (clean-accuracy trade-off).
# Physical degradations (revision_tgrs/PHYS_PLAN.md): <base>_physopt25 swaps the generic
# optical families for spectral haze / cloud+shadow; <base>_phys25 also uses the physical
# SAR families and co-registration shifts; <base>_gp25 uses generic and physical optical.
RECIPES = {"degaug": dict(degrade=0.5), "kd": dict(kd=1.0), "dpd": dict(degrade=0.5, kd=1.0),
           "dpd5": dict(degrade=0.5, kd=5.0), "degaug25": dict(degrade=0.25),
           "dpd25": dict(degrade=0.25, kd=1.0),
           "physopt25": dict(degrade=0.25, optical="phys"),
           "phys25": dict(degrade=0.25, optical="phys", sar="phys", coreg=0.25),
           "gp25": dict(degrade=0.25, optical="both", sar="phys", coreg=0.25)}
# Component ablations of physopt25 (PHYS_PLAN.md section 5): one physical ingredient
# removed each; identical parameter draws otherwise.
_PO25 = dict(degrade=0.25, optical="phys")
RECIPES.update({
    "physopt25noshadow": dict(_PO25, ablate=("noshadow",)),       # clouds without shadows
    "physopt25coloc": dict(_PO25, ablate=("coloc",)),             # shadow under its cloud
    "physopt25indshadow": dict(_PO25, ablate=("indepshadow",)),   # shadow unrelated to cloud
    "physopt25grey": dict(_PO25, ablate=("grey",)),               # no wavelength dependence
    "physopt25cloudonly": dict(_PO25, optical_types=("cloud_shadow",)),  # no haze family
})
# The same ablations on phys25 (used if phys25 is the selected recipe).
_P25 = dict(degrade=0.25, optical="phys", sar="phys", coreg=0.25)
RECIPES.update({f"phys25{k[len('physopt25'):]}": dict(v, **{a: b for a, b in _P25.items()})
                for k, v in list(RECIPES.items()) if k.startswith("physopt25") and k != "physopt25"})
# Physically supervised optical reliability (PHYS_PLAN.md section 6; moddrop only):
# rel<recipe> gates the optical features per pixel by the predicted reliability,
# relaux<recipe> trains the same head as an auxiliary task only (control).
_REL = dict(lam=1.0, gate=True, level=2)
for _base in ("physopt25", "phys25"):
    RECIPES[f"rel{_base}"] = dict(RECIPES[_base], rel=_REL)
    RECIPES[f"relaux{_base}"] = dict(RECIPES[_base], rel=dict(_REL, gate=False))


# Improvement round (PHYS_PLAN.md section 7): EMA weights for every new run; 'u' = U-shaped
# cloud cover reaching overcast; relaux = physically supervised reliability head (no gating).
_GP25 = RECIPES["gp25"]
RECIPES.update({
    "gp25u": dict(_GP25, coverage="ushape"),  # PHYS_PLAN 9: no EMA, same protocol as the baselines
    "gp25ema": dict(_GP25, ema=0.999),
    "gp25uema": dict(_GP25, coverage="ushape", ema=0.999),
    "relauxgp25uema": dict(_GP25, coverage="ushape", ema=0.999, rel=dict(_REL, gate=False)),
})


def _recipe(config: dict, ds: dict, degrade: float = 0.0, kd: float = 0.0,
            optical: str = "generic", sar: str = "generic", coreg: float = 0.0,
            ablate: tuple = (), optical_types: tuple | None = None,
            rel: dict | None = None, coverage: str = "", ema: float = 0.0) -> dict:
    p = config["model"]["params"]
    if degrade:
        p["degrade"] = dict(prob=degrade, sar_prob=degrade / 2, rgb_mean=ds["rgb_mean"],
                            rgb_std=ds["rgb_std"], sar_mean=ds["sar_mean"], sar_std=ds["sar_std"],
                            optical=optical, sar=sar, coreg_prob=coreg,
                            unit=ds.get("cloud_raw", 235.0) / 235.0 if ds.get("max_dn") else 1.0,
                            max_dn=ds.get("max_dn", 255.0), sar_domain=ds.get("sar_domain", "amp"),
                            gsd=ds.get("gsd", 5.0))
        if ablate:
            p["degrade"]["ablate"] = list(ablate)
        if optical_types:
            p["degrade"]["optical_types"] = list(optical_types)
        if coverage:
            p["degrade"]["coverage"] = coverage
    if ema:  # weight averaging; Best.pth / Last.pth then hold the EMA weights (train.py)
        config["train"]["ema_decay"] = ema
        config["train"]["ema_warmup"] = 500
    if rel:
        if config["model"]["type"] != "ModalityDropoutFusion" or optical not in ("phys", "both"):
            raise ValueError("reliability supervision: moddrop with physical optical recipes")
        config["model"]["type"] = "ModalityDropoutFusionRel"
        p["rel"] = dict(rel)
    if kd:
        p["kd"] = dict(lam=kd, start=2000, ramp=4000, ema=0.999)
    return config


def build(dataset: str, method: str) -> dict:
    ds = DATASETS[dataset]
    base, _, recipe = method.rpartition("_")
    if recipe in RECIPES and (base in TYPES or base in ("metars", "stars")):  # metars/stars: degrade only
        return _recipe(build(dataset, base), ds, **RECIPES[recipe])
    if method in OCCLUSION:  # ModDrop + spatial occlusion training (revision ablation)
        config = build(dataset, "moddrop")
        config["model"]["params"].update(occlusion_mode=OCCLUSION[method], occlusion_prob=0.5,
                                         cloud_value_norm=cloud_value_norm(ds))
        return config
    if method == "stars":  # module/baseline/stars.py: authors' release + paper objective
        model = dict(type="STARS", params=dict(
            num_classes=ds["num_classes"], sar_in_channels=ds.get("sar_channels", 1),
            infer_mode="fuse", loss=dict(harmonized=seg_loss(dataset))))
        return assemble(model, dataset)
    if method == "metars":
        model = copy.deepcopy(_metars_whu["model"])
        p = model["params"]
        p["num_classes"] = ds["num_classes"]
        p["sar_in_channels"] = ds.get("sar_channels", 1)
        p["infer_mode"] = "fuse"
        p["missing_modality_total_iterations"] = ds["iters"]
        p["loss"]["harmonized"] = seg_loss(dataset)
        p["data"] = data_block(dataset)["mmr"]
    else:
        model = dict(type=TYPES[method], params=dict(
            num_classes=ds["num_classes"], sar_in_channels=ds.get("sar_channels", 1),
            infer_mode="fuse", loss=seg_loss(dataset)))

    return assemble(model, dataset)
