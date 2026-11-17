"""Exp2 (hallucination architecture-confound test) — ReBound MAIN architecture
(ResNet50 + MoAR, the WHU/OEM headline arch) trained on EarthMiss.

WHY: the paper's prediction-level hallucination signature (full recovery is
confidently-wrong 1.8-2.6x more often on optical-specific classes) was measured on
`em_moar`, which is **ConvNeXt-base + latent flow** -- a DIFFERENT architecture from
the main ResNet50+MoAR. On WHU/OEM the same eval is NULL, but those datasets have no
optical-specific classes. This run isolates dataset vs architecture: EarthMiss HAS
optical-specific classes (barren/forest/agricultural), so if the signature reproduces
HERE on the MAIN architecture, the EarthMiss result is not an architecture artifact.

= deepcopy(CoDER_whu_moar_ml_r1) (ResNet50 encoders + MoAR fusion + MultiLevelDecoder
+ EDL + bounded recovery + modality-dropout), with ONLY the data block + class count
swapped to EarthMiss -- exactly how CoDER_oem retargets the same architecture.

Eval (after training): tools/eval_hallucination.py --config baseline.CoDER_em_resnet_moar
--ckpt <Best.pth> --per_class  (toggles force_beta_rec=1 vs trained on ONE ckpt;
read conf_wrong on the 'specific' group = barren/forest/agri).
"""
import copy

from configs.baseline.CoDER_whu_moar_ml_r1 import config as _base
from configs.base.EarthMiss import data as _em_data, optimizer as _em_opt, \
    learning_rate as _em_lr, train as _em_train, test as _em_test

config = copy.deepcopy(_base)

# swap the data pipeline wholesale (WHU root-based -> EarthMiss city layout)
config["data"] = _em_data
config["optimizer"] = _em_opt
config["learning_rate"] = _em_lr
config["train"] = _em_train
config["test"] = _em_test

_K = 8  # EarthMiss: background, building, road, water, barren, forest, agri, playground
_p = config["model"]["params"]
_p["data"] = _em_data["mmr"]
_p["num_classes"] = _K
# mild 2x boost on the optical-specific classes we test (barren=4, forest=5, agri=6)
# so the model actually learns them -> the hallucination contrast is well-sampled.
_p["loss"]["class_weights"] = (1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0)
