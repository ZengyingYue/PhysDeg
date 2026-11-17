"""Full-recovery baseline (ReBound ablation) — naive recovery, NO reliability bound.

= s4 recipe (CoDER_whu_moar_ml_r1) but the recovered modality is used at FULL weight:
  recovery_discount = False   (β_rec = 1 everywhere; recovered evidence un-discounted)
  missing_readout   = dempster (recovery enters the prediction, not just the SAR head)
  loss.lam_ce_dst_rec ↑       (the model is trained to predict from the recovered-fused
                               Dempster, i.e. to TRUST hallucinated content)

This trains the model to rely on cross-modal recovery as if it were observed. Per the
recoverable-boundary result (Fig. boundary), recovery fabricates optical-specific
content, so this baseline should be *confidently wrong* on optical-specific classes,
whereas ReBound (s4) bounds recovery and defers. Compared via tools/eval_hallucination.py.
"""
from copy import deepcopy

from configs.baseline.CoDER_whu_moar_ml_r1 import config as _base

config = deepcopy(_base)
p = config["model"]["params"]
p["recovery_discount"] = False
p["missing_readout"] = "dempster"
p["loss"]["lam_ce_dst_rec"] = 1.5
