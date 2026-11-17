"""WHU calibration-retune arm (candidate fix for QUESTIONS Q9).

A3 found the evidential confidence max(α/S) is severely underconfident (ECE 0.476):
the Dirichlet is too diffuse (u=K/S≈0.33 everywhere) and meanU barely responds to
degradation. Root cause: the KL-to-uniform EDL regularizer + weak evidence weights
keep S small, and beta_floor=0.3 limits how far β (hence u) can move.

This arm = s4 recipe (baseline.CoDER_whu_moar_ml_r1) with the evidential path
re-tuned to SHARPEN the Dirichlet and widen β's dynamic range:
  - kl_max_lambda 0.02 → 0.0   (remove the uniform pull that inflates u)
  - lam_evi_*      ×3           (stronger evidence supervision → larger S → smaller u, sharper α/S)
  - beta_scale     3.0 → 6.0    (reliable channels → β≈1, more evidence passes)
  - beta_floor     0.3 → 0.1    (corrupted channels → β can drop further → u rises under degradation)

Accuracy path (fused softmax CE) is UNCHANGED, so clean mIoU and the SAR-only floor
should be preserved; the only intended change is a better-calibrated evidential readout.
Eval: same calib sweep; success = EVI ECE ≪ 0.476 and meanU rising with severity.
"""
from copy import deepcopy

from configs.baseline.CoDER_whu_moar_ml_r1 import config as _base

config = deepcopy(_base)
p = config["model"]["params"]
p["beta_scale"] = 6.0
p["beta_floor"] = 0.1
L = p["loss"]
L["kl_max_lambda"] = 0.0
L["lam_evi_sar"] = 0.9
L["lam_evi_rgb"] = 0.6
L["lam_evi_fuse"] = 0.9
L["lam_evi_missing"] = 0.9
L["lam_evi_fuse_rec"] = 0.9
