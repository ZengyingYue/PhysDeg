"""EarthMiss long-run (B1) — recipe-aligned re-run of the MoAR arm.

= baseline.CoDER_flow_cb_moar (the VALIDATED EarthMiss CoDER+MoAR config that
produced runs/em_moar) with the single code-level recipe change carried here:
  - loss.label_smoothing = 0.05   (matches the WHU s4 recipe; CoDER default 0.0)

Rationale: em_moar reached only 0.3008 SAR-only — but that was 15k iters
(~7-10 epochs, undertrained) AND it is the *held-out-cities SAR-only* metric
(infer_mode='sar'), the hardest readout. A longer schedule should lift all
readouts; the clean fused number (infer_mode='fuse') is the paper's main
EarthMiss cell. The rest of the recipe is launch overrides (run_em_r1.sh):
  num_iters 50000  (~25-33 ep), base_lr 1e-4, bs8, fp32 (convnext_base+flow
  backbone is untested under bf16 → keep fp32 for the unattended run).

Eval offline: fuse (clean), sar (RGB-missing), rgb (SAR-missing), + robust +
calibration sweeps. Train/val/test city split is fixed in metadata/EarthMiss.py
(train 7 / test 3 unseen cities).
"""
from copy import deepcopy

from configs.baseline.CoDER_flow_cb_moar import config as _base

config = deepcopy(_base)
config["model"]["params"]["loss"]["label_smoothing"] = 0.05
