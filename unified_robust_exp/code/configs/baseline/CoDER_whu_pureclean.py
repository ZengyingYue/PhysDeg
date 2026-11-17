"""Diagnostic variant: CoDER on WHU clean with missing-modality training OFF.

Isolates how much the missing-modality auxiliary (recovery + EDL-on-recovered
+ recon losses) "taxes" the clean fused head.  Identical to baseline.CoDER_whu
in every other respect (architecture, MMR, EDL on the complete path, infer_mode
'fuse'), so the only difference vs CoDER_whu is `missing_modality_training=False`.

Used for Step-1 of the unified-model paper plan (see ../../README.md): if this
run jumps toward 57.79 (our CMNeXtSGSLN clean SOTA), the clean gap is the
robustness tax; if it stays ~55, the gap is the fusion architecture.
"""
import copy

from configs.baseline.CoDER_whu import config as _base

config = copy.deepcopy(_base)
config["model"]["params"]["missing_modality_training"] = False
