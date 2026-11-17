"""ReBound on OpenEarthMap-SAR (DFC25 Track 1) — the 3rd robustness dataset.

= the WHU s4 headline architecture+recipe (baseline.CoDER_whu_moar_ml_r1:
ResNet50 encoders + MoAR fusion + MultiLevelDecoder fused head + label
smoothing 0.05), retargeted to OEM-SAR by swapping ONLY the data block and the
class count. Architecture/loss/EDL/recovery are identical, so OEM-SAR is a clean
transfer of the same model — the point is to show graceful RGB-degradation +
modality-drop on a third, genuine optical-SAR dataset.

  - data            -> configs.base.OEM_SAR (EarthM3DALoader, OEM stats/dirs)
  - num_classes     -> 8  (OEM 8 land-cover classes; 0 -> ignore)
  - class_weights   -> uniform (no curated tail-class boost; OEM classes are
                       more balanced than WHU's Road/Others tail)
  - infer_mode stays 'fuse' (clean = complete-modality fused readout)

Long-run schedule (bf16, bs16, lr2e-4, 40k iters) is passed as launch overrides
in scripts/train_oem.sh, matching the s4 recipe.
"""
import copy

from configs.baseline.CoDER_whu_moar_ml_r1 import config as _base
from configs.base.OEM_SAR import data as _oem_data, optimizer as _oem_opt, \
    learning_rate as _oem_lr, train as _oem_train, test as _oem_test
from configs.metadata.OEM_SAR import num_classes as _K

config = copy.deepcopy(_base)

# swap the data pipeline wholesale (WHU root-based -> OEM EarthM3 layout)
config["data"] = _oem_data
config["optimizer"] = _oem_opt
config["learning_rate"] = _oem_lr
config["train"] = _oem_train
config["test"] = _oem_test

_p = config["model"]["params"]
_p["data"] = _oem_data["mmr"]
_p["num_classes"] = _K
_p["loss"]["class_weights"] = tuple([1.0] * _K)

# --- s4 long-run recipe, baked in (single source of truth; no launch-arg
#     fragility). bf16 itself is the only piece passed on the CLI. ---
_NUM_ITERS = 40000
_p["missing_modality_total_iterations"] = _NUM_ITERS
config["train"]["num_iters"] = _NUM_ITERS
config["learning_rate"]["params"]["max_iters"] = _NUM_ITERS
config["learning_rate"]["params"]["base_lr"] = 2e-4
config["data"]["train"]["params"]["batch_size"] = 16
