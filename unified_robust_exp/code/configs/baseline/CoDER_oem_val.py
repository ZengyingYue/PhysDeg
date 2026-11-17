"""ReBound on OpenEarthMap-SAR — eval-only variant pointing test at OFFICIAL val.

Identical to baseline.CoDER_oem except the test loader reads the tiled official
val split (SAR-only, zero-RGB placeholder; see tools/prep_oem_val.py). Used only
to report the SAR-only floor on the official missing-modality benchmark:

  python tools/eval_robust.py --ckpt <Best.pth> --config baseline.CoDER_oem_val --quick

Read ONLY the "RGB-missing (SAR-only)" line from its output -- clean-fuse / RGB
ladder are meaningless here (RGB is a zero placeholder).
"""
import copy
import os

from configs.baseline.CoDER_oem import config as _base
from configs.metadata.OEM_SAR import root_path

config = copy.deepcopy(_base)

_val_root = os.path.join(root_path, "officialval")
_tp = config["data"]["test"]["params"]
_tp["image_dir"] = [os.path.join(_val_root, "images")]
_tp["mask_dir"] = [os.path.join(_val_root, "masks")]
