"""C2 of the clean-58 sweep — = CoDER_whu_moar_ml_r1 with STRONGER train-time
augmentation only (user hypothesis: stronger aug usually gains a point).

Arch-preserving, robustness-preserving: identical model/recipe to r1; the ONLY
change is the train loader's transform Compose:
  - wider multi-scale    RandomScale (-0.25,0.5)->(-0.5,1.0), p 0.5->0.7  (the
                         single highest-ROI segmentation aug)
  - small continuous rotation  Affine(rotate +/-15deg, p 0.3)  (on top of the
                         existing RandomRotate90 which only does 90deg multiples)
  - occlusion regulariser CoarseDropout (holes filled 0 in image, ignore_index
                         in mask so loss skips them), p 0.3
Eval transform is untouched (no test-time aug — fair vs baselines).
All aug is geometric/occlusion (modality-agnostic) — safe for the stacked
SAR(1)+RGB(3) input; NO photometric jitter (would corrupt the SAR channel).
"""
import copy

import cv2
from albumentations import (
    Affine, CoarseDropout, Compose, HorizontalFlip, Normalize, PadIfNeeded,
    RandomCrop, RandomRotate90, RandomScale, VerticalFlip,
)
import ever as er

from configs.baseline.CoDER_whu_moar_ml_r1 import config as _base
from configs.metadata.WHU import sar_mean_v, sar_std_v, rgb_mean_v, rgb_std_v

config = copy.deepcopy(_base)

config["data"]["train"]["params"]["transforms"] = Compose([
    RandomScale(scale_limit=(-0.5, 1.0), p=0.7),
    PadIfNeeded(min_height=512, min_width=512,
                border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=-1),
    RandomCrop(512, 512),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomRotate90(p=0.5),
    Affine(rotate=(-15, 15), border_mode=cv2.BORDER_CONSTANT,
           fill=0, fill_mask=-1, p=0.3),
    CoarseDropout(num_holes_range=(1, 8),
                  hole_height_range=(0.02, 0.10),
                  hole_width_range=(0.02, 0.10),
                  fill=0, fill_mask=-1, p=0.3),
    Normalize(mean=sar_mean_v + rgb_mean_v,
              std=sar_std_v + rgb_std_v,
              max_pixel_value=1),
    er.preprocess.albu.ToTensor(),
])
