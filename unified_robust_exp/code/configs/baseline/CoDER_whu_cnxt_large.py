"""ReBound with ConvNeXt-Large backbone (matches PAD's backbone strength).

Same full ReBound method (MoAR + DMFIA + multi-level decoder + recipe) as
CoDER_whu_moar_ml_r1_aug, but swaps ResNet-50 -> ConvNeXt-Large backbone.

ConvNeXt-Large output channels: [192, 384, 768, 1536] (vs ResNet [256, 512, 1024, 2048]).
This matches PAD's ConvNeXt-V2-Large backbone capacity, removing the backbone
disadvantage that limited the ResNet-50 version to 57.26%.
"""
import copy

import cv2
from albumentations import (
    Affine, CoarseDropout, Compose, HorizontalFlip, Normalize, PadIfNeeded,
    RandomCrop, RandomRotate90, RandomScale, VerticalFlip,
)
import ever as er

from configs.baseline.CoDER_whu_moar_ml_r1_aug import config as _base
from configs.metadata.WHU import sar_mean_v, sar_std_v, rgb_mean_v, rgb_std_v

config = copy.deepcopy(_base)

# --- ConvNeXt-Large backbone ---
config["model"]["params"]["backbone"] = "convnext_large"

# Disable MMR/whitening for ConvNeXt (IN layers not fully compatible with MMR
# covariance machinery; MMR is for robustness, not clean accuracy).
config["model"]["params"]["encoder"]["wt_layer"] = [0, 0, 0, 0, 0, 0, 0]
config["model"]["params"]["begin_mmr_iter"] = 999999  # effectively disable

# ConvNeXt-Large feature pyramid channels: [192, 384, 768, 1536]
config["model"]["params"]["fuse_inchan_list"] = [192, 384, 768, 1536]
config["model"]["params"]["in_channels_list"] = (0, 0, 192, 384, 768, 1536, 3072)

# FPN in_channels_list = fuse_inchan_list doubled (SAR+RGB concat)
config["model"]["params"]["fpn"]["in_channels_list"] = [384, 768, 1536, 3072]

# Keep the same augmentation as r1_aug (already set in _base)
