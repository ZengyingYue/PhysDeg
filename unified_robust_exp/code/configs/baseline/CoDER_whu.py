"""CoDER on WHU-OPT-SAR — CLEAN (complete-modality) benchmark vs PAD (56.26).

Same CoDER method/recipe as configs/baseline/CoDER.py, retargeted to WHU:
  - num_classes = 7  (Farmland/City/Village/Water/Forest/Road/Others)
  - infer_mode = 'fuse'  → eval uses softmax(fused_logits), i.e. the
    complete-modality (RGB+SAR) prediction.  This is the "clean" headline
    compared to PAD.
  - missing_modality_training stays ON (the CoDER recipe): the fused head is
    trained every iteration with both modalities present (the missing branch
    is purely additive), so eval-fuse reflects a fully trained joint head —
    and the SAME checkpoint can be re-evaluated in 'sar' mode for the
    SAR-only number as a bonus.
  - class_weights: boost the two rarest classes Road(5) & Others(6) by 3x
    (mirrors the EarthMiss CoDER tail-class boost).

Speed: data IO is solved by the loader's RAM cache; here we also re-enable
cuDNN (train.py's seed_torch disables it for determinism) so the 3x ResNet-50
encoders run at full conv throughput.  Set WHU_FAST=0 to keep determinism.
"""
import copy
import os

import cv2
import torch
from albumentations import (
    Compose, HorizontalFlip, Normalize, PadIfNeeded, RandomCrop,
    RandomRotate90, RandomScale, VerticalFlip,
)
import ever as er

# Re-enable cuDNN (seed_torch in train.py turns it off for bit-determinism).
if os.environ.get("WHU_FAST", "1") == "1":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

from module.baseline.base import default_conv_block
from configs.base.WHU import train, test, data, optimizer, learning_rate
from configs.metadata.WHU import sar_mean_v, sar_std_v, rgb_mean_v, rgb_std_v

# v3 augmentations for the CoDER train loader (scale invariance).
_aug_data = copy.deepcopy(data)
_aug_data["train"]["params"]["transforms"] = Compose([
    RandomScale(scale_limit=(-0.25, 0.5), p=0.5),
    PadIfNeeded(min_height=512, min_width=512,
                border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=-1),
    RandomCrop(512, 512),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomRotate90(p=0.5),
    Normalize(mean=sar_mean_v + rgb_mean_v,
              std=sar_std_v + rgb_std_v,
              max_pixel_value=1),
    er.preprocess.albu.ToTensor(),
])
data = _aug_data


config = dict(
    model=dict(
        type='CoDER',
        params=dict(
            sar_in_channels=1,
            rgb_in_channels=3,
            infer_rgb=False,
            infer_mode='fuse',  # CLEAN: complete-modality fused prediction
            # missing-modality training (kept — CoDER recipe; fused head is
            # always trained complete, so this only adds the recovery aux).
            missing_modality_training=True,
            missing_modality_prob=dict(
                complete=1 / 3,
                sar_missing=1 / 3,
                rgb_missing=1 / 3,
            ),
            missing_modality_total_iterations=15000,
            in_channels_list=(0, 0, 64, 256, 512, 1024, 2048),
            fuse_inchan_list=(256, 512, 1024, 2048),
            relax_denom=0,
            clusters=10,
            begin_mmr_iter=1600,
            beta_scale=3.0,
            beta_floor=0.3,
            gamma_rec_init=0.3,
            beta_ramp_iters=3000,
            encoder=dict(
                resnet_type='resnet50',
                include_conv5=True,
                batchnorm_trainable=True,
                pretrained=True,
                freeze_at=0,
                output_stride=32,
                wt_layer=[0, 0, 2, 2, 2, 0, 0],
                num_classes=1,
            ),
            fpn=dict(
                in_channels_list=(512, 1024, 2048, 4096),
                out_channels=256,
                conv_block=default_conv_block,
                top_blocks=None,
            ),
            decoder=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
            ),
            scale_factor=16,
            num_classes=7,
            use_moe=True,
            router_hidden=128,
            router_dropout=0.1,
            loss=dict(
                lam_ce_sar=1.0,
                lam_ce_rgb=0.5,
                lam_ce_fuse=1.5,
                lam_ce_missing=1.0,
                evidence_activation='softplus',
                edl_form='mse',
                ce_aux_weight=0.0,
                kl_max_lambda=0.02,
                kl_ramp_iters=10000,
                lam_evi_sar=0.3,
                lam_evi_rgb=0.2,
                lam_evi_fuse=0.3,
                lam_evi_missing=0.3,
                lam_evi_fuse_rec=0.3,
                lam_distill=0.0,
                lam_dice=0.3,
                lam_balance=0.05,
                mmr=dict(),
                mse=dict(),
                alpha=1.25,
                lam_mmr=1.0,
                lam_mse=1.0,
                lam_recon=0.5,
                # 7 classes: Farmland,City,Village,Water,Forest,Road,Others
                # boost the two rarest (Road, Others).
                class_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0),
                ohem_keep_ratio=0.5,
                ignore_index=-1,
            ),
            data=data["mmr"],
        )
    ),
    data=data,
    optimizer=optimizer,
    learning_rate=learning_rate,
    train=train,
    test=test,
)
