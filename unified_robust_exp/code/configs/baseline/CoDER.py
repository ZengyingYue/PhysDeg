"""CoDER baseline config.

Derived from configs/baseline/MetaRS.py but with the evidential head and
covariance-driven reliability discount wired in.  The fused/MoE decoder is
removed; Dempster combination at the evidence level replaces it.

v3 adds stronger train-time augmentations (RandomScale + PadIfNeeded) by
overriding the data["train"] transforms block.  MetaRS reproduction remains
unaffected because it reads `data` directly from `configs.base.EarthMiss`.
"""
import copy

import cv2
from albumentations import (
    Compose, HorizontalFlip, Normalize, PadIfNeeded, RandomCrop,
    RandomRotate90, RandomScale, VerticalFlip,
)
import ever as er

from module.baseline.base import default_conv_block
from configs.base.EarthMiss import train, test, data, optimizer, learning_rate
from configs.metadata.EarthMiss import sar_mean_v, sar_std_v, rgb_mean_v, rgb_std_v

# v3 augmentations — only used for the CoDER train loader.
# RandomScale(-0.25, +0.5) shrinks / enlarges 0.75x ~ 1.5x; PadIfNeeded
# guards against shrink-below-crop-size; then RandomCrop(512).  Increases
# scale invariance vs MetaRS's fixed-crop transforms.
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
    er.preprocess.albu.ToTensor()
])
data = _aug_data


config = dict(
    model=dict(
        type='CoDER',
        params=dict(
            sar_in_channels=1,
            rgb_in_channels=3,
            infer_rgb=False,
            infer_mode='sar',  # main test mode = SAR-only (paper headline)
            # missing-modality training
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
            # Pillar-2 reliability discount hyperparams (v2.2 calibrated).
            # v2.1 probe showed mIoU regression at MMR activation (β cliff from
            # 1.0 → β_mmr instantly).  v2.2 adds a linear ramp over
            # `beta_ramp_iters` steps after `begin_mmr_iter` so the EDL head
            # has time to adapt to the discount.
            beta_scale=3.0,
            beta_floor=0.3,        # slightly lower than v2.1 (0.4) — with ramp, more room
            gamma_rec_init=0.3,
            beta_ramp_iters=3000,  # linear ramp from β=1 to β_mmr over 3k steps after MMR
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
            num_classes=8,
            # MoE fusion knobs (v3-S1) — replaces MetaRS `Fuse` module
            use_moe=True,
            router_hidden=128,
            router_dropout=0.1,
            loss=dict(
                # ===== Primary classification (CE on softmax) =====
                # The fused head is the headline (matches MetaRS structure);
                # SAR head is the eval path for "sar" mode.  RGB head is
                # auxiliary because we never report RGB-only mIoU.
                lam_ce_sar=1.0,
                lam_ce_rgb=0.5,
                lam_ce_fuse=1.5,
                lam_ce_missing=1.0,
                # ===== Auxiliary EDL (uncertainty path) =====
                evidence_activation='softplus',
                edl_form='mse',
                ce_aux_weight=0.0,      # CE already supplied at the primary; no need to double up here
                kl_max_lambda=0.02,
                kl_ramp_iters=10000,
                lam_evi_sar=0.3,
                lam_evi_rgb=0.2,
                lam_evi_fuse=0.3,       # EDL on Dempster(α_sar, α_rgb) — uncertainty path
                lam_evi_missing=0.3,
                lam_evi_fuse_rec=0.3,
                lam_distill=0.0,
                # ===== Auxiliary regularisers =====
                lam_dice=0.3,           # dice on softmax(fused_logits)
                lam_balance=0.05,       # MoE entropy reward
                # ===== Pillar-1 anchors (kept from MetaRS) =====
                mmr=dict(),
                mse=dict(),
                alpha=1.25,
                lam_mmr=1.0,
                lam_mse=1.0,
                lam_recon=0.5,
                # ===== Tail-class boosts =====
                # Class weights indexed by label (Background, Building, Road,
                # Water, Barren, Forest, Agricultural, Playground).  Boost
                # the two zero-IoU classes; keep others at 1.0.
                class_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0),
                # OHEM — keep hardest 50% pixels in CE (and EDL via the same flag)
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
