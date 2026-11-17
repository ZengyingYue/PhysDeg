"""MetaRS on WHU-OPT-SAR — ReBound Table 1 baseline (clean dual-modal).

MetaRS (Zhou et al., RSE 2026, "Remote Sensing Meta Modal Representation for
Missing Modality Land Cover Mapping") is a missing-modality-robust model that
also reports a clean dual-modal number. ReBound Table 1 reports MetaRS at
mIoU 55.37 / OA 82.25 (Rob ✓, Unc ✗).

There is no public official MetaRS code release; this config drives the
in-repo reimplementation at module/baseline/MetaRS.py (the same one whose
numbers are quoted in the ReBound paper). It reuses the existing WHU data
contract (4-channel [SAR,R,G,B], WhuM3DALoader, 7 classes, ignore=-1, PAD
seed-42 80/20 split) — the only adaptation is pointing root_path at the local
data and the WHU 7-class / ignore-index settings.

Usage (from unified_robust_exp/code/):
    CONFIG=baseline.MetaRS_whu bash scripts/train_whu.sh
    # or directly:
    python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py \
        --config_path=baseline.MetaRS_whu --model_dir=./log/WHU/MetaRS_whu \
        --trainer=th_ddp train.num_iters 15000

NOTE on data root: configs/metadata/WHU.py hardcodes a WSL path
(/mnt/e/...). Set WHU_ROOT to your local raw WHU root before running, e.g.
    export WHU_ROOT=/home/yue/experiment/earthmiss_baseline/datasets_raw/whu-opt-sar
This config reads WHU_ROOT at import time and falls back to the metadata default.
"""
import os

import cv2
from albumentations import (
    Compose, HorizontalFlip, Normalize, PadIfNeeded, RandomCrop,
    RandomRotate90, RandomScale, VerticalFlip,
)
import ever as er

# --- locate the WHU raw data: prefer WHU_ROOT, else metadata default ---------
# (configs/metadata/WHU.py points at a WSL path; on this Linux box the data is
#  under datasets_raw/whu-opt-sar — set WHU_ROOT to override.)
from configs.metadata.WHU import (
    sar_mean_v, sar_std_v, rgb_mean_v, rgb_std_v,
    patch_size, split_seed, train_frac,
)
WHU_ROOT = os.environ.get(
    "WHU_ROOT",
    "/home/yue/experiment/earthmiss_baseline/datasets_raw/whu-opt-sar",
)

# Rebuild the WHU data dict with our local root (copy of configs/base/WHU.py
# but with root=WHU_ROOT) so we don't depend on the WSL metadata path.
from data.WHU import WhuM3DALoader  # noqa: F401 — registers the loader

_norm = Normalize(mean=sar_mean_v + rgb_mean_v, std=sar_std_v + rgb_std_v,
                  max_pixel_value=1, always_apply=True)

_train_aug = Compose([
    RandomScale(scale_limit=(-0.25, 0.5), p=0.5),
    PadIfNeeded(min_height=patch_size, min_width=patch_size,
                border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=-1),
    RandomCrop(patch_size, patch_size),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomRotate90(p=0.5),
    _norm,
    er.preprocess.albu.ToTensor(),
])
_eval_tf = Compose([_norm, er.preprocess.albu.ToTensor()])

_common = dict(root=WHU_ROOT, patch=patch_size, seed=split_seed,
               train_frac=train_frac, cache_ram=True, CV=dict(k=10, i=-1))

data = dict(
    train=dict(type='WhuM3DALoader', params=dict(
        split='train', training=True, distributed=True, batch_size=8,
        num_workers=8, transforms=_train_aug, **_common)),
    val=dict(type='WhuM3DALoader', params=dict(
        split='test', training=False, batch_size=8, num_workers=8,
        transforms=_eval_tf, **_common)),
    mmr=dict(type='WhuM3DALoader', params=dict(
        split='train', training=False, distributed=False, batch_size=4,
        num_workers=8, transforms=_eval_tf, **_common)),
    test=dict(type='WhuM3DALoader', params=dict(
        split='test', training=False, distributed=True, batch_size=16,
        num_workers=8, transforms=_eval_tf, **_common)),
)

optimizer = dict(type='adamw', params=dict(lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01))
learning_rate = dict(type='poly', params=dict(base_lr=1e-4, power=0.9, max_iters=15000))
train = dict(
    forward_times=1, num_iters=15000, eval_per_epoch=True,
    summary_grads=False, summary_weights=False, distributed=True,
    apex_sync_bn=True, sync_bn=True, eval_after_train=True,
    log_interval_step=50, save_ckpt_interval_epoch=20, eval_interval_epoch=20,
    distributed_evaluate=True,
    best_checkpoint=dict(enabled=True, metric='miou', filename='Best.pth',
                         initial_score=float('-inf'), delete_old_checkpoints=True),
)
test = dict()

from module.baseline.base import default_conv_block

config = dict(
    model=dict(
        type='MetaRS',
        params=dict(
            sar_in_channels=1,
            rgb_in_channels=3,
            infer_rgb=False,
            infer_mode='fuse',  # CLEAN dual-modal prediction (ReBound Table 1)
            missing_modality_training=True,   # MetaRS recipe (fused head trained complete)
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
            num_classes=7,           # WHU 7-class protocol (EarthMiss config used 8)
            loss=dict(
                ce=dict(),
                ignore_index=-1,     # WHU ignore (EarthMiss also uses -1)
                mmr=dict(),
                mse=dict(),
                alpha=1.25,
            ),
            data=data["mmr"],
        ),
    ),
    data=data,
    optimizer=optimizer,
    learning_rate=learning_rate,
    train=train,
    test=test,
)
