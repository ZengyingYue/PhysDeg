"""OpenEarthMap-SAR data/optimizer/schedule blocks (EarthM3DALoader).

Mirrors configs/base/EarthMiss.py but:
  - pulls OEM-SAR stats/dirs from metadata/OEM_SAR.py
  - the train loader uses the CoDER 'v3' augmentation (RandomScale + Pad +
    RandomCrop + flips), matching the WHU s4 headline recipe, instead of the
    plain RandomCrop. val/mmr/test keep Normalize+ToTensor only.

train -> patches/train ; val/mmr/test -> patches/heldout (the held-out RGB+SAR
split carved from the labelled train set). Schedule/optimizer mirror EarthMiss;
the long-run iters/lr/bs are passed as launch overrides (scripts/train_oem.sh).
"""
import cv2
from albumentations import (
    Compose, Normalize, HorizontalFlip, VerticalFlip, RandomRotate90,
    RandomCrop, RandomScale, PadIfNeeded,
)
import ever as er
from ..metadata.OEM_SAR import *

_train_tf = Compose([
    RandomScale(scale_limit=(-0.25, 0.5), p=0.5),
    PadIfNeeded(min_height=patch_size, min_width=patch_size,
                border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=-1),
    RandomCrop(patch_size, patch_size),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomRotate90(p=0.5),
    Normalize(mean=sar_mean_v + rgb_mean_v, std=sar_std_v + rgb_std_v,
              max_pixel_value=1, always_apply=True),
    er.preprocess.albu.ToTensor(),
])

_eval_tf = Compose([
    Normalize(mean=sar_mean_v + rgb_mean_v, std=sar_std_v + rgb_std_v,
              max_pixel_value=1, always_apply=True),
    er.preprocess.albu.ToTensor(),
])

data = dict(
    train=dict(
        type='EarthM3DALoader',
        params=dict(
            image_dir=train_image_dir,
            mask_dir=train_mask_dir,
            sensors=('SAR', 'RGB'),
            transforms=_train_tf,
            CV=dict(k=10, i=-1),
            distributed=True,
            training=True,
            batch_size=8,
            num_workers=4,
        ),
    ),
    val=dict(
        type='EarthM3DALoader',
        params=dict(
            image_dir=val_image_dir,
            mask_dir=val_mask_dir,
            sensors=('SAR', 'RGB'),
            transforms=_eval_tf,
            CV=dict(k=10, i=-1),
            training=False,
            batch_size=4,
            num_workers=4,
        ),
    ),
    mmr=dict(
        type='EarthM3DALoader',
        params=dict(
            image_dir=train_image_dir,
            mask_dir=train_mask_dir,
            sensors=('SAR', 'RGB'),
            transforms=_eval_tf,
            CV=dict(k=10, i=-1),
            distributed=False,
            training=False,
            batch_size=4,
            num_workers=4,
        ),
    ),
    test=dict(
        type='EarthM3DALoader',
        params=dict(
            image_dir=test_image_dir,
            mask_dir=test_mask_dir,
            sensors=('SAR', 'RGB'),
            transforms=_eval_tf,
            CV=dict(k=10, i=-1),
            distributed=True,
            training=False,
            batch_size=16,
            num_workers=4,
        ),
    ),
)

optimizer = dict(
    type='adamw',
    params=dict(lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01),
)

learning_rate = dict(
    type='poly',
    params=dict(base_lr=1e-4, power=0.9, max_iters=15000),
)

train = dict(
    forward_times=1,
    num_iters=15000,
    eval_per_epoch=True,
    summary_grads=False,
    summary_weights=False,
    distributed=True,
    apex_sync_bn=True,
    sync_bn=True,
    eval_after_train=True,
    log_interval_step=50,
    save_ckpt_interval_epoch=4,
    eval_interval_epoch=4,
    distributed_evaluate=True,
    best_checkpoint=dict(
        enabled=True,
        metric='miou',
        filename='Best.pth',
        initial_score=float('-inf'),
        delete_old_checkpoints=True,
    ),
)

test = dict()
