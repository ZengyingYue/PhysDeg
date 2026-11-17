"""Base config for WHU-OPT-SAR (CoDER), PAD protocol.

Mirrors configs/base/EarthMiss.py but swaps the loader for `WhuM3DALoader`
(internal 80/20 seed-42 patch split + RAM cache) and pulls WHU normalization
statistics.  num_classes=7, ignore=-1.
"""
from albumentations import Compose, Normalize
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomCrop
import ever as er

from data.WHU import WhuM3DALoader  # noqa: F401 — side-effect: registers the loader
from ..metadata.WHU import (
    sar_mean_v, sar_std_v, rgb_mean_v, rgb_std_v, root_path,
    patch_size, split_seed, train_frac,
)

_norm = Normalize(mean=sar_mean_v + rgb_mean_v, std=sar_std_v + rgb_std_v,
                  max_pixel_value=1, always_apply=True)

data = dict(
    train=dict(
        type='WhuM3DALoader',
        params=dict(
            root=root_path, split='train', patch=patch_size,
            seed=split_seed, train_frac=train_frac, cache_ram=True,
            transforms=Compose([
                RandomCrop(patch_size, patch_size),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                RandomRotate90(p=0.5),
                _norm,
                er.preprocess.albu.ToTensor(),
            ]),
            CV=dict(k=10, i=-1),
            distributed=True,
            training=True,
            batch_size=8,
            num_workers=8,
        ),
    ),
    val=dict(
        type='WhuM3DALoader',
        params=dict(
            root=root_path, split='test', patch=patch_size,
            seed=split_seed, train_frac=train_frac, cache_ram=True,
            transforms=Compose([_norm, er.preprocess.albu.ToTensor()]),
            CV=dict(k=10, i=-1),
            training=False,
            batch_size=8,
            num_workers=8,
        ),
    ),
    mmr=dict(
        type='WhuM3DALoader',
        params=dict(
            root=root_path, split='train', patch=patch_size,
            seed=split_seed, train_frac=train_frac, cache_ram=True,
            transforms=Compose([_norm, er.preprocess.albu.ToTensor()]),
            CV=dict(k=10, i=-1),
            distributed=False,
            training=False,
            batch_size=4,
            num_workers=8,
        ),
    ),
    test=dict(
        type='WhuM3DALoader',
        params=dict(
            root=root_path, split='test', patch=patch_size,
            seed=split_seed, train_frac=train_frac, cache_ram=True,
            transforms=Compose([_norm, er.preprocess.albu.ToTensor()]),
            CV=dict(k=10, i=-1),
            distributed=True,
            training=False,
            batch_size=16,
            num_workers=8,
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
    save_ckpt_interval_epoch=20,
    eval_interval_epoch=20,
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
