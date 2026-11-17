"""Shared pieces of the revision (protocol-v1) configs.

Every method trained for the revision uses the data blocks built here, so the
split, augmentation, batch size, schedule and checkpoint-selection rule are
identical across methods.  Conventions:

- ``data['train']``        : training split.
- ``data['test']``         : the VALIDATION split.  The ever trainer evaluates
                             ``data.test`` during training and keeps Best.pth on
                             it, so pointing it at validation keeps the test set
                             out of model selection.
- ``data['heldout_test']`` : the real test split.  Only
                             ``experiments/revision/evaluate.py`` reads it, after
                             training and post-hoc calibration are frozen.
- ``data['mmr']``          : training split without augmentation (MetaRS
                             whitening statistics / reliability calibration).

Checkpoint selection: mean validation mIoU over clean (fuse), missing-optical
(sar) and missing-SAR (rgb) readouts (``train['select_modes']``).
"""
from __future__ import annotations

import copy
import os

import cv2
from albumentations import (
    Compose, HorizontalFlip, Normalize, PadIfNeeded, RandomCrop,
    RandomRotate90, RandomScale, VerticalFlip,
)
import ever as er

from data.WHU import WhuM3DALoader  # noqa: F401 - registers the loader
from data.EarthMiss import EarthM3DALoader  # noqa: F401 - registers the loader
from data.Sen12msCR import Sen12msCRLoader  # noqa: F401 - registers the loader

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
WHU_ROOT = os.environ.get("WHU_ROOT", os.path.join(REPO, "datasets_raw", "whu-opt-sar"))
EM_ROOT = os.environ.get("EARTHMISS_ROOT", os.path.join(REPO, "datasets_raw", "EarthMiss"))
S12CR_ROOT = os.environ.get(
    "S12CR_ROOT", os.path.join(REPO, "unified_robust_exp", "datasets", "sen12mscr_winter"))


def _s12cr_stats():
    path = os.path.join(S12CR_ROOT, "stats.json")
    if not os.path.exists(path):  # before tools/prep_sen12mscr.py has run
        return dict(sar_mean=[0.0, 0.0], sar_std=[1.0, 1.0], rgb_mean=[0.0] * 3, rgb_std=[1.0] * 3)
    import json
    return json.load(open(path))


DFC_HR_ROOT = os.environ.get(
    "DFC_HR_ROOT", os.path.join(REPO, "unified_robust_exp", "datasets", "dfc2020_hr"))
OEM_ROOT = os.environ.get(
    "OEM_V1_ROOT", os.path.join(REPO, "unified_robust_exp", "datasets", "oem_sar_proc", "patches_v1"))

SEEDS = (2333, 42, 1234)

DATASETS = {
    "whu": dict(
        kind="whu", split_mode="patch", num_classes=7, patch=512, cloud_raw=235.0,
        sar_mean=[52.702], sar_std=[48.917],
        rgb_mean=[69.481, 87.825, 96.323], rgb_std=[28.228, 26.235, 23.927],
        class_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0),
        iters=40000,
    ),
    "whureg": dict(
        kind="whu", split_mode="region", num_classes=7, patch=512, cloud_raw=235.0,
        sar_mean=[52.702], sar_std=[48.917],
        rgb_mean=[69.481, 87.825, 96.323], rgb_std=[28.228, 26.235, 23.927],
        class_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0),
        iters=40000,
    ),
    "whupad": dict(  # official PAD image-level split (20 test images); 10 of the 80 train images = validation
        kind="whu", split_mode="pad", num_classes=7, patch=512, cloud_raw=235.0,
        sar_mean=[52.702], sar_std=[48.917],
        rgb_mean=[69.481, 87.825, 96.323], rgb_std=[28.228, 26.235, 23.927],
        class_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0),
        iters=40000,
    ),
    "em": dict(
        kind="em", num_classes=8, patch=512,
        sar_mean=[63.30051921735858], sar_std=[68.20405016],
        rgb_mean=[111.97261949846957, 118.42392668739762, 115.031314378774],
        rgb_std=[72.97088063, 72.98526569, 81.91774359],
        # background, building, road, water, barren, forest, agricultural, playground
        class_weights=None,
        cities=dict(
            train=["America-Eugene", "America-Louisville", "French-Paris",
                   "Morocco-Casablanca", "Nanjing", "Netherlands-Rotterdam", "Singapore"],
            val=["Australia-PortHedland", "America-Pake", "Russian-Engels"],
            test=["America-NewYork", "Japan-Hakodate", "Peru-Callao"],
        ),
        # 0.6 m (EarthMiss release / Zhou et al. 2026).  Until 2026-11-08 this key was missing and
        # the cloud-shadow offset used the 5 m default; those runs are kept as em_moddrop_gp25gsd5.
        gsd=0.6,
        iters=20000,
    ),
    # SEN12MS-CR winter: Sentinel-1 VV/VH + Sentinel-2 RGB, MODIS-derived labels
    # (DFC2020 8 classes); whole-ROI split; real clouds in the cloudy test variant.
    "s12cr": dict(
        kind="s12cr", num_classes=8, patch=256, sar_channels=2, cloud_raw=5500.0,
        # radiometry for training degradations: reflectance x 1e4, backscatter in dB, 10 m
        max_dn=10000.0, sar_domain="db", gsd=10.0,
        sar_mean=_s12cr_stats()["sar_mean"], sar_std=_s12cr_stats()["sar_std"],
        rgb_mean=_s12cr_stats()["rgb_mean"], rgb_std=_s12cr_stats()["rgb_std"],
        class_weights=None,
        iters=20000,
    ),
    "oem": dict(
        kind="em", num_classes=8, patch=512,
        sar_mean=[49.4736], sar_std=[34.8493],
        rgb_mean=[125.7341, 129.4400, 116.1242], rgb_std=[55.0530, 46.0364, 44.8725],
        class_weights=None,
        iters=40000,
    ),
}


def cloud_value_norm(ds):
    """Synthetic-cloud brightness in normalised optical units (per band)."""
    raw = ds.get("cloud_raw", 235.0)  # 8-bit imagery unless stated
    return [(raw - m) / sd for m, sd in zip(ds["rgb_mean"], ds["rgb_std"])]


def _normalize(ds):
    return Normalize(mean=ds["sar_mean"] + ds["rgb_mean"], std=ds["sar_std"] + ds["rgb_std"],
                     max_pixel_value=1)


def train_transforms(ds):
    """Paired geometric augmentation shared by every method (no photometric jitter)."""
    p = ds["patch"]
    return Compose([
        RandomScale(scale_limit=(-0.25, 0.5), p=0.5),
        PadIfNeeded(min_height=p, min_width=p, border_mode=cv2.BORDER_CONSTANT,
                    fill=0, fill_mask=-1),
        RandomCrop(p, p),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        _normalize(ds),
        er.preprocess.albu.ToTensor(),
    ])


def eval_transforms(ds):
    return Compose([_normalize(ds), er.preprocess.albu.ToTensor()])


def _whu_loader(ds, split, training, transforms, batch_size, num_workers, distributed):
    return dict(type="WhuM3DALoader", params=dict(
        root=WHU_ROOT, split=split, patch=ds["patch"], seed=42, train_frac=0.8,
        split_mode=ds["split_mode"], val_count=560 if ds["split_mode"] == "patch" else 0,
        val_seed=4242, region_counts=(70, 10, 20), cache_ram=True,
        transforms=transforms, CV=dict(k=10, i=-1), distributed=distributed,
        training=training, batch_size=batch_size, num_workers=num_workers))


def _em_dirs(name, split):
    if name == "em":
        cities = DATASETS["em"]["cities"][split]
        return ([os.path.join(EM_ROOT, c, "images") for c in cities],
                [os.path.join(EM_ROOT, c, "masks") for c in cities])
    return [os.path.join(OEM_ROOT, split, "images")], [os.path.join(OEM_ROOT, split, "masks")]


def _em_loader(name, split, training, transforms, batch_size, num_workers, distributed):
    image_dir, mask_dir = _em_dirs(name, split)
    return dict(type="EarthM3DALoader", params=dict(
        image_dir=image_dir, mask_dir=mask_dir, sensors=("SAR", "RGB"),
        transforms=transforms, CV=dict(k=10, i=-1), distributed=distributed,
        training=training, batch_size=batch_size, num_workers=num_workers))


def _s12cr_loader(ds, split, training, transforms, batch_size, num_workers, distributed,
                  variant="clear"):
    return dict(type="Sen12msCRLoader", params=dict(
        root=S12CR_ROOT, split=split, variant=variant, transforms=transforms,
        distributed=distributed, training=training, batch_size=batch_size,
        num_workers=num_workers))


def data_block(name):
    ds = DATASETS[name]
    if ds["kind"] == "s12cr":
        return dict(
            train=_s12cr_loader(ds, "train", True, train_transforms(ds), 16, 8, True),
            test=_s12cr_loader(ds, "val", False, eval_transforms(ds), 8, 2, True),
            heldout_test=_s12cr_loader(ds, "test", False, eval_transforms(ds), 8, 0, False),
            heldout_cloudy=_s12cr_loader(ds, "test", False, eval_transforms(ds), 8, 0, False,
                                         variant="cloudy"),
            # validation ROIs with their real cloudy acquisitions (method selection)
            val_clear=_s12cr_loader(ds, "val", False, eval_transforms(ds), 8, 0, False),
            val_cloudy=_s12cr_loader(ds, "val", False, eval_transforms(ds), 8, 0, False,
                                     variant="cloudy"),
            mmr=_s12cr_loader(ds, "train", False, eval_transforms(ds), 4, 2, False),
            # DFC2020 with the official 10 m labels (same sensors): clean test sets
            dfc_hr_validation=dict(type="Sen12msCRLoader", params=dict(
                root=DFC_HR_ROOT, split="validation", variant="clear",
                transforms=eval_transforms(ds), distributed=False, training=False,
                batch_size=8, num_workers=0)),
            dfc_hr_test=dict(type="Sen12msCRLoader", params=dict(
                root=DFC_HR_ROOT, split="test", variant="clear",
                transforms=eval_transforms(ds), distributed=False, training=False,
                batch_size=8, num_workers=0)),
        )
    if ds["kind"] == "whu":
        def make(*args):
            return _whu_loader(ds, *args)
    else:
        def make(*args):
            return _em_loader(name, *args)
    return dict(
        train=make("train", True, train_transforms(ds), 16, 8, True),
        test=make("val", False, eval_transforms(ds), 8, 2, True),
        heldout_test=make("test", False, eval_transforms(ds), 8, 0, False),
        mmr=make("train", False, eval_transforms(ds), 4, 2, False),
    )


def schedule(name):
    iters = DATASETS[name]["iters"]
    optimizer = dict(type="adamw", params=dict(lr=2e-4, betas=(0.9, 0.999), weight_decay=0.01))
    learning_rate = dict(type="poly", params=dict(base_lr=2e-4, power=0.9, max_iters=iters))
    train = dict(
        forward_times=1, num_iters=iters, eval_per_epoch=True, summary_grads=False,
        summary_weights=False, distributed=True, apex_sync_bn=True, sync_bn=True,
        eval_after_train=True, log_interval_step=50,
        save_ckpt_interval_epoch=4, eval_interval_epoch=4, distributed_evaluate=True,
        select_modes=["fuse", "sar", "rgb"],
        best_checkpoint=dict(enabled=True, metric="miou", filename="Best.pth",
                             initial_score=float("-inf"), delete_old_checkpoints=True),
    )
    return optimizer, learning_rate, train


def assemble(model, name):
    """Wrap a model block with the dataset's data/schedule blocks."""
    optimizer, learning_rate, train = schedule(name)
    data = data_block(name)
    model = copy.deepcopy(model)
    return dict(model=model, data=data, optimizer=optimizer, learning_rate=learning_rate,
                train=train, test=dict())
