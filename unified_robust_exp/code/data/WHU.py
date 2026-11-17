"""WHU-OPT-SAR data loader for CoDER, matched to the PAD protocol
(Zheng et al., TGRS 2025, https://github.com/RanFeng2/PAD).

- 100 co-registered optical(4-band GF-1) / SAR(1-band GF-3) / label triples
  (5556x3704).  Each image is cut into NON-overlapping 512x512 patches (edges
  dropped) = floor(3704/512)=7 x floor(5556/512)=10 = 70/image x 100 = 7000.
- 7000 patches shuffled (seed 42) and split 80/20 train/test.
- 7 classes (Farmland/City/Village/Water/Forest/Road/Others); background
  (raw label 0) is the ignore class.  Raw labels are {0,10,...,70} -> //10-1
  giving {ignore(-1), 0..6}.
- Optical band order is GF-1 [B,G,R,NIR]; we feed R,G,B (rasterio bands
  [3,2,1]) and drop NIR.  The model input is stacked [SAR, R, G, B] (4ch) to
  match the EarthMiss pipeline (Normalize means = sar_mean_v + rgb_mean_v).

Output matches `data.EarthMiss.EarthM3` so CoDER's train/eval contract is
unchanged: `(ms_images_tensor(4,H,W), dict(cls=mask, fname=...))`.  The
`mask_down`/`points`/`mask_id` fields EarthM3 produces are NOT included
because no CoDER code path consumes them (verified) — skipping them keeps the
CPU loader light.

RAM cache: all 100 full images (native uint8, RGB 3-band only) ~= 10.3 GB are
preloaded ONCE in the main process; DataLoader workers inherit the cache
copy-on-write on fork, so it is shared, not duplicated per worker.  This
removes the slow per-patch windowed reads off the Windows drvfs mount
(measured ~130x faster patch fetch).
"""
from __future__ import annotations

import glob
import logging
import os

import numpy as np
import rasterio
import torch
from albumentations import Compose, Normalize
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

import ever as er
from ever.data import CrossValSamplerGenerator, distributed
from ever.interface import ConfigurableMixin

logger = logging.getLogger(__name__)

WHU_CLASSES = ["Farmland", "City", "Village", "Water", "Forest", "Road", "Others"]
WHU_NUM_CLASSES = 7
WHU_IGNORE_INDEX = -1

# Module-level shared store: optical-path -> (rgb uint8 (3,H,W) R,G,B,
# sar uint8 (H,W), lbl uint8 (H,W)).  Populated once in the MAIN process.
_RAM_CACHE: dict = {}


# Decoded uint8 arrays written by tools/build_whu_npy_cache.py.  When present
# they are memory-mapped, so all processes share one page-cache copy.
NPY_DIR = os.environ.get("WHU_NPY_CACHE", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "datasets", "whu_npy"))


def _npy_paths(of):
    stem = os.path.splitext(os.path.basename(of))[0]
    return [os.path.join(NPY_DIR, f"{stem}_{name}.npy") for name in ("rgb", "sar", "lbl")]


def _npy_available(opt_files):
    return all(os.path.exists(p) for of in opt_files for p in _npy_paths(of))


def _preload_ram(opt_files, sar_files, lbl_files):
    use_npy = _npy_available(opt_files)
    for of, sf, lf in zip(opt_files, sar_files, lbl_files):
        if of in _RAM_CACHE:
            continue
        if use_npy:
            _RAM_CACHE[of] = tuple(np.load(p, mmap_mode="r") for p in _npy_paths(of))
            continue
        with rasterio.open(of) as ds:
            opt = ds.read()                       # (4,H,W) GF-1 [B,G,R,NIR]
        rgb = np.ascontiguousarray(opt[[2, 1, 0]])  # -> R,G,B
        with rasterio.open(sf) as ds:
            sar = np.ascontiguousarray(ds.read(1))  # (H,W)
        with rasterio.open(lf) as ds:
            lbl = np.ascontiguousarray(ds.read(1))  # (H,W)
        _RAM_CACHE[of] = (rgb, sar, lbl)


def _build_patch_index(root, patch, seed, train_frac):
    opt_files = sorted(glob.glob(os.path.join(root, "optical", "*.tif")))
    if not opt_files:
        raise FileNotFoundError(f"no optical tifs under {root}/optical")
    patches = []
    for fi, of in enumerate(opt_files):
        with rasterio.open(of) as ds:
            h, w = ds.height, ds.width
        for r in range(0, h - patch + 1, patch):
            for c in range(0, w - patch + 1, patch):
                patches.append((fi, r, c))
    order = np.random.RandomState(seed).permutation(len(patches))
    cut = int(round(len(patches) * train_frac))
    return opt_files, patches, order[:cut], order[cut:]


def split_indices(root, patch=512, seed=42, train_frac=0.8, split_mode="patch",
                  val_count=0, val_seed=4242, region_counts=(70, 10, 20)):
    """Return ``(opt_files, patches, {"train","val","test"} -> index array)``.

    ``split_mode='patch'`` keeps the PAD protocol: the seed-42 test patches are
    unchanged, and ``val_count`` patches are carved from the train portion with
    an independent ``val_seed`` (``val_count=0`` reproduces the legacy 80/20
    split with an empty validation set).

    ``split_mode='region'`` assigns whole source images to train/val/test
    (``region_counts`` images each, permuted with ``seed``), so no scene
    contributes patches to more than one split.
    """
    opt_files, patches, train_idx, test_idx = _build_patch_index(
        root, patch, seed, train_frac)
    if split_mode == "patch":
        if val_count:
            perm = np.random.RandomState(val_seed).permutation(len(train_idx))
            val_idx = train_idx[perm[:val_count]]
            train_idx = train_idx[perm[val_count:]]
        else:
            val_idx = train_idx[:0]
        return opt_files, patches, dict(train=train_idx, val=val_idx, test=test_idx)
    if split_mode == "region":
        n_train, n_val, n_test = region_counts
        if n_train + n_val + n_test != len(opt_files):
            raise ValueError(f"region_counts {region_counts} != {len(opt_files)} images")
        image_order = np.random.RandomState(seed).permutation(len(opt_files))
        owner = np.empty(len(opt_files), dtype=object)
        owner[image_order[:n_train]] = "train"
        owner[image_order[n_train:n_train + n_val]] = "val"
        owner[image_order[n_train + n_val:]] = "test"
        split = {name: np.array([i for i, (fi, _, _) in enumerate(patches)
                                 if owner[fi] == name], dtype=np.int64)
                 for name in ("train", "val", "test")}
        return opt_files, patches, split
    if split_mode == "pad":
        # Official PAD split (RanFeng2/PAD, splited_files/whuoptsar_{train,test}.txt):
        # 20 whole source images are the test set and 80 are training images.
        # PAD has no validation set, so ``region_counts[1]`` training images are
        # carved out as validation with ``val_seed``; the test images are untouched.
        pad_dir = os.environ.get("WHU_PAD_SPLIT_DIR", os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "datasets", "pad_split"))
        read = lambda n: [l.strip() for l in open(os.path.join(pad_dir, n)) if l.strip()]
        pad_train, pad_test = read("whuoptsar_train.txt"), read("whuoptsar_test.txt")
        names = [os.path.splitext(os.path.basename(f))[0] for f in opt_files]
        missing = (set(pad_train) | set(pad_test)) - set(names)
        if missing or len(pad_train) + len(pad_test) != len(names):
            raise ValueError(f"PAD split does not match the images under {root}: missing {sorted(missing)[:5]}")
        n_val = region_counts[1]
        perm = np.random.RandomState(val_seed).permutation(len(pad_train))
        val_names = set(pad_train[i] for i in perm[:n_val])
        owner = {n: "test" for n in pad_test}
        owner.update({n: ("val" if n in val_names else "train") for n in pad_train})
        split = {name: np.array([i for i, (fi, _, _) in enumerate(patches)
                                 if owner[names[fi]] == name], dtype=np.int64)
                 for name in ("train", "val", "test")}
        return opt_files, patches, split
    raise ValueError(f"unknown split_mode {split_mode!r}")


class WhuM3(Dataset):
    def __init__(self, root, split="train", patch=512, seed=42, train_frac=0.8,
                 transforms=None, cache_ram=True, split_mode="patch", val_count=0,
                 val_seed=4242, region_counts=(70, 10, 20)):
        if split not in {"train", "val", "test"}:
            raise ValueError(split)
        self.patch = patch
        self.transforms = transforms
        self.cache_ram = cache_ram
        opt_files, patches, split_idx = split_indices(
            root, patch, seed, train_frac, split_mode, val_count, val_seed,
            tuple(region_counts))
        if len(split_idx[split]) == 0:
            raise ValueError(f"WHU split {split!r} is empty (split_mode={split_mode}, "
                             f"val_count={val_count})")
        self.opt_files = opt_files
        self.sar_files = [f.replace(os.sep + "optical" + os.sep, os.sep + "sar" + os.sep) for f in opt_files]
        self.lbl_files = [f.replace(os.sep + "optical" + os.sep, os.sep + "lbl" + os.sep) for f in opt_files]
        self.patches = patches
        self.indices = split_idx[split].tolist()
        if _npy_available(opt_files):
            cache_ram = self.cache_ram = True  # memory-mapped: shared, no per-process copy
        if cache_ram:
            _preload_ram(self.opt_files, self.sar_files, self.lbl_files)
        logger.info("WHU %s split: %d patches (cache_ram=%s)", split, len(self.indices), cache_ram)

    def __len__(self):
        return len(self.indices)

    def _read_patch(self, fi, r, c):
        p = self.patch
        if self.cache_ram:
            rgb_f, sar_f, lbl_f = _RAM_CACHE[self.opt_files[fi]]
            rgb = rgb_f[:, r:r + p, c:c + p]               # (3,P,P) R,G,B uint8
            sar = sar_f[r:r + p, c:c + p]                   # (P,P)
            lbl = lbl_f[r:r + p, c:c + p]                   # (P,P)
        else:
            win = Window(c, r, p, p)
            with rasterio.open(self.opt_files[fi]) as ds:
                rgb = ds.read([3, 2, 1], window=win)        # (3,P,P) R,G,B
            with rasterio.open(self.sar_files[fi]) as ds:
                sar = ds.read(1, window=win)
            with rasterio.open(self.lbl_files[fi]) as ds:
                lbl = ds.read(1, window=win)
        return rgb, sar, lbl

    def __getitem__(self, i):
        fi, r, c = self.patches[self.indices[i]]
        rgb, sar, lbl = self._read_patch(fi, r, c)
        # ms_images HWC float32: channel order [SAR, R, G, B]
        ms = np.concatenate([sar[None], rgb], axis=0).astype(np.float32)  # (4,P,P)
        ms = np.ascontiguousarray(ms.transpose(1, 2, 0))                  # (P,P,4)
        # raw {0,10,...,70} -> {ignore(-1), 0..6}
        mask = (lbl.astype(np.int64) // 10) - 1
        if self.transforms is not None:
            blob = self.transforms(image=ms, mask=mask)
            ms, mask = blob["image"], blob["mask"]
        fname = f"{os.path.splitext(os.path.basename(self.opt_files[fi]))[0]}_{r}_{c}.tif"
        return ms, dict(cls=mask, fname=fname)


@er.registry.DATALOADER.register()
class WhuM3DALoader(DataLoader, ConfigurableMixin):
    def __init__(self, config):
        ConfigurableMixin.__init__(self, config)
        dataset = WhuM3(
            self.config.root, self.config.split, self.config.patch,
            self.config.seed, self.config.train_frac,
            self.config.transforms, self.config.cache_ram,
            self.config.split_mode, self.config.val_count,
            self.config.val_seed, self.config.region_counts,
        )
        if self.config.CV.i != -1:
            use_distributed = getattr(self.config, "distributed", True)
            CV = CrossValSamplerGenerator(dataset, distributed=use_distributed, seed=2333)
            sampler_pairs = CV.k_fold(self.config.CV.k)
            train_sampler, val_sampler = sampler_pairs[self.config.CV.i]
            sampler = train_sampler if self.config.training else val_sampler
        else:
            use_distributed = getattr(self.config, "distributed", True)
            if use_distributed:
                sampler = (distributed.StepDistributedSampler(dataset) if self.config.training
                           else distributed.DistributedNonOverlapSeqSampler(dataset))
            else:
                sampler = RandomSampler(dataset) if self.config.training else SequentialSampler(dataset)
        super(WhuM3DALoader, self).__init__(
            dataset, self.config.batch_size, sampler=sampler,
            num_workers=self.config.num_workers, pin_memory=True, drop_last=False,
        )

    def set_default_config(self):
        self.config.update(dict(
            root=None,
            split="train",
            patch=512,
            seed=42,
            train_frac=0.8,
            split_mode="patch",
            val_count=0,
            val_seed=4242,
            region_counts=(70, 10, 20),
            cache_ram=True,
            batch_size=8,
            num_workers=8,
            distributed=True,
            training=True,
            CV=dict(k=10, i=-1),
            transforms=Compose([
                Normalize(mean=(), std=(), max_pixel_value=1, always_apply=True),
                er.preprocess.albu.ToTensor(),
            ]),
        ))
