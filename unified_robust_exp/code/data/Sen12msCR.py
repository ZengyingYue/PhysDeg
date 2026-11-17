"""SEN12MS-CR (winter) loader over the arrays written by tools/prep_sen12mscr.py.

Returns the same contract as the WHU / EarthMiss loaders:
``(image HWC float32 [VV, VH, R, G, B], dict(cls=label, fname=id))`` with
label 255 mapped to the ignore index -1.  ``variant='cloudy'`` swaps in the
cloudy Sentinel-2 acquisition of the same location (real clouds).
"""
from __future__ import annotations

import os

import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

import ever as er
from ever.data import distributed
from ever.interface import ConfigurableMixin


class Sen12msCR(Dataset):
    def __init__(self, root, split="train", variant="clear", transforms=None):
        if variant not in ("clear", "cloudy"):
            raise ValueError(variant)
        self.dir = os.path.join(root, split)
        self.variant = variant
        self.transforms = transforms
        self.ids = open(os.path.join(self.dir, "ids.txt")).read().split()
        self._arrays = None

    def _open(self):
        # Opened lazily so that forked data-loader workers share the page cache.
        rgb = "rgb_cloudy" if self.variant == "cloudy" else "rgb"
        names = [("s1", "s1"), ("rgb", rgb), ("label", "label"), ("cloud", "cloud")]
        # cloud.npy exists for SEN12MS-CR only (DFC2020 has no cloudy acquisition)
        self._arrays = {k: np.load(os.path.join(self.dir, f"{v}.npy"), mmap_mode="r")
                        for k, v in names if os.path.exists(os.path.join(self.dir, f"{v}.npy"))}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        if self._arrays is None:
            self._open()
        a = self._arrays
        image = np.concatenate([a["s1"][i].astype(np.float32), a["rgb"][i].astype(np.float32)], 0)
        image = np.ascontiguousarray(image.transpose(1, 2, 0))
        mask = a["label"][i].astype(np.int64)
        mask[mask == 255] = -1
        if self.transforms is not None:
            blob = self.transforms(image=image, mask=mask)
            image, mask = blob["image"], blob["mask"]
        return image, dict(cls=mask, fname=self.ids[i])

    def cloud_mask(self, i):
        if self._arrays is None:
            self._open()
        return np.asarray(self._arrays["cloud"][i])


@er.registry.DATALOADER.register()
class Sen12msCRLoader(DataLoader, ConfigurableMixin):
    def __init__(self, config):
        ConfigurableMixin.__init__(self, config)
        dataset = Sen12msCR(self.config.root, self.config.split, self.config.variant,
                            self.config.transforms)
        if getattr(self.config, "distributed", True):
            sampler = (distributed.StepDistributedSampler(dataset) if self.config.training
                       else distributed.DistributedNonOverlapSeqSampler(dataset))
        else:
            sampler = RandomSampler(dataset) if self.config.training else SequentialSampler(dataset)
        super().__init__(dataset, self.config.batch_size, sampler=sampler,
                         num_workers=self.config.num_workers, pin_memory=True, drop_last=False)

    def set_default_config(self):
        self.config.update(dict(root=None, split="train", variant="clear", transforms=None,
                                batch_size=8, num_workers=4, distributed=True, training=True))
