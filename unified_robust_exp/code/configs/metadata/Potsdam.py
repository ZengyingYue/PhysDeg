"""ISPRS Potsdam metadata for the EarthM3-layout loader.

Patches produced by tools/prep_potsdam.py: nDSM(1ch)=SAR slot, RGB(3ch).
Channel stats computed over the 24 train tiles (3456 patches).
6 classes (stored value class+1; EarthM3 does mask-1):
  0 Impervious surfaces · 1 Building · 2 Low vegetation · 3 Tree · 4 Car · 5 Clutter
Official split: 24 train tiles / 14 test tiles. No separate val -> val=test
(checkpoint selection on the test set; standard for the Potsdam benchmark).
"""
sar_mean_v = [46.4467]                      # nDSM
sar_std_v = [55.1559]
rgb_mean_v = [85.8011, 91.7143, 84.9305]
rgb_std_v = [35.8229, 35.1507, 36.5273]

_root = "/home/yue/experiments/earthmiss_baseline/unified_robust_exp/datasets/potsdam_proc/patches"

train_image_dir = [f"{_root}/train/images"]
train_mask_dir = [f"{_root}/train/masks"]
val_image_dir = [f"{_root}/test/images"]
val_mask_dir = [f"{_root}/test/masks"]
test_image_dir = [f"{_root}/test/images"]
test_mask_dir = [f"{_root}/test/masks"]
