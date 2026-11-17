"""OpenEarthMap-SAR (DFC25 Track 1) metadata for the EarthM3 layout.

Built by tools/prep_oem_sar.py: the official labelled train set is tiled
1024->512 and split seed-42 at the SOURCE-IMAGE level 90/10 into
patches/{train,heldout}, both carrying real SAR+RGB+labels. heldout is the
eval set (clean / RGB-degradation ladder / SAR-only floor) -- the official
val/test are SAR-only so cannot drive the RGB ladder.

8 land-cover classes (stored 1..8; EarthM3 maps mask-1 -> 0..7, 0 -> -1 ignore).

Channel stats below are over the TRAIN patches (printed by prep_oem_sar.py).
"""
import os

# === filled from tools/prep_oem_sar.py full-run output (train: 15600 patches) ===
sar_mean_v = [49.4736]
sar_std_v = [34.8493]
rgb_mean_v = [125.7341, 129.4400, 116.1242]
rgb_std_v = [55.0530, 46.0364, 44.8725]
# ===============================================================================

root_path = "/home/yue/experiments/earthmiss_baseline/unified_robust_exp/datasets/oem_sar_proc/patches"

num_classes = 8
patch_size = 512

# EarthM3 layout: image_dir/{SAR,RGB} + masks (sibling). Lists, as EarthM3DALoader
# accepts a list of dirs.
train_image_dir = [os.path.join(root_path, "train", "images")]
train_mask_dir = [os.path.join(root_path, "train", "masks")]

# heldout serves as both val (best-ckpt selection) and test (final eval).
val_image_dir = [os.path.join(root_path, "heldout", "images")]
val_mask_dir = [os.path.join(root_path, "heldout", "masks")]

test_image_dir = [os.path.join(root_path, "heldout", "images")]
test_mask_dir = [os.path.join(root_path, "heldout", "masks")]
