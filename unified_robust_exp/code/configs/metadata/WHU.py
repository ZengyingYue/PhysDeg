"""WHU-OPT-SAR metadata (PAD protocol, Zheng et al. TGRS 2025).

Channel order matches the EarthMiss pipeline: the dataset stacks
[SAR(1), R, G, B] so the Normalize means/stds are `sar_*_v + rgb_*_v`.
RGB stats are in R,G,B order (optical bands [3,2,1] of GF-1 [B,G,R,NIR],
NIR dropped).  7 land-cover classes; label 0 (background) is ignore.
"""
import os

# SAR (1-band GF-3) then RGB (R,G,B) — raw uint8 value statistics.
sar_mean_v = [52.702]
sar_std_v = [48.917]
rgb_mean_v = [69.481, 87.825, 96.323]
rgb_std_v = [28.228, 26.235, 23.927]

root_path = os.environ.get(
    "WHU_ROOT",
    "/home/yue/experiment/earthmiss_baseline/datasets_raw/whu-opt-sar")

WHU_CLASSES = ["Farmland", "City", "Village", "Water", "Forest", "Road", "Others"]
num_classes = 7

# PAD split protocol
patch_size = 512
split_seed = 42
train_frac = 0.8
