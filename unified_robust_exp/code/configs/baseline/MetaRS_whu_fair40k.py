"""MetaRS WHU configuration for the unified 40k fair-comparison protocol.

Architecture and method-specific losses are unchanged from MetaRS_whu.  The
formal runner supplies the common 40k iterations, batch size 16, learning rate,
augmentation, evaluation cadence, seed, and checkpoint rule used by ReBound.
Only the loader is made workstation-safe here: no RAM preload and no worker
forks.
"""
from copy import deepcopy

from configs.baseline.MetaRS_whu import config as _base


config = deepcopy(_base)
for split in ("train", "val", "mmr", "test"):
    params = config["data"][split]["params"]
    params["cache_ram"] = False
    params["num_workers"] = 0

# MetaRS receives this block through model.params.data for its train-only MMR
# statistics pass.  Keep it identical to the safe top-level MMR loader.
config["model"]["params"]["data"] = config["data"]["mmr"]

