"""EarthMiss MoAR-validation arm.

EXACT copy of `CoDER_flow_cb.py` (the convnext_base + latent-flow config that
gives the EarthMiss SAR-only baseline ~0.309) with the SINGLE change of swapping
the fused-path module to the grafted **MoAR/DMFIA** fusion (`fusion_type='moar'`).

Purpose: validate cross-dataset that the WHU MoAR gain (clean +0.91 / SAR-only
+0.33, on the leaky patch split) is real on EarthMiss (clean held-out split),
and that SAR-only stays >= 0.31. NOTE: with missing_readout='dempster' +
infer_mode='sar', the SAR-only readout does NOT pass through the fused module,
so MoAR affects SAR-only only INDIRECTLY (better shared-encoder training via the
fused-path auxiliary losses) — the same mechanism as the WHU +0.33.
"""
from copy import deepcopy

from configs.baseline.CoDER_flow_cb import config as _base

config = deepcopy(_base)
_p = config["model"]["params"]
_p["fusion_type"] = "moar"     # CoDER.py switch: builds MoARFpnFuse, forces use_moe=True
_p["moar_dmfia_units"] = 1
_p["moar_num_experts"] = 3
_p["moar_top_k"] = 1
