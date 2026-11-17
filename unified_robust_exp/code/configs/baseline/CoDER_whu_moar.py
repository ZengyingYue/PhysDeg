"""Step-2: CoDER on WHU clean with the MoAR+DMFIA fused path (clean-SOTA graft).

Identical to baseline.CoDER_whu except the fused-path fusion module is swapped
FpnMoE -> MoARFpnFuse (CMNeXtSGSLN's clean recipe: 1 DMFIA unit, 3 experts,
top_k=1).  Everything else — encoders, decoder_fused, decoder_sar/rgb, Project,
EDL, Dempster, missing-modality, MMR — is unchanged, so the SAR-only path is
architecturally untouched (verified empirically by evaluating sar mode).
"""
import copy

from configs.baseline.CoDER_whu import config as _base

config = copy.deepcopy(_base)
_p = config["model"]["params"]
_p["fusion_type"] = "moar"
_p["moar_dmfia_units"] = 1
_p["moar_num_experts"] = 3
_p["moar_top_k"] = 1
