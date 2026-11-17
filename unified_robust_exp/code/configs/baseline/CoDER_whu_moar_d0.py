"""Phase A — capacity/design decomposition for the MoAR clean gain.

Identical to baseline.CoDER_whu_moar EXCEPT `moar_dmfia_units = 0`: drops the
215M-param DMFIA refinement, keeping ONLY the cheap MoAR rectify+fuse blocks
(~55M for experts=3). Total model ≈ 155M — LESS than the FpnMoE baseline (s1c,
174M, 0.5534) and far less than full MoAR (s2b, 371M, 0.5625).

Decisive single run:
  - vs s1c (FpnMoE 174M): if this (~155M) still > 0.5534 ⇒ the gain is the MoAR
    fusion DESIGN, not capacity (fewer params, higher mIoU) ⇒ drop DMFIA.
  - vs s2b (full MoAR 371M): the gap = DMFIA's marginal value for its 215M.
"""
import copy

from configs.baseline.CoDER_whu_moar import config as _base

config = copy.deepcopy(_base)
config["model"]["params"]["moar_dmfia_units"] = 0
