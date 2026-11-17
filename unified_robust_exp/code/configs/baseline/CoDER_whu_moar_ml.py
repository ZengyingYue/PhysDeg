"""Phase C — push CoDER clean to >=0.57 by aligning the fused-path architecture
to CMNeXtSGSLN's clean-SOTA (0.5779) design.

= baseline.CoDER_whu_moar (full MoAR+DMFIA, top_k=1) with a SINGLE change:
  - fused_head='multilevel'  : swap CoDER's FPN+AssymetricDecoder fused head for
    CMNeXtSGSLN's SegFormer-style MultiLevelDecoder (the head used in the 0.5779
    WHU run), + fused_head_dropout=0.1 to match that run's decoder_dropout.

top_k stays 1 — VERIFIED that the 0.5779 WHU run used moa_top_k=1 (config dump in
cloud_server/whuoptsar_exp/outputs/logs/train.log), and a prior comparison found
top_k=1 best. top_k is deferred to a later experiment.

ONLY the fused/clean head changes (decoder_sar untouched) ⇒ SAR-only is
architecturally protected; verify empirically (eval sar) each run. This isolates
ONE variable: does CMNeXtSGSLN's decode head close the clean gap?
Targets: clean(fuse) -> >=0.57 (toward 0.5779), SAR-only >= 0.4933.
"""
import copy

from configs.baseline.CoDER_whu_moar import config as _base

config = copy.deepcopy(_base)
_p = config["model"]["params"]
_p["fused_head"] = "multilevel"
_p["fused_head_dropout"] = 0.1
# moar_top_k stays 1 (inherited) — see note above.
