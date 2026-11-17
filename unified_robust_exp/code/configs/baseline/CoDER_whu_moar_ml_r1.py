"""Phase C / s4 — RECIPE-aligned run to chase clean >=0.57.

= baseline.CoDER_whu_moar_ml (s3a: full MoAR+DMFIA + MultiLevelDecoder fused
  head, top_k=1) with the TRAINING RECIPE aligned to the 0.5779 CMNeXtSGSLN run.

Diagnosis (2026-06-17): after the decode-head graft (s3a), the residual ~1.5pt
gap to CMNeXtSGSLN 0.5779 is RECIPE, not architecture. s3a trained only 42 epochs
(bs8 x 30k) vs the reference's 120; no label smoothing; bs8 vs 16; lr1e-4 vs 2e-4.

This config carries the ONE code-level recipe change that needs a config key:
  - loss.label_smoothing = 0.05   (matches the 0.5779 run; CoDER default 0.0)

The rest of the recipe is passed as launch overrides (run_s4.sh), so this file
stays a thin, auditable diff:
  --mixed_precision bf16          (GPU efficiency: ~2x speed, ~half memory ->
                                   enables native bs16; bf16 is numerically safe
                                   here — same exponent range as fp32, autocast
                                   keeps softmax/log/linalg/loss in fp32, master
                                   weights fp32. Validate early-step loss is finite.)
  batch_size 16                   (recipe; fits once bf16 halves activations)
  base_lr 2e-4                    (recipe; paired with bs16)
  num_iters 40000  (== ~114 ep @ bs16, matching the reference's 120 ep)
  + missing_modality_total_iterations 40000

Only the fused/clean path + global recipe change; decoder_sar untouched =>
SAR-only architecturally protected; verify empirically (eval sar) each run.
Targets: clean(fuse) -> >=0.57 (toward 0.5779), SAR-only >= 0.4933.

NOTE on precision confound (paper): baselines (s1c/s2b/pa_d0) are fp32. If s4
lands borderline (0.568-0.572), re-run the winning config in fp32 to lock the
headline number; a clear >=0.57 or <0.57 verdict is robust to the bf16 ~±0.2 noise.
"""
import copy

from configs.baseline.CoDER_whu_moar_ml import config as _base

config = copy.deepcopy(_base)
config["model"]["params"]["loss"]["label_smoothing"] = 0.05
