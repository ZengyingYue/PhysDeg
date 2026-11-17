"""CoDER v3 Option-2 — Plan A: Opt1 + L_mse(stages 3+4) + L_recon.

Plan A goal (audit-driven, 2026-06-05):
  Opt1 (ConvNeXt-base + MetaRS 3-decoder + CE only) plateaus at 0.32 mIoU
  because three of MetaRS's four algorithmic pillars are DISABLED:
      Pillar-1   L_mmr (cov whitening)              ← needs IBN (Plan B)
      Pillar-1   L_mse (stage 3+4 meta alignment)   ← enabled here
      Pillar-3   L_recon (missing-mode anchor)      ← enabled here

  ConvNeXt features are ~50× larger in magnitude than ResNet50-BN features,
  so the original MetaRS weights (alpha=1.25, lam_recon=0.5) over-dominated
  total loss.  Here we scale both down by ~25× to keep them in the same
  ballpark as the CE terms (each ~1.5 contribution).

  Loss budget (init magnitudes from dryrun, real-data scaling ~0.3× of noise):
      CE_fuse   ≈ 3.2  (lam=1.5 × CE_init≈2.1)
      CE_sar    ≈ 2.1
      CE_rgb    ≈ 1.1
      CE_miss   ≈ 2.1  (when missing-mode fires, 2/3 of steps)
      L_mse     ≈ 0.4-1.2 (lam_mse=0.2 × 0.1-warmup × meanMSE≈20-60)
      L_recon   ≈ 0.4-1.0 (lam_recon=0.02 × recon≈20-50; only on missing steps)
      EDL aux   ≈ 0.1-0.2 each × 5 streams = 0.5-1.0

  We expect each alignment loss to converge to ~0.2-0.5 contribution — same
  order as a per-pixel CE term, but never dominating. If the 5k probe shows
  alignment losses are <0.1 (under-utilised), scale lam up 2.5× next run.

  We expect L_mse + L_recon to add a ~+0.015 ~ +0.025 mIoU bump over Opt1
  (toward 0.33-0.35) by enforcing the meta-feature alignment the paper §4.3
  relies on for cross-modal knowledge transfer.

  If Plan A produces stable curves (loss decreases monotonically, mIoU
  trajectory monotone-or-bumpy with positive trend), Plan B will add:
      P3: IBN-style InstanceNorm in `en_content` (after stem + each stage)
      P4: Re-enable L_mmr on the new IN points

  Differences from Opt1:
    - `mse=dict()` present  →  meta-feature MSE branch fires
    - `lam_mse=0.5`         →  effective 0.05× (×0.1 warmup factor in code)
    - `alpha=1.0`           →  trade-off factor (only used post-MMR; here = no-op)
    - `lam_recon=0.05`      →  was 0.0

  Everything else stays Opt1 — no aug change, no class weights, no OHEM,
  EDL aux at low weight for ECE deliverable.
"""
from module.baseline.base import default_conv_block

# Reuse the original MetaRS-style data block (NO RandomScale).
from configs.base.EarthMiss import train, test, data, optimizer, learning_rate

config = dict(
    model=dict(
        type='CoDER',
        params=dict(
            backbone='convnext_base',
            sar_in_channels=1,
            rgb_in_channels=3,
            infer_rgb=False,
            infer_mode='sar',
            # Missing-modality training (MetaRS recipe)
            missing_modality_training=True,
            missing_modality_prob=dict(
                complete=1 / 3,
                sar_missing=1 / 3,
                rgb_missing=1 / 3,
            ),
            missing_modality_total_iterations=15000,
            # ConvNeXt-base output channels per stage (strides 4/8/16/32)
            in_channels_list=(0, 0, 128, 256, 512, 1024, 0),
            fuse_inchan_list=(128, 256, 512, 1024),
            relax_denom=0,
            clusters=10,
            # MMR disabled (no BN in ConvNeXt) — value kept for code-path compat
            begin_mmr_iter=15001,
            # β-discount disabled (no MMR signal)
            beta_scale=0.0,
            beta_floor=1.0,
            gamma_rec_init=1.0,
            beta_ramp_iters=0,
            # MoE disabled — revert to MetaRS-style Fusespfnv2
            use_moe=False,
            encoder=dict(
                resnet_type='resnet50',  # ignored; kept for old code paths
                include_conv5=True,
                batchnorm_trainable=True,
                pretrained=True,
                freeze_at=0,
                output_stride=32,
                wt_layer=[0, 0, 0, 0, 0, 0, 0],  # all zeros → no MMR
                num_classes=1,
            ),
            fpn=dict(
                in_channels_list=(256, 512, 1024, 2048),
                out_channels=256,
                conv_block=default_conv_block,
                top_blocks=None,
            ),
            decoder=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
            ),
            scale_factor=16,
            num_classes=8,
            loss=dict(
                # ===== Primary classification (CE on softmax) =====
                lam_ce_sar=1.0,
                lam_ce_rgb=0.5,
                lam_ce_fuse=1.5,
                lam_ce_missing=1.0,
                # ===== Auxiliary EDL (uncertainty path, low weight) =====
                evidence_activation='softplus',
                edl_form='mse',
                ce_aux_weight=0.0,
                kl_max_lambda=0.02,
                kl_ramp_iters=10000,
                lam_evi_sar=0.2,
                lam_evi_rgb=0.1,
                lam_evi_fuse=0.2,
                lam_evi_missing=0.2,
                lam_evi_fuse_rec=0.2,
                lam_distill=0.0,
                # ===== Auxiliary regularisers =====
                lam_dice=0.0,
                lam_balance=0.0,
                # ===== Pillar-1 L_mse (last two stages) — Plan A =====
                # CoDER.py picks the elif branch (apply_mmr=False) and applies
                # `lam_mse * 0.1 * mean_MSE(sar_content[2:], rgb_content[2:])`.
                # Real-data raw MSE expected ~20-60. lam_mse=0.2 → contribution
                # ~0.4-1.2 at init, much smaller after alignment converges.
                mse=dict(),
                alpha=1.0,
                lam_mse=0.2,
                # ===== Pillar-3 L_recon (missing-mode meta anchor) — Plan A =====
                # `_recon_loss` returns mean MSE across all 4 stages; raw ~20-50.
                # lam_recon=0.02 → contribution ~0.4-1.0 on missing steps.
                lam_recon=0.02,
                # ===== MMR (Pillar-1 whitening) still off — Plan B =====
                lam_mmr=0.0,
                # ===== No class weights, no OHEM =====
                ohem_keep_ratio=1.0,
                ignore_index=-1,
            ),
            data=data["mmr"],
        )
    ),
    data=data,
    optimizer=optimizer,
    learning_rate=learning_rate,
    train=train,
    test=test,
)
