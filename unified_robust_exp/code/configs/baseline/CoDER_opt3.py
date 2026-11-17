"""CoDER v3 Option-3 — Plan B: Opt2 + IBN-style IN + L_mmr.

Plan B goal (after Plan A under-performed on the full 15k run):
  Plan A (Opt2) added L_mse and L_recon to Opt1 but plateaued at ~0.29.
  Diagnosis: ConvNeXt's `en_content` lacks instance normalisation, so the
  L_mse pressure to align SAR/RGB meta features fights the encoder's natural
  tendency to preserve modality-specific style.  MetaRS paper §5.1 fixes
  this by adopting IBN-Net (Pan 2018) — replacing stem BN with IN and
  inserting an IN in the last block of Layer1 and Layer2.

  Plan B does the equivalent on ConvNeXt: optional InstanceNorm2d at the
  end of feature-pyramid outputs 0 and 1 (matching MetaRS Layer1/Layer2
  positions, strides 4 and 8).  This also gives us a whitening point for
  the original `L_mmr` cov-mask whitening loss, which we re-enable.

  Loss recipe:
      L_task   (CE on sar/rgb/fuse/missing)        ← same as Opt2
      L_r      (CE on fused, weight 1.5)           ← same as Opt2
      L_mse    on en_content stages 2+3            ← effective ×0.02 (Opt2 same)
      L_recon  on missing-mode meta features       ← ×0.02 (Opt2 same)
      L_mmr    cov whitening at stage 0+1 outputs  ← NEW, ×1.0
      EDL aux  ×0.2 each                            ← unchanged

  Key differences from Opt2:
    - `wt_layer = [0, 0, 0, 2, 2, 0, 0]`  →  IN at feat[0] (128ch) and feat[1] (256ch)
    - `in_channels_list = (0, 0, 0, 128, 256, 0, 0)` for cov-matrix sizing
    - `begin_mmr_iter = 1600` (restored from Opt2's 15001)
    - `mmr = dict()` re-added to loss
    - `lam_mmr = 1.0` (paper default)
    - `lam_mse = 0.02, alpha = 1.0` → effective 0.02 post-MMR (same as Opt2 effective)
    - β-discount stays off (β=1.0) — affects EDL only, neutral to mIoU

  Expected outcome (per paper ablation Table 6):
    - L_mmr addition (= "without Recover" → with Recover): +1.58 mIoU
    - L_mse addition over Plan A: already counted in Plan A's nominal +0.02
    - Together with ConvNeXt's +0.02 backbone boost: target 0.34-0.36
"""
from module.baseline.base import default_conv_block

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
            missing_modality_training=True,
            missing_modality_prob=dict(
                complete=1 / 3,
                sar_missing=1 / 3,
                rgb_missing=1 / 3,
            ),
            missing_modality_total_iterations=15000,
            # ConvNeXt-base channels; positions 3..6 correspond to feature
            # pyramid outputs 0..3 (strides 4 / 8 / 16 / 32).  Positions 3 and 4
            # carry MMR cov tracking (where wt_layer == 2).
            in_channels_list=(0, 0, 0, 128, 256, 0, 0),
            fuse_inchan_list=(128, 256, 512, 1024),
            relax_denom=0,
            clusters=10,
            # MMR machinery activates at this step (matches paper §5.1).
            begin_mmr_iter=1600,
            # β-discount stays off — affects EDL aux only.
            beta_scale=0.0,
            beta_floor=1.0,
            gamma_rec_init=1.0,
            beta_ramp_iters=0,
            use_moe=False,
            encoder=dict(
                resnet_type='resnet50',  # ignored
                include_conv5=True,
                batchnorm_trainable=True,
                pretrained=True,
                freeze_at=0,
                output_stride=32,
                # IN at feature-pyramid outputs 0 and 1
                # (analog of MetaRS Bottleneck IN at Layer1-out and Layer2-out).
                wt_layer=[0, 0, 0, 2, 2, 0, 0],
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
                # ===== Auxiliary EDL =====
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
                # ===== Pillar-1: L_mmr + L_mse =====
                # `mmr` key triggers the post-warmup branch (lam_mmr * mmr_loss +
                # lam_mse * alpha * MSE(stages 2+3)).  Pre-MMR step keeps the
                # warmup 0.1 factor (negligible).
                mmr=dict(),
                mse=dict(),
                alpha=1.0,
                lam_mmr=1.0,
                lam_mse=0.02,    # effective 0.02 post-MMR (= Opt2 effective)
                # ===== Pillar-3: L_recon on missing-mode meta features =====
                lam_recon=0.02,
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
