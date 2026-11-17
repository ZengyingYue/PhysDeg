"""CoDER M3-v2 — TransOSS ViT-B/16 meta encoder with proper ViT-finetuning.

Identical to `CoDER_transoss.py` (the M3-v1 run, in-train peak 0.313 < ConvNeXt
0.317-0.322) except for three principled corrections to *how the pretrained
ViT is adapted* — diagnosed against the v1 run:

  R1  Differential LR.  v1 trained the pretrained TransOSS trunk at the same
      flat lr=1e-4 as the random-init taps/FPN/decoders → the ViT forgot its
      SAR+optical pretraining early.  v2 sets `vit_lr_scale=0.1` so the trunk
      adapts at 1e-5 while the new modules keep 1e-4.  (`vit_layer_decay`, if
      set e.g. 0.8, switches to BEiT/MAE layer-wise LR decay instead.)
      Honoured by the patched `set_lr` in train.py.

  R2  No weight decay on norms / biases / pos_embed / cls_token / mie_embed.
      v1's single AdamW group applied wd=0.01 to all of them.  `custom_param_groups`
      now excludes every 1-D parameter + the ViT positional/modality embeds.

  R3  Per-tap LayerNorm.  The ViT residual stream grows ~7× in magnitude from
      block 3 → 12; v1 fed the four taps to their projectors un-normalised, so
      the deepest FPN level dominated.  v2 normalises each tapped feature over
      the channel dim before its up/down-sample projector (`tap_norm=True`,
      ViTDet recipe).

Everything else (losses, pillars, augs, fusion) is byte-identical to v1 so the
v1→v2 delta isolates the ViT-finetuning fix.
"""
from module.baseline.base import default_conv_block

from configs.base.EarthMiss import train, test, data, optimizer, learning_rate

config = dict(
    model=dict(
        type='CoDER',
        params=dict(
            backbone='transoss',
            # TransOSS-specific knobs.
            transoss_img_size=512,
            transoss_tap_blocks=(3, 6, 9, 12),
            sar_in_channels=1,
            rgb_in_channels=3,
            # ===== M3-v2 ViT-finetuning corrections =====
            tap_norm=True,          # R3: per-tap LayerNorm (ViTDet)
            vit_lr_scale=0.1,       # R1: pretrained ViT trunk LR = 0.1 × base
            vit_layer_decay=None,   # set e.g. 0.8 for layer-wise LR decay instead
            infer_rgb=False,
            infer_mode='sar',
            missing_modality_training=True,
            missing_modality_prob=dict(
                complete=1 / 3,
                sar_missing=1 / 3,
                rgb_missing=1 / 3,
            ),
            missing_modality_total_iterations=15000,
            # Channel widths CoDER's FPN/Project/Concat expect (mirror ConvNeXt-base).
            in_channels_list=(0, 0, 128, 256, 512, 1024, 0),
            fuse_inchan_list=(128, 256, 512, 1024),
            relax_denom=0,
            clusters=10,
            # MMR is gated off (no IN in ViT) — keep iter > total to never fire.
            begin_mmr_iter=15001,
            # β-discount disabled (no MMR signal in this config — Pillar-2 is a
            # later milestone).
            beta_scale=0.0,
            beta_floor=1.0,
            gamma_rec_init=1.0,
            beta_ramp_iters=0,
            # Keep MetaRS-style Fusespfnv2 fusion until M4 swaps it for cross-attn.
            use_moe=False,
            encoder=dict(
                # Kept for old code paths; values irrelevant when backbone=transoss.
                resnet_type='resnet50',
                include_conv5=True,
                batchnorm_trainable=True,
                pretrained=True,
                freeze_at=0,
                output_stride=32,
                wt_layer=[0, 0, 0, 0, 0, 0, 0],
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
                # ===== Primary CE on softmax =====
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
                # ===== Pillar-1 L_mse (Plan A heritage, kept) =====
                mse=dict(),
                alpha=1.0,
                lam_mse=0.2,        # × 0.1 warmup = effective 0.02
                # ===== Pillar-3 L_recon =====
                lam_recon=0.02,
                # ===== MMR OFF =====
                lam_mmr=0.0,
                # ===== No class weights / OHEM =====
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
