"""CoDER M3 — TransOSS ViT-B/16 as the shared meta encoder (`en_content`).

Baseline-equivalent to `CoDER_opt2.py` (Plan A: CE-on-fuse + L_mse + L_recon,
no IBN, no MMR — ConvNeXt-base had no native IN layers to whiten so MMR was
disabled).  The only structural change here is the backbone:

    en_content      → TransOSSEncoder (ViT-B/16, dual patch_embed, mie_embed)
    en_sar, en_rgb  → ConvNeXt-base (unchanged, cheap modality-specific paths)

Why TransOSS as `en_content` only:
  TransOSS is pretrained jointly on SAR + optical (Wang et al. ICCV 2025) and
  ships with two patch-embeds + a 2-entry modality-identity embedding —
  exactly the role of CoDER's shared meta encoder.  Putting it on `en_sar` /
  `en_rgb` would waste pretraining and triple the param count.

Multi-scale FPN signature stays identical to ConvNeXt-base:
    strides   (4, 8, 16, 32)
    channels  (128, 256, 512, 1024)
A ViTDet SimpleFPN (Li et al. ECCV 2022) maps blocks {3, 6, 9, 12} to those
levels through transposed-conv / stride-2 conv + 1×1 projection.

Pillar status in this config (CoDER_transoss minimum):
    Pillar-1 cov-decoupling : OFF  (no IN in ViT-B/16, β stays 1.0)
    Pillar-1 L_mmr           : OFF
    Pillar-1 L_mse           : ON  (stages 2+3 of `en_content`, ×0.02 effective)
    Pillar-2 β-reliability   : OFF (β_scale=0)
    Pillar-3 L_recon         : ON  (×0.02 on missing-mode meta features)
    Pillar-3 Dempster combo  : ON  (auxiliary, eval-time uncertainty)
    EDL auxiliary heads       : ON  (×0.2 each)

`transoss_img_size` controls the initial pos_embed grid (re-interpolated at
runtime per batch as well, so this only affects the saved buffer size).
`transoss_tap_blocks` lets us ablate which 4 blocks feed the multi-scale FPN.
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
            # v1 (this config) predates the M3-v2 ViT-finetuning fixes; keep the
            # encoder byte-identical to what its checkpoints were trained with.
            tap_norm=False,
            vit_lr_scale=1.0,
            vit_layer_decay=None,
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
