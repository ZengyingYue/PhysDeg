"""CoDER v3 Option-1 config — ConvNeXt-base backbone verification.

Goal: test whether replacing ResNet50 with ConvNeXt-base lifts the EarthMiss
SAR-only ceiling from ~0.30 (MetaRS) to ≥ 0.32 with the MetaRS-equivalent
training recipe.  The CoDER framework (EDL + Dempster) is kept only as an
auxiliary uncertainty path (low weight); the headline classification is
softmax + CE on the fused decoder.

Differences from the v2.2 / S1 config:
- `backbone='convnext_base'` (was `resnet50`)
- channels:  `fuse_inchan_list=(128,256,512,1024)`,
             `fpn.in_channels_list=(256,512,1024,2048)`  (= 2 × fuse_inchan_list for Concat)
- MMR disabled (`wt_layer=[0,…]`) — ConvNeXt has no BN to whiten
- β-discount disabled (`beta_floor=1.0`, `beta_scale=0.0`)
- MoE disabled (`use_moe=False`) → Fusespfnv2, the original MetaRS fuse module
- Aggressive tricks dropped: no RandomScale aug, no class_weights, no OHEM, no dice loss
- EDL aux losses kept at very low weight (`lam_evi_*=0.2`) so we still get
  ECE numbers for §5.4 uncertainty story
- 15000 steps (matches MetaRS baseline)
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
                # 2 × fuse_inchan_list because decoder_sar/rgb take Concat(fuse, content)
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
                # ===== Auxiliary regularisers (off in Option-1) =====
                lam_dice=0.0,
                lam_balance=0.0,
                # ===== mmr and mse keys are NOT present — see below =====
                # ===== Pillar-1 anchors =====
                # MetaRS uses these to align en_content's outputs across modalities,
                # but ConvNeXt features have a much larger magnitude than ResNet50
                # BN outputs (no normalisation at the feature level), so the raw
                # MSE between sar_content and rgb_content explodes to ~50.
                # We disable the meta-feature MSE for the ConvNeXt path; the
                # shared en_content already enforces modality-invariance via the
                # shared weights themselves (different stem, same backbone).
                # NOTE: `mse` and `mmr` keys are deliberately ABSENT so the
                # CoDER forward never enters those loss branches.
                # `lam_recon` set to 0 so the missing-mode recon term doesn't
                # blow up either (same ConvNeXt-scale issue).
                lam_recon=0.0,
                lam_mmr=0.0,
                lam_mse=0.0,
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
