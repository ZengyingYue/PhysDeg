"""CoDER M3 — TransOSS ViT-B/16 + ViT-Adapter genuine multi-scale.

Builds on `CoDER_transoss_v2.py` (R1 differential LR + R2 no-WD + R3 per-tap
LN) and adds the headline backbone change:

  R4/R5  use_adapter=True → replace the ViTDet "fake" pyramid (one stride-16
         map up/down-sampled) with a ViT-Adapter (Chen et al. ICLR 2023):
         a parallel conv Spatial Prior Module + Injector/Extractor
         cross-attention give the contrastively-pretrained ViT a *genuine*
         {1/4,1/8,1/16,1/32} feature pyramid.  The Injector gate `gamma` is
         init-0, so training starts as the pristine pretrained ViT and the
         adapter grows in — protecting the (re-ID-contrastive, weakly-spatial)
         pretraining while supplying the high-res spatial structure dense
         segmentation needs.

LR strategy is kept identical to v2 (flat vit_lr_scale=0.1) so the
adapter-vs-v2 delta isolates the multi-scale contribution.  `tap_norm` is
irrelevant here (the adapter path replaces the taps entirely).
"""
from module.baseline.base import default_conv_block

from configs.base.EarthMiss import train, test, data, optimizer, learning_rate

config = dict(
    model=dict(
        type='CoDER',
        params=dict(
            backbone='transoss',
            transoss_img_size=512,
            transoss_tap_blocks=(3, 6, 9, 12),
            sar_in_channels=1,
            rgb_in_channels=3,
            # ===== M3 ViT-finetuning corrections (from v2) =====
            tap_norm=False,         # n/a under adapter path
            vit_lr_scale=0.1,       # R1: pretrained ViT trunk LR = 0.1 × base
            vit_layer_decay=None,
            # ===== ViT-Adapter (R4/R5) =====
            use_adapter=True,
            adapter_spm_dim=64,
            adapter_heads=8,
            infer_rgb=False,
            infer_mode='sar',
            missing_modality_training=True,
            missing_modality_prob=dict(
                complete=1 / 3,
                sar_missing=1 / 3,
                rgb_missing=1 / 3,
            ),
            missing_modality_total_iterations=15000,
            in_channels_list=(0, 0, 128, 256, 512, 1024, 0),
            fuse_inchan_list=(128, 256, 512, 1024),
            relax_denom=0,
            clusters=10,
            begin_mmr_iter=15001,
            beta_scale=0.0,
            beta_floor=1.0,
            gamma_rec_init=1.0,
            beta_ramp_iters=0,
            use_moe=False,
            encoder=dict(
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
                lam_ce_sar=1.0,
                lam_ce_rgb=0.5,
                lam_ce_fuse=1.5,
                lam_ce_missing=1.0,
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
                lam_dice=0.0,
                lam_balance=0.0,
                mse=dict(),
                alpha=1.0,
                lam_mse=0.2,
                lam_recon=0.02,
                lam_mmr=0.0,
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
