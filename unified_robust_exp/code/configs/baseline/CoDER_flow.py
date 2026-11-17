"""CoDER M5a — Uncertainty-Calibrated Latent Flow Restoration.

Builds on `CoDER_opt3.py` (ConvNeXt-base Plan B — the M3-chosen main backbone)
and swaps the deterministic missing-modality recovery for a conditional
**latent rectified flow** (`module/baseline/recovery_flow.py`).

Headline (M5a): the recovered modality's *content pyramid* is restored by a
few-step rectified flow conditioned on the available modality; the spread of
the ODE endpoint over K noise seeds gives a per-image restoration uncertainty
`sigma2`, which discounts the recovered Dirichlet evidence::

    beta_rec = beta_in * gamma_rec * exp(-sigma2 / flow_tau)

The contribution is the *calibrated uncertainty*, not the flow per se — we must
show sigma2 (a) tracks true restoration error (AUSE / sparsification), and
(b) improves calibration / robustness / segmentation vs the deterministic
recovery.  `use_flow_recovery=False` (i.e. `CoDER_opt3.py`) is the ablation
baseline; this config is the `use_flow_recovery=True` arm.

M5a scope = framework + dryrun only; the full train/finetune is M5b/M5c.
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
            # #17: headline readout — α_dst/S so the σ² gate drives mIoU.
            # Set 'head' to recover the MetaRS-equivalent ablation arm.
            missing_readout='dempster',
            missing_modality_training=True,
            missing_modality_prob=dict(
                complete=1 / 3,
                sar_missing=1 / 3,
                rgb_missing=1 / 3,
            ),
            missing_modality_total_iterations=15000,
            in_channels_list=(0, 0, 0, 128, 256, 0, 0),
            fuse_inchan_list=(128, 256, 512, 1024),
            relax_denom=0,
            clusters=10,
            begin_mmr_iter=1600,
            beta_scale=0.0,
            beta_floor=1.0,
            gamma_rec_init=1.0,
            beta_ramp_iters=0,
            use_moe=False,
            # ===== M5a: latent rectified-flow restoration =====
            use_flow_recovery=True,
            flow_tau=0.5,          # learned gate sigmoid(-h/τ) sharpness; τ=0.5
                                   # picked by tests/calibrate_flow_tau.py
                                   # (spread 0.30, median 0.60 on frozen ckpt)
            flow_t_dim=64,
            flow_n_steps=4,        # Euler ODE steps (few-step)
            flow_n_samples=4,      # K noise seeds for sigma2
            flow_sigma_detach=True,
            # #18: per-position σ²(x) → β_rec(x) gate. Global per-image σ² has
            # ~zero dynamic range (claim-1 ρ=0.12); spatial map is selective.
            flow_spatial_sigma=True,
            # #22: reliability source. 'learned' = head trained to predict the
            # restoration error (offline ρ≈0.53 vs seed-variance 0.20). β_rec(x)
            # = β·exp(−r(x)/τ) with r = predicted err (≥0, scale ~0.3–5) ⇒
            # flow_tau gates harder than the σ² era; retune at M5b/c.
            reliability_source='learned',
            encoder=dict(
                resnet_type='resnet50',
                include_conv5=True,
                batchnorm_trainable=True,
                pretrained=True,
                freeze_at=0,
                output_stride=32,
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
                lam_ce_sar=1.0,
                lam_ce_rgb=0.5,
                lam_ce_fuse=1.5,
                lam_ce_missing=1.0,
                # #17: direct CE on σ²-gated Dempster-fused output (train-what-
                # you-read-out). Matches lam_ce_missing so α_dst is a competitive
                # predictor, not 5× under-supervised vs the recovery head.
                lam_ce_dst_rec=1.0,
                # #22: reliability-head regression to true restoration error.
                lam_rel=1.0,
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
                mmr=dict(),
                mse=dict(),
                alpha=1.0,
                lam_mmr=1.0,
                lam_mse=0.02,
                lam_recon=0.02,
                # ===== M5a: rectified-flow velocity-matching loss =====
                lam_flow=1.0,
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
