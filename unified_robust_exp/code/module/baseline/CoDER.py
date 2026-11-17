"""CoDER — Covariance-Driven Evidential Reliability for missing-modality
remote-sensing land-cover mapping.

v3-S1 architecture (this file):
  - Three classification decoders are restored, MetaRS-style:
        decoder_sar    — per-modality SAR head
        decoder_rgb    — per-modality RGB head
        decoder_fused  — joint head over MoE-fused content features
  - The MetaRS `Fuse` module is replaced by `FpnMoE` (`module/baseline/moe_fusion.py`):
        4 experts per FPN level (2 private + 2 collaborative) with a global
        softmax router and an entropy balance reward.
  - The primary classification supervision is softmax + cross-entropy on
    each head's raw logits.  The third head (fused) is the eval-time
    output for "fuse" mode; the SAR head is the eval-time output for the
    SAR-only mode (the §5.4 headline).
  - EDL + β-discount + Dempster combination are kept as an *auxiliary
    uncertainty path*.  The same logits tensor that feeds CE also feeds
    `softplus` to produce non-negative evidence.  EDL loss is added at a
    smaller weight; Dempster combines `α_sar` and `α_rgb` (or `α_recovered`)
    for the uncertainty map / ECE deliverable.

See `V3_PLAN.md` for the staged roadmap and `V3_EXPERIMENT_LOG.md` for
per-stage results.
"""
from __future__ import annotations

import copy
import logging
import os
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from ever import registry
from ever.core.builder import make_dataloader
from ever.interface import ERModule
from ever.module import FPN
from tqdm import tqdm

from module.baseline.base import AssymetricDecoder
from module.baseline.base_metars.mmr import CovMatrix_mmr, instance_whitening_loss
from module.baseline.base_metars.model import (
    Concat,
    Fusespfnv2,
    Fusespfnv2_res,
    RandomDropModality,
)
from module.baseline.base_resnet.resnet import ResNetEncoder
from module.baseline.complete_recovery import (
    FullFeaturePyramidRecoverer,
    FullImageRecoverer,
)
from module.baseline.moe_fusion import FpnMoE
from module.baseline.recovery_flow import LatentRectifiedFlow
from module.reliability.covariance_deviation import CovarianceDeviationReliability
from module.reliability import pixel_stats as pxs
from module.evidential import (
    CovarianceReliability,
    alpha_to_prob,
    alpha_to_uncertainty,
    dempster_combine,
    dirichlet_kl_distillation,
    discount_evidence,
    edl_loss,
    evidence_to_alpha,
    kl_annealing,
    logits_to_evidence,
    soft_dice_loss,
)

logger = logging.getLogger(__name__)


@registry.MODEL.register("CoDER")
class CoDER(ERModule):
    """Two-head evidential model with covariance-driven reliability + Dempster.

    Forward semantics (parallels MetaRS but cleaner):

    Training (complete modality):
        sar_content = en_content(sar);  sar_specific = en_sar(sar)
        rgb_content = en_content(rgb);  rgb_specific = en_rgb(rgb)
        sar_fuse = Project(sar_specific, sar_content)
        rgb_fuse = Project(rgb_specific, rgb_content)

        e_sar_in = softplus( decoder_sar( Concat(sar_fuse, sar_content) ) )
        e_rgb_in = softplus( decoder_rgb( Concat(rgb_fuse, rgb_content) ) )

        beta_in = CovReliability(cov_layers, recovered=False)
        alpha_sar = beta_in * e_sar_in + 1
        alpha_rgb = beta_in * e_rgb_in + 1
        alpha_fuse = Dempster(alpha_sar, alpha_rgb)

        L = L_evi(alpha_fuse) + L_evi(alpha_sar) + L_evi(alpha_rgb)
            + lam_mmr * L_mmr + lam_mse * L_mse

    Training (one modality dropped, e.g. RGB missing):
        Only sar_content / sar_specific are *available*; produce an
        in-modality SAR alpha as above, then re-feed (sar_fuse, sar_content)
        through decoder_rgb to obtain a recovered RGB evidence head, and
        discount it by beta_rec = beta_in * gamma_rec  (gamma_rec ≪ 1).
        Add L_evi(alpha_rgb_rec) plus L_recon (an L2 between the recovered
        meta features and the ground-truth ones, which we still have during
        training since the modality is "missing" only for the forward path).
        Dempster combine to get alpha_fuse and supervise L_evi(alpha_fuse).

    Inference (modality `infer_mode` is the only one available):
        Use that modality's content/specific to build in-modality alpha and
        recovered alpha (by passing the same fused features through the
        opposite head), Dempster-combine, return prob = alpha / S.
        u = K / S is exposed as `last_uncertainty` for §5.4 heatmaps.
    """

    def __init__(self, config):
        super().__init__(config)

        # === encoders =====================================================
        # backbone='resnet50' (default) keeps MetaRS structure
        # backbone='convnext_base' swaps in ConvNeXt-base for all 3 encoders
        # (channels become [128,256,512,1024]; MMR is disabled automatically).
        self.backbone_name = str(getattr(self.config, "backbone", "resnet50"))
        if self.backbone_name == "transoss":
            # TransOSS shines as the shared meta encoder (`en_content`): its
            # dual patch embeds (SAR/RGB) + `mie_embed` are exactly designed
            # for cross-modal feature sharing.  We keep `en_sar` / `en_rgb`
            # as ConvNeXt-base so the modality-specific paths stay cheap.
            from module.baseline.convnext_encoder import ConvNeXtEncoder
            from module.baseline.transoss_encoder import TransOSSEncoder
            self.en_sar = ConvNeXtEncoder(
                in_channels=self.config.sar_in_channels, pretrained=True,
                wt_layer=None,
            )
            self.en_rgb = ConvNeXtEncoder(
                in_channels=self.config.rgb_in_channels, pretrained=True,
                wt_layer=None,
            )
            self.en_content = TransOSSEncoder(
                sar_in_channels=self.config.sar_in_channels,
                rgb_in_channels=self.config.rgb_in_channels,
                img_size=int(getattr(self.config, "transoss_img_size", 512)),
                tap_blocks=tuple(getattr(self.config, "transoss_tap_blocks", (3, 6, 9, 12))),
                fpn_channels=tuple(self.config.fuse_inchan_list),
                fpn_strides=(4, 8, 16, 32),
                pretrained=True,
                tap_norm=bool(getattr(self.config, "tap_norm", True)),
                use_adapter=bool(getattr(self.config, "use_adapter", False)),
                adapter_spm_dim=int(getattr(self.config, "adapter_spm_dim", 64)),
                adapter_heads=int(getattr(self.config, "adapter_heads", 8)),
            )
        elif self.backbone_name in ("convnext_base", "convnext_large"):
            from module.baseline.convnext_encoder import ConvNeXtEncoder
            _cn_model = "convnext_large" if self.backbone_name == "convnext_large" else "convnext_base"
            # MetaRS pattern: only en_content gets the IN layers; en_sar /
            # en_rgb stay plain so they can preserve modality-specific style.
            self.en_sar = ConvNeXtEncoder(
                in_channels=self.config.sar_in_channels, pretrained=True,
                wt_layer=None, model_name=_cn_model,
            )
            self.en_rgb = ConvNeXtEncoder(
                in_channels=self.config.rgb_in_channels, pretrained=True,
                wt_layer=None, model_name=_cn_model,
            )
            self.en_content = ConvNeXtEncoder(
                in_channels=self.config.sar_in_channels, pretrained=True,
                wt_layer=list(self.config.encoder.wt_layer), model_name=_cn_model,
            )
            # If no IN positions are configured (Opt1/Opt2 path), MMR stays off
            # automatically because `apply_mmr = 2 in wt_layer` is False.
        else:
            # Original MetaRS pattern (ResNet50).
            org_wt_layer = self.config.encoder.wt_layer
            self.config.encoder.wt_layer = [0] * len(org_wt_layer)
            self.config.encoder.in_channels = self.config.sar_in_channels
            self.en_sar = ResNetEncoder(self.config.encoder)
            self.config.encoder.in_channels = self.config.rgb_in_channels
            self.en_rgb = ResNetEncoder(self.config.encoder)

            self.config.encoder.wt_layer = org_wt_layer
            self.config.encoder.in_channels = self.config.sar_in_channels
            self.en_content = ResNetEncoder(self.config.encoder)

        # === fusion / projection (kept from MetaRS) =======================
        self.Project = Fusespfnv2_res(self.config.fuse_inchan_list)
        self.Concat = Concat()

        # === per-modality classification heads (raw logits) =================
        # Each head produces logits.  softmax(logits) → primary CE loss.
        # softplus(logits) → evidence → auxiliary EDL path.
        self.decoder_sar = nn.Sequential(
            FPN(**self.config.fpn),
            AssymetricDecoder(**self.config.decoder),
            nn.Conv2d(self.config.decoder.out_channels, self.config.num_classes, 1),
            nn.UpsamplingBilinear2d(scale_factor=4),
        )
        self.decoder_rgb = nn.Sequential(
            FPN(**self.config.fpn),
            AssymetricDecoder(**self.config.decoder),
            nn.Conv2d(self.config.decoder.out_channels, self.config.num_classes, 1),
            nn.UpsamplingBilinear2d(scale_factor=4),
        )

        # === Cross-modal fusion module ======================================
        # `use_moe=True`  → FpnMoE (4 experts/level, learned routing); v3-S1.
        # `use_moe=False` → MetaRS's Fusespfnv2 (SCSE-based fused features).
        # Either way, the output channels match `fuse_inchan_list` and the
        # downstream `decoder_fused` consumes that pyramid.
        # `fusion_type='moar'` → CMNeXtSGSLN MoAR+DMFIA fusion (clean-SOTA graft;
        #   returns the same (fused, router, balance) 3-tuple so call sites are
        #   unchanged).  Default 'fpnmoe' keeps the original FpnMoE behaviour.
        self.fusion_type = str(getattr(self.config, "fusion_type", "fpnmoe"))
        self.use_moe = bool(getattr(self.config, "use_moe", True))
        if self.fusion_type == "moar":
            from module.baseline.moar_fusion import MoARFpnFuse
            self.MoE_Fuse = MoARFpnFuse(
                inchan_list=list(self.config.fuse_inchan_list),
                num_dmfia_units=int(getattr(self.config, "moar_dmfia_units", 1)),
                moa_num_experts=int(getattr(self.config, "moar_num_experts", 3)),
                moa_top_k=int(getattr(self.config, "moar_top_k", 1)),
                gate_mode=str(getattr(self.config, "moar_gate_mode", "renorm")),
                null_expert=bool(getattr(self.config, "moar_null_expert", False)),
            )
            self.use_moe = True  # use the 3-tuple unpacking path at the call sites
        elif self.use_moe:
            self.MoE_Fuse = FpnMoE(
                inchan_list=list(self.config.fuse_inchan_list),
                router_hidden=int(getattr(self.config, "router_hidden", 128)),
                router_dropout=float(getattr(self.config, "router_dropout", 0.1)),
            )
        else:
            # MetaRS-style fused features — no router, no balance loss.
            self.MoE_Fuse = Fusespfnv2(self.config.fuse_inchan_list)
        # Fused decoder — FPN (default) or CMNeXtSGSLN's SegFormer-style
        # MultiLevelDecoder (`fused_head='multilevel'`).  Only the CLEAN/fused
        # readout uses this; decoder_sar is untouched, so SAR-only is unaffected.
        self.fused_head = str(getattr(self.config, "fused_head", "fpn"))
        if self.fused_head == "multilevel":
            from module.baseline.moar_fusion import MultiLevelFusedDecoder
            self.decoder_fused = MultiLevelFusedDecoder(
                in_channels=tuple(self.config.fuse_inchan_list),
                decoder_channels=int(self.config.decoder.out_channels),
                num_classes=int(self.config.num_classes),
                fuse_level=0,
                dropout=float(getattr(self.config, "fused_head_dropout", 0.0)),
                scale_factor=4,
            )
        else:
            fpn_cfg_fused = copy.deepcopy(dict(self.config.fpn))
            fpn_cfg_fused["in_channels_list"] = tuple(self.config.fuse_inchan_list)
            self.decoder_fused = nn.Sequential(
                FPN(**fpn_cfg_fused),
                AssymetricDecoder(**self.config.decoder),
                nn.Conv2d(self.config.decoder.out_channels, self.config.num_classes, 1),
                nn.UpsamplingBilinear2d(scale_factor=4),
            )

        # === missing-modality sampler (reused from MetaRS) ================
        self.missing_modality_training = bool(
            getattr(self.config, "missing_modality_training", False)
        )
        self.missing_modality_sampler = RandomDropModality(
            modality_num=2,
            total_iterations=getattr(
                self.config, "missing_modality_total_iterations", 15000
            ),
        )
        missing_modality_prob = getattr(self.config, "missing_modality_prob", None)
        if missing_modality_prob is not None:
            def get_prob(name):
                if isinstance(missing_modality_prob, dict):
                    return missing_modality_prob[name]
                return getattr(missing_modality_prob, name)
            self.missing_modality_sampler.prob = np.array(
                [
                    get_prob("complete"),
                    get_prob("sar_missing"),
                    get_prob("rgb_missing"),
                ],
                dtype=np.float64,
            )

        # === RGB-stem swap-in (so en_content can also encode RGB) =========
        # TransOSS doesn't need this — its `patch_embed` (RGB) and
        # `patch_embed_SAR` are both built-in and selected via `set_modality`.
        if self.backbone_name == "transoss":
            pass
        elif self.backbone_name in ("convnext_base", "convnext_large"):
            # ConvNeXt stem: base -> 128, large -> 192
            _stem_out = 192 if self.backbone_name == "convnext_large" else 128
            self.add_module(
                "rgb_conv1",  # name kept for state-dict compat with old ckpts
                nn.Conv2d(
                    self.config.rgb_in_channels,
                    _stem_out,
                    kernel_size=4,
                    stride=4,
                ),
            )
        else:
            self.add_module(
                "rgb_conv1",
                nn.Conv2d(
                    self.config.rgb_in_channels,
                    64,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                ),
            )

        # === MMR / covariance machinery (Pillar 1 base) ===================
        self.apply_mmr = 2 in self.config.encoder.wt_layer
        self.eps = 1e-5
        self.cov_matrix_layer = []
        self.cov_type = []
        for i in range(len(self.config.encoder.wt_layer)):
            if self.config.encoder.wt_layer[i] == 2:
                self.cov_matrix_layer.append(
                    CovMatrix_mmr(
                        dim=self.config.in_channels_list[i],
                        relax_denom=self.config.relax_denom,
                        clusters=self.config.clusters,
                    )
                )
                self.cov_type.append(self.config.encoder.wt_layer[i])

        # === Pillar 2 — covariance → reliability discount ================
        self.cov_reliability = CovarianceReliability(
            scale=float(getattr(self.config, "beta_scale", 6.0)),
            eps=float(getattr(self.config, "beta_floor", 0.05)),
            gamma_rec_init=float(getattr(self.config, "gamma_rec_init", 0.7)),
            ramp_iters=int(getattr(self.config, "beta_ramp_iters", 0)),
            # ramp begins at MMR activation by default
            ramp_begin_iter=int(getattr(self.config, "begin_mmr_iter", 0)),
        )
        # Core-ablation switches.  Keep the historical behaviour as defaults,
        # but make the observed-quality and recovered-source discounts
        # independently removable.  This is necessary for a true one-factor
        # ablation: beta_rec previously bundled beta_obs and gamma_rec inside
        # CovarianceReliability.compute(recovered=True).
        self.observed_reliability_enabled = bool(
            getattr(self.config, "observed_reliability_enabled", True)
        )
        self.observed_reliability_mode = str(
            getattr(self.config, "observed_reliability_mode", "legacy")
        )
        # 'whitening' (default): reliability from the shared encoder's
        # instance-whitening features.  'content': from the first two levels
        # of the shared pyramid (for encoders without whitening layers).
        self.reliability_features = str(
            getattr(self.config, "reliability_features", "whitening")
        )
        # What the fused head reads: 'content' (default, shared features only)
        # or 'projected' (shared + sensor-specific, the P(r, c) features).
        self.fused_input = str(getattr(self.config, "fused_input", "content"))
        self.observed_cov_reliability = None
        if self.observed_reliability_mode == "covariance_deviation":
            self.observed_cov_reliability = CovarianceDeviationReliability(
                channels=([layer.dim for layer in self.cov_matrix_layer]
                          if self.reliability_features == "whitening"
                          else list(self.config.fuse_inchan_list[:2])),
                beta_floor=float(getattr(self.config, "obs_beta_floor", 0.1)),
                mask_fraction=float(getattr(self.config, "obs_cov_mask_fraction", 0.1)),
                clean_quantile=float(getattr(self.config, "obs_clean_quantile", 0.90)),
                scale_quantile=float(getattr(self.config, "obs_scale_quantile", 0.99)),
                tau=float(getattr(self.config, "obs_beta_tau", 1.0)),
                diag_weight=float(getattr(self.config, "obs_diag_weight", 0.0)),
            )
            stats_path = str(getattr(self.config, "observed_reliability_stats", ""))
            if stats_path:
                payload = torch.load(stats_path, map_location="cpu")
                state = payload.get("state_dict", payload)
                self.observed_cov_reliability.load_state_dict(state, strict=True)
                logger.info("[CoDER] loaded beta_obs reference: %s", stats_path)
        elif self.observed_reliability_mode != "legacy":
            raise ValueError(
                f"unknown observed_reliability_mode={self.observed_reliability_mode!r}"
            )
        self.recovered_reliability_enabled = bool(
            getattr(self.config, "recovered_reliability_enabled", True)
        )
        self.dempster_enabled = bool(
            getattr(self.config, "dempster_enabled", True)
        )
        # How an observed stream is combined with evidence *recovered from that
        # same observation*.  'dempster' (legacy) treats the two as distinct
        # sources; 'average' is the provenance-aware rule (see
        # ``_combine_recovered``); 'additive' pools evidence.
        self.recovered_combination = str(
            getattr(self.config, "recovered_combination", "dempster")
        )
        if self.recovered_combination not in ("dempster", "average", "additive"):
            raise ValueError(
                f"unknown recovered_combination={self.recovered_combination!r}"
            )
        # Reliability-gated readout for dual-modal input: an image whose
        # optical (SAR) reliability falls below tau is read out by the SAR
        # (optical) head.  tau is set post hoc from clean validation images
        # only (experiments/revision/finalize_reliability.py) and saved in the
        # checkpoint; the default 0 never triggers.
        self.reliability_fallback = bool(
            getattr(self.config, "reliability_fallback", False)
        )
        self.register_buffer("fallback_tau", torch.zeros(2))  # (sar, rgb)
        # 'beta': both sensors judged by covariance reliability.  'hybrid':
        # optical by covariance reliability, SAR by the SAR head's mean max
        # probability (the whitened SAR covariance does not respond to speckle).
        self.fallback_rule = str(getattr(self.config, "fallback_rule", "beta"))
        self.register_buffer("fallback_msp_tau", torch.zeros(2))  # (sar head, rgb head)
        # ReBound-v4: spatial modality dropout (training) and masked fusion
        # (inference).  spatial_dropout_prob: fraction of training samples whose
        # optical input is occluded by a random cloud-like mask; the occluded
        # area is removed from the optical input, from the fused path's optical
        # stream and from optical-side losses.  spatial_mask_infer='pix'
        # estimates the unreliable optical area from local pixel statistics.
        self.spatial_dropout_prob = float(getattr(self.config, "spatial_dropout_prob", 0.0))
        self.spatial_mask_infer = str(getattr(self.config, "spatial_mask_infer", "none"))
        # 'hole': occluded optical content is removed (ReBound-v4).  'cloud': a
        # synthetic cloud is blended in and nothing else changes -- plain
        # cloud augmentation, the baseline for 'hole'.
        self.occlusion_mode = str(getattr(self.config, "occlusion_mode", "hole"))
        if self.occlusion_mode not in ("hole", "cloud"):
            raise ValueError(f"unknown occlusion_mode {self.occlusion_mode!r}")
        # Synthetic-cloud appearance in normalised optical units (per band).
        self.register_buffer("cloud_value", torch.tensor(
            list(getattr(self.config, "cloud_value_norm", [3.0, 3.0, 3.0])), dtype=torch.float32),
            persistent=False)
        self.mask_dilate = int(getattr(self.config, "mask_dilate", 31))
        self.mask_max_cover = float(getattr(self.config, "mask_max_cover", 0.9))
        # Clean-data statistics (fitted post hoc by experiments/revision/evaluate.py).
        self.register_buffer("rel_img_mu_o", torch.zeros(9))
        self.register_buffer("rel_img_sd_o", torch.ones(9))
        self.register_buffer("rel_img_mu_s", torch.zeros(3 * self.config.sar_in_channels))
        self.register_buffer("rel_img_sd_s", torch.ones(3 * self.config.sar_in_channels))
        self.register_buffer("rel_img_tau", torch.full((2,), float("inf")))  # (sar, rgb)
        self.register_buffer("rel_loc_mu", torch.zeros(6))
        self.register_buffer("rel_loc_sd", torch.ones(6))
        self.register_buffer("rel_loc_tau", torch.full((1,), float("inf")))
        self.external_mask = None  # analysis hook: ground-truth occlusion mask
        self._spatial_keep = None
        self.last_mask_cover = None
        if self.spatial_dropout_prob > 0:
            loss_cfg_ = self.config.loss
            if any(float(loss_cfg_.get(f"lam_evi_{k}", 0.0)) > 0
                   for k in ("sar", "rgb", "fuse", "missing", "fuse_rec")):
                raise ValueError("spatial dropout is implemented for the EDL-free (v3c) objective")
        self.last_route = None
        if bool(getattr(self.config, "freeze_gamma_rec", False)):
            self.cov_reliability.gamma_rec_raw.requires_grad_(False)

        # === M5a: latent rectified-flow restoration ========================
        # When enabled, the missing-modality branch restores the absent
        # modality's content pyramid with a conditional rectified flow and
        # discounts the recovered evidence by its restoration uncertainty:
        #     beta_rec = beta_in * gamma_rec * exp(-sigma2 / flow_tau)
        # When disabled (default), the deterministic MetaRS recovery is kept,
        # so `use_flow_recovery` toggles the headline ablation.
        self.use_flow_recovery = bool(getattr(self.config, "use_flow_recovery", False))
        self.recovery_method = str(getattr(self.config, "recovery_method", "auto"))
        self.flow_tau = float(getattr(self.config, "flow_tau", 1.0))
        self.lam_flow = float(self.config.loss.get("lam_flow", 0.0))
        # #18: per-position σ²(x) → β_rec(x) gate (vs per-image scalar σ²).
        # Global σ² has near-zero dynamic range (claim-1: ρ=0.12 on the first
        # 15k ckpt); the spatial map is what lets the gate be selective.
        self.flow_spatial_sigma = bool(getattr(self.config, "flow_spatial_sigma", False))
        # #22: reliability source for the β_rec gate. "seed" = K-seed ensemble
        # variance σ² (weak, ρ≈0.19 — ablation); "learned" = a head trained to
        # predict the restoration error (ρ≈0.53 offline). Learned mode implies
        # spatial (the head outputs a per-position map).
        self.reliability_source = str(getattr(self.config, "reliability_source", "seed"))
        if self.reliability_source == "learned":
            self.flow_spatial_sigma = True
        if self.use_flow_recovery:
            self.recovery_flow = LatentRectifiedFlow(
                channels=list(self.config.fuse_inchan_list),
                t_dim=int(getattr(self.config, "flow_t_dim", 64)),
                n_steps=int(getattr(self.config, "flow_n_steps", 4)),
                n_samples=int(getattr(self.config, "flow_n_samples", 4)),
                sigma_detach=bool(getattr(self.config, "flow_sigma_detach", True)),
                learned_reliability=(self.reliability_source == "learned"),
            )
        else:
            self.recovery_flow = None

        # === Complete-recovery baselines =================================
        # These arms are deliberately distinct from ReBound's shared-content
        # recovery.  They reconstruct either the entire missing feature
        # decomposition (content + modality-specific pyramids) or the raw
        # missing sensor image, and feed the result through the ordinary fused
        # prediction path with no reliability discount.
        self.complete_recovery_mode = str(
            getattr(self.config, "complete_recovery_mode", "none")
        )
        if self.complete_recovery_mode == "full_feature":
            bottleneck_ratio = int(
                getattr(self.config, "complete_feature_bottleneck_ratio", 8)
            )
            self.complete_sar_to_rgb = FullFeaturePyramidRecoverer(
                self.config.fuse_inchan_list, bottleneck_ratio
            )
            self.complete_rgb_to_sar = FullFeaturePyramidRecoverer(
                self.config.fuse_inchan_list, bottleneck_ratio
            )
        elif self.complete_recovery_mode == "full_image":
            width = int(getattr(self.config, "complete_image_width", 32))
            self.complete_sar_to_rgb = FullImageRecoverer(
                self.config.sar_in_channels, self.config.rgb_in_channels, width
            )
            self.complete_rgb_to_sar = FullImageRecoverer(
                self.config.rgb_in_channels, self.config.sar_in_channels, width
            )
        elif self.complete_recovery_mode != "none":
            raise ValueError(
                "complete_recovery_mode must be one of "
                "{'none', 'full_feature', 'full_image'}, got "
                f"{self.complete_recovery_mode!r}"
            )

        # === loss hyperparameters =========================================
        loss_cfg = self.config.loss
        self.evidence_activation = str(loss_cfg.get("evidence_activation", "softplus"))
        self.edl_form = str(loss_cfg.get("edl_form", "mse"))  # "mse" or "risk"
        self.ce_aux_weight = float(loss_cfg.get("ce_aux_weight", 0.0))
        self.kl_max_lambda = float(loss_cfg.get("kl_max_lambda", 1.0))
        self.kl_ramp_iters = int(loss_cfg.get("kl_ramp_iters", 5000))
        self.lam_mmr = float(loss_cfg.get("lam_mmr", 1.0))
        self.lam_mse = float(loss_cfg.get("lam_mse", 1.0))
        self.lam_recon = float(loss_cfg.get("lam_recon", 0.5))
        # Primary CE weights (S1).  fuse > sar > rgb because the fused head
        # is the test path for "fuse" mode and gets the strongest gradient.
        self.lam_ce_sar = float(loss_cfg.get("lam_ce_sar", 1.0))
        self.lam_ce_rgb = float(loss_cfg.get("lam_ce_rgb", 0.5))
        self.lam_ce_fuse = float(loss_cfg.get("lam_ce_fuse", 1.5))
        self.lam_ce_missing = float(loss_cfg.get("lam_ce_missing", 1.0))
        # #17: direct CE/NLL on the σ²-gated Dempster-fused expected prob
        # (α_dst/S) in the missing branch — "train what you read out" so the
        # gated-fused output is a competitive predictor, not just EDL-supervised.
        self.lam_ce_dst_rec = float(loss_cfg.get("lam_ce_dst_rec", 0.0))
        # #22: reliability-head regression loss (predict true restoration error).
        self.lam_rel = float(loss_cfg.get("lam_rel", 0.0))
        # Auxiliary EDL weights — smaller than the CE primary weights.
        self.lam_evi_sar = float(loss_cfg.get("lam_evi_sar", 0.3))
        self.lam_evi_rgb = float(loss_cfg.get("lam_evi_rgb", 0.2))
        self.lam_evi_fuse = float(loss_cfg.get("lam_evi_fuse", 0.3))
        self.lam_evi_missing = float(loss_cfg.get("lam_evi_missing", 0.3))
        self.lam_evi_fuse_rec = float(loss_cfg.get("lam_evi_fuse_rec", 0.3))
        self.lam_distill = float(loss_cfg.get("lam_distill", 0.0))
        self.lam_dice = float(loss_cfg.get("lam_dice", 0.3))
        self.lam_balance = float(loss_cfg.get("lam_balance", 0.05))
        # Anti-spillover consistency (ReBound-v4s): outside the occluded area the
        # fused prediction for the occluded input should match the (no-grad)
        # prediction for the unoccluded input.
        self.lam_spill = float(loss_cfg.get("lam_spill", 0.0))
        self.ignore_index = int(loss_cfg.get("ignore_index", -1))
        cw = loss_cfg.get("class_weights", None)
        if cw is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(list(cw), dtype=torch.float32),
                persistent=False,
            )
        else:
            self.class_weights = None
        self.ohem_keep_ratio = float(loss_cfg.get("ohem_keep_ratio", 1.0))
        # label smoothing for the logit cross-entropy heads (ce_sar/rgb/fuse/
        # missing). 0.0 = off (backwards-compatible with all prior runs); the
        # 0.5779 CMNeXtSGSLN recipe used 0.05.
        self.label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))

        # === inference mode ==============================================
        self.infer_mode = getattr(self.config, "infer_mode", "sar")
        # #17: missing-modality eval readout switch (the headline ablation).
        #   "head"     → softmax(available head logits)  [MetaRS-equivalent;
        #                 the σ² gate only feeds last_uncertainty → does NOT
        #                 touch mIoU]
        #   "dempster" → α_dst/S, the σ²-gated Dempster fusion of the available
        #                 evidence + the recovered evidence → the uncertainty
        #                 gate now drives the prediction (and thus mIoU).
        # Orthogonal to use_flow_recovery, so {head,dempster} × {flow,det} is a
        # clean 2×2 ablation.
        self.missing_readout = str(getattr(self.config, "missing_readout", "head"))
        self.infer_rgb = self.config.infer_rgb

        # === MSE between meta features (pillar 1's L_mse term) ============
        self.mse_loss = nn.MSELoss()

        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))
        self.gate_enabled = bool(
            getattr(self.config, "recovery_uncertainty_enabled", True)
        )  # False => beta_rec has no flow/reliability-map gate
        # eval-time toggle (𝒱-information harness): zero the *recovered* evidence
        # stream so Dempster fusion reduces to the available-modality-only readout
        # under IDENTICAL machinery (vacuous 2nd Dirichlet ⇒ Dempster identity).
        # Lets `avail`/`recovered`/`full` differ ONLY in the 2nd stream's content.
        self.zero_recovered_evidence = bool(
            getattr(self.config, "zero_recovered_evidence", False)
        )
        # hallucination harness (ReBound): override the recovered-stream reliability
        # to a constant. force_beta_rec=1.0 ⇒ recovered evidence at FULL weight =
        # naive full recovery (no boundedness); None ⇒ trained β_rec (bounded).
        self.force_beta_rec = None
        # full-recovery baseline (ReBound ablation): recovery_discount=False ⇒ recovered
        # evidence is used at FULL weight (β_rec=1) in BOTH train and eval ⇒ the model
        # learns to trust hallucinated content. Default True = bounded (ReBound).
        self.recovery_discount = bool(getattr(self.config, "recovery_discount", True))
        if not self.recovery_discount:
            self.force_beta_rec = 1.0
        self.last_uncertainty = None  # cached for eval-time heatmap dump
        self.last_beta_obs = None
        self.last_beta_rec = None
        self.last_beta_sar = None
        self.last_beta_rgb = None
        self.last_restore_sigma2 = None  # M5a: cached flow restoration σ² (validation harness)
        self.last_r_map = None  # M5a: cached learned reliability r(x) on the missing path
                                #      (selective-prediction / stratified-ECE deliverable)

        try:
            self.local_rank = int(os.environ["LOCAL_RANK"])
        except Exception:
            self.local_rank = None
        if self.local_rank in (None, 0):
            logger.info("[CoDER] initialised — K=%d, missing_modality=%s",
                        self.config.num_classes, self.missing_modality_training)

    def _compute_beta(
        self,
        recovered: bool = False,
        features=None,
        modality: str | None = None,
    ) -> torch.Tensor:
        """Return a reliability scalar with independently ablatable factors.

        Observed evidence uses beta_obs.  Recovered evidence uses
        beta_obs*gamma_rec when its extra reliability discount is enabled.
        ``recovery_discount=False`` is the genuine full-recovery arm and must
        return one during *training as well as evaluation*; the previous
        implementation only forced one in the eval path for deterministic
        recovery, so the purported full-recovery checkpoint was still trained
        with bounded evidence.
        """
        device = self.cov_reliability.gamma_rec_raw.device
        if (
            self.observed_reliability_enabled
            and self.observed_reliability_mode == "covariance_deviation"
        ):
            if features is None or modality is None:
                beta = torch.ones((), device=device)
            else:
                beta = self.observed_cov_reliability(features, modality)
        elif self.observed_reliability_enabled:
            beta = self.cov_reliability.compute(
                self.cov_matrix_layer, recovered=False
            )
        else:
            beta = torch.ones((), device=device)
            # Preserve a zero-gradient connection for DDP configurations that
            # keep find_unused_parameters disabled.
            beta = beta + 0.0 * self.cov_reliability.gamma_rec_raw

        if not recovered:
            return beta
        if not self.recovery_discount:
            return torch.ones_like(beta) + 0.0 * self.cov_reliability.gamma_rec_raw
        if self.recovered_reliability_enabled:
            beta = beta * self.cov_reliability.gamma_rec
        return beta

    def _combine_evidence(
        self, alpha_a: torch.Tensor, alpha_b: torch.Tensor, classes: int
    ) -> torch.Tensor:
        """Combine two Dirichlet streams with an ablatable conflict rule.

        The control arm pools evidence additively, preserving both streams and
        their evidence scale while removing Dempster's conflict redistribution.
        This makes ``dempster_enabled=False`` a one-factor conflict ablation.
        """
        if self.dempster_enabled:
            return dempster_combine(alpha_a, alpha_b, classes)
        return 1.0 + (alpha_a - 1.0) + (alpha_b - 1.0)

    def _combine_recovered(
        self,
        alpha_obs: torch.Tensor,
        alpha_rec: torch.Tensor,
        beta_rec: torch.Tensor,
        classes: int,
    ) -> torch.Tensor:
        """Combine an observed stream with evidence recovered *from it*.

        The recovered stream is a deterministic function of the observation,
        so it is not a distinct source.  The 'average' rule is a weighted
        average of evidence, e_f = (e_obs + beta_rec e_rec) / (1 + beta_rec),
        followed by a cap so the total evidence never exceeds the observed
        stream's.  Properties: (i) idempotent -- an exact copy leaves the
        opinion unchanged (Dempster would shrink its uncertainty to
        u^2/(1-C)); (ii) S_f <= S_obs, hence u_f >= u_obs; (iii) beta_rec -> 0
        recovers the observation-only opinion.  ``alpha_rec`` already carries
        the discount, i.e. alpha_rec - 1 = beta_rec * e_rec.
        """
        if getattr(self, "zero_recovered_evidence", False):
            # No recovered stream (R0): the observation is the whole opinion.
            return alpha_obs
        mode = self.recovered_combination
        if mode == "dempster":
            return self._combine_evidence(alpha_obs, alpha_rec, classes)
        if mode == "additive":
            return alpha_obs + alpha_rec - 1.0
        e_obs = alpha_obs - 1.0
        e_fused = (e_obs + (alpha_rec - 1.0)) / (1.0 + beta_rec)
        total_obs = e_obs.sum(dim=1, keepdim=True)
        total_fused = e_fused.sum(dim=1, keepdim=True).clamp_min(1e-8)
        e_fused = e_fused * torch.clamp(total_obs / total_fused, max=1.0)
        return e_fused + 1.0

    # ------------------------------------------------------------------
    # optimizer param groups — pretrained-ViT-aware LR + weight decay
    # ------------------------------------------------------------------
    def _transoss_lr_scale(self, name, flat_scale, layer_decay, n_layers=12):
        """LR multiplier for a TransOSS `en_content` parameter.

        New (random-init) sub-modules (`taps`, `tap_norms`) always train at the
        full base LR (scale 1.0).  The pretrained ViT trunk is slowed down — by
        a flat `flat_scale` (default) or, if `layer_decay` is given, by BEiT/MAE
        layer-wise LR decay so shallow blocks move least.
        """
        sub = name[len("en_content."):]
        if sub.startswith(("taps", "tap_norms", "adapter")):
            return 1.0
        if layer_decay is None:
            return flat_scale
        if sub.startswith("patch_embed") or sub in ("pos_embed", "cls_token", "mie_embed"):
            layer_id = 0
        elif sub.startswith("blocks."):
            layer_id = int(sub.split(".")[1]) + 1
        else:  # final `norm`, anything else at the trunk top
            layer_id = n_layers + 1
        max_layer = n_layers + 1
        return float(layer_decay) ** (max_layer - layer_id)

    def custom_param_groups(self):
        """Split parameters so the optimizer can (a) exclude norms / biases /
        positional & modality embeddings from weight decay, and (b) give the
        pretrained TransOSS ViT trunk a reduced LR via a per-group ``lr_scale``
        (honoured by the patched ``set_lr`` in train.py).

        For non-TransOSS backbones this still applies the universally-safe
        no-weight-decay split (lr_scale stays 1.0 everywhere), which is benign.
        """
        is_transoss = (self.backbone_name == "transoss")
        flat_scale = float(getattr(self.config, "vit_lr_scale", 0.1))
        layer_decay = getattr(self.config, "vit_layer_decay", None)

        groups: dict = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            no_wd = (
                p.ndim <= 1
                or name.endswith("pos_embed")
                or name.endswith("cls_token")
                or name.endswith("mie_embed")
            )
            if is_transoss and name.startswith("en_content."):
                lr_scale = self._transoss_lr_scale(name, flat_scale, layer_decay)
            else:
                lr_scale = 1.0
            key = (round(lr_scale, 6), no_wd)
            if key not in groups:
                g = {"params": [], "lr_scale": float(lr_scale)}
                if no_wd:
                    g["weight_decay"] = 0.0
                groups[key] = g
            groups[key]["params"].append(p)

        if self.local_rank in (None, 0):
            summary = sorted({k[0] for k in groups})
            logger.info("[CoDER] param groups: %d (lr_scales=%s, transoss=%s, layer_decay=%s)",
                        len(groups), summary, is_transoss, layer_decay)
        return list(groups.values())

    # ------------------------------------------------------------------
    # state-dict shim — preserves the global_step buffer across resume
    # ------------------------------------------------------------------
    def load_state_dict(self, state_dict, strict=True):
        from collections import OrderedDict
        if isinstance(state_dict, OrderedDict):
            state_dict = OrderedDict(state_dict)
        else:
            state_dict = dict(state_dict)
        has_global_step = any("global_step" in k for k in state_dict.keys())
        prefix = "module." if state_dict and list(state_dict.keys())[0].startswith("module.") else ""
        if not has_global_step:
            state_dict[f"{prefix}global_step"] = torch.tensor(0, dtype=torch.long)
        # Buffers added for the revision (fallback threshold, diagonal
        # reliability calibration) default to their inactive values when an
        # older checkpoint lacks them.
        for key, value in super().state_dict().items():
            if (prefix + key) not in state_dict and (
                key in ("fallback_tau", "fallback_msp_tau") or "_diag_" in key
                or key.startswith("rel_")
            ):
                state_dict[prefix + key] = value
        return super().load_state_dict(state_dict, strict=strict)

    # ------------------------------------------------------------------
    # per-modality forward helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_encoder_out(out):
        """Encoders return either `(features, whitening_positions)` or just
        the features list.  Return a uniform `(features, whitening)` pair."""
        if isinstance(out, tuple):
            features = out[0]
            whitening = out[1] if len(out) > 1 else []
        else:
            features = out
            whitening = []
        return features, whitening

    def _get_en_content_stem(self):
        """Return the trainable parameter holder for en_content's first conv —
        the layer we swap with rgb_conv1 to encode RGB through the shared
        meta encoder.  ResNet → `resnet.conv1`; ConvNeXt → `model.stem_0`.
        TransOSS handles modality routing internally and has no swappable stem
        — callers should use `_en_content_as_modality` instead."""
        if self.backbone_name == "transoss":
            return None, None
        if self.backbone_name in ("convnext_base", "convnext_large"):
            return self.en_content.model.stem_0, ("model", "stem_0")
        return self.en_content.resnet.conv1, ("resnet", "conv1")

    def _set_en_content_stem(self, new_conv):
        if self.backbone_name == "transoss":
            return
        if self.backbone_name in ("convnext_base", "convnext_large"):
            self.en_content.model.stem_0 = new_conv
        else:
            self.en_content.resnet.conv1 = new_conv

    @contextmanager
    def _en_content_as_modality(self, modality: str):
        """Context manager: temporarily configure `en_content` to encode the
        given modality.

        ResNet / ConvNeXt: stem conv is swapped with `rgb_conv1` for RGB; SAR
        uses the default stem (no-op).
        TransOSS: forwards `set_modality('sar'|'rgb')` to the dual-patch-embed
        ViT and restores the prior sticky modality on exit.
        """
        if self.backbone_name == "transoss":
            prev = self.en_content._sticky_modality
            self.en_content.set_modality(modality)
            try:
                yield
            finally:
                self.en_content.set_modality(prev)
        elif modality == "rgb":
            org_conv, _ = self._get_en_content_stem()
            self._set_en_content_stem(self._modules["rgb_conv1"])
            try:
                yield
            finally:
                self._set_en_content_stem(org_conv)
        else:
            # SAR on ResNet / ConvNeXt is the default stem — nothing to swap.
            yield

    def _encode_sar(self, sar):
        with self._en_content_as_modality("sar"):
            sar_content, sar_w = self._normalise_encoder_out(self.en_content(sar))
        if self.reliability_features == "content":
            sar_w = list(sar_content[:2])
        elif 2 not in self.config.encoder.wt_layer:
            sar_w = []
        sar_specific, _ = self._normalise_encoder_out(self.en_sar(sar))
        sar_fuse = self.Project(sar_specific, sar_content)
        return sar_content, sar_specific, sar_fuse, sar_w

    def _encode_rgb(self, rgb):
        with self._en_content_as_modality("rgb"):
            rgb_content, rgb_w = self._normalise_encoder_out(self.en_content(rgb))
        if self.reliability_features == "content":
            rgb_w = list(rgb_content[:2])
        elif 2 not in self.config.encoder.wt_layer:
            rgb_w = []
        rgb_specific, _ = self._normalise_encoder_out(self.en_rgb(rgb))
        rgb_fuse = self.Project(rgb_specific, rgb_content)
        return rgb_content, rgb_specific, rgb_fuse, rgb_w

    def _logits_and_evidence(self, decoder, fuse_features, content_features):
        """Run a per-modality decoder; return both raw logits and evidence.

        logits → primary softmax + CE  (raw network output, unbounded)
        evidence = softplus(logits) → α = β·e + 1 → auxiliary EDL path
        """
        logits = decoder(self.Concat(fuse_features, content_features))
        evidence = logits_to_evidence(logits, self.evidence_activation)
        return logits, evidence

    def _logits_and_evidence_fused(self, fused_features):
        """Run the fused decoder on a list of pyramid features."""
        logits = self.decoder_fused(fused_features)
        evidence = logits_to_evidence(logits, self.evidence_activation)
        return logits, evidence

    def _fusion_streams(self, content, projected):
        return projected if self.fused_input == "projected" else content

    @staticmethod
    def _mask_stream(stream, mask):
        """Zero the optical stream where the occlusion mask covers each cell."""
        if mask is None:
            return list(stream)
        return [f * (1.0 - pxs.pool_to(mask, f.shape[-2:]).to(f.dtype)) for f in stream]

    @torch.no_grad()
    def _clean_fused_prob(self, sar_content, sar_fuse, rgb_clean):
        """No-grad fused prediction for the unoccluded optical input (spill target).

        Modules run in eval mode so the target pass neither updates batch-norm
        statistics nor MoAR's load-balancing bias.
        """
        modules = [self.en_content, self.en_rgb, self.Project, self.MoE_Fuse, self.decoder_fused]
        modes = [m.training for m in modules]
        for m in modules:
            m.eval()
        try:
            rgb_content, _, rgb_fuse, _ = self._encode_rgb(rgb_clean)
            fused, _, _ = self.MoE_Fuse(
                [f.detach() for f in self._fusion_streams(sar_content, sar_fuse)],
                list(self._fusion_streams(rgb_content, rgb_fuse)))
            logits, _ = self._logits_and_evidence_fused(fused)
            return F.softmax(logits.float(), dim=1)
        finally:
            for m, mode in zip(modules, modes):
                m.train(mode)

    def _aligned_mse(self, a, b):
        """MSE between shared features, ignoring occluded optical cells."""
        if self._spatial_keep is None:
            return self.mse_loss(a, b)
        w = pxs.pool_to(self._spatial_keep, a.shape[-2:]).to(a.dtype)
        return ((a - b) ** 2 * w).sum() / (w.sum() * a.shape[1]).clamp_min(1.0)

    def _estimate_mask(self, rgb):
        """Unreliable optical area from local pixel statistics (inference)."""
        if self.external_mask is not None:
            return self.external_mask.to(rgb.dtype)
        if self.spatial_mask_infer != "pix" or not torch.isfinite(self.rel_loc_tau).all():
            return None
        dev = pxs.max_abs_z(pxs.local_stats(rgb), self.rel_loc_mu, self.rel_loc_sd)
        mask = (dev > self.rel_loc_tau).float().unsqueeze(1)
        return pxs.dilate(mask, self.mask_dilate)

    def _image_deviation(self, sar, rgb):
        """Image-level pixel-statistics deviation (max |z|) of each sensor."""
        dev_s = pxs.max_abs_z(pxs.image_stats(sar), self.rel_img_mu_s, self.rel_img_sd_s)
        dev_o = pxs.max_abs_z(pxs.image_stats(rgb), self.rel_img_mu_o, self.rel_img_sd_o)
        return dev_s, dev_o

    def _fused_logits_from_content(
        self, sar_content, rgb_content, checkpointed: bool = False
    ):
        """Run the normal paired-modality fusion head on two content pyramids."""
        def run_fusion(*flat_features):
            levels = len(flat_features) // 2
            sar_levels = list(flat_features[:levels])
            rgb_levels = list(flat_features[levels:])
            if self.use_moe:
                fused_pyramid, _, _ = self.MoE_Fuse(
                    sar_levels, rgb_levels
                )
            else:
                fused_pyramid = self.MoE_Fuse(sar_levels, rgb_levels)
            fused_logits, _ = self._logits_and_evidence_fused(fused_pyramid)
            return fused_logits

        flat = tuple(sar_content) + tuple(rgb_content)
        if checkpointed and self.training:
            # The clean fused loss already retains one full MoAR graph.  A
            # second graph for missing-modality complete recovery exceeds a
            # 96-GB GPU at batch 16.  Non-reentrant checkpointing preserves
            # the same function/gradients while recomputing this path.  Only
            # the clean fusion is checkpointed in complete-recovery training;
            # wrapping both calls to the same MoAR would create two reentrant
            # backward graphs over shared parameters, which DDP rejects.
            return checkpoint(run_fusion, *flat, use_reentrant=False)
        return run_fusion(*flat)

    def _complete_recover(
        self,
        direction: str,
        available_image,
        available_content,
        available_specific,
    ):
        """Recover the complete missing input and its feature decomposition.

        Returns ``(content, specific, fuse, recovered_image)``.  The last item
        is ``None`` for full-feature recovery.
        """
        if self.complete_recovery_mode == "none":
            raise RuntimeError("_complete_recover called with mode='none'")
        if direction == "sar_to_rgb":
            recoverer = self.complete_sar_to_rgb
            encoder = self._encode_rgb
        elif direction == "rgb_to_sar":
            recoverer = self.complete_rgb_to_sar
            encoder = self._encode_sar
        else:
            raise ValueError(f"unknown complete-recovery direction {direction!r}")

        if self.complete_recovery_mode == "full_feature":
            recovered_content, recovered_specific = recoverer(
                list(available_content), list(available_specific)
            )
            recovered_image = None
            recovered_fuse = self.Project(
                recovered_specific, recovered_content
            )
        else:
            recovered_image = recoverer(available_image)
            recovered_content, recovered_specific, recovered_fuse, _ = encoder(
                recovered_image
            )
        return (
            recovered_content,
            recovered_specific,
            recovered_fuse,
            recovered_image,
        )

    def _complete_reconstruction_loss(
        self,
        recovered_content,
        recovered_specific,
        target_content,
        target_specific,
        recovered_image=None,
        target_image=None,
    ):
        """Supervise all recovered dimensions, not only a shared subspace."""
        if self.complete_recovery_mode == "full_image":
            if recovered_image is None or target_image is None:
                raise ValueError("full-image recovery requires image targets")
            return F.smooth_l1_loss(recovered_image, target_image)

        losses = []
        for recovered, target in zip(recovered_content, target_content):
            losses.append(F.smooth_l1_loss(recovered, target.detach()))
        for recovered, target in zip(recovered_specific, target_specific):
            losses.append(F.smooth_l1_loss(recovered, target.detach()))
        return sum(losses) / max(len(losses), 1)

    def _flow_recover(self, decoder, available_fuse, cond_content, beta_rec):
        """M5a latent rectified-flow recovery.

        Restore the *missing* modality's content pyramid from the available
        modality's content pyramid (``cond_content``), run ``decoder`` on the
        restored content + the available fused features, and fold the
        per-image restoration uncertainty ``sigma2`` into the reliability
        discount::

            beta_rec_eff = beta_rec * exp(-sigma2 / flow_tau)

        Returns ``(logits_rec, e_rec, beta_rec_eff, sigma2)``.
        """
        z_hat, sigma2, sigma2_map = self.recovery_flow.restore(list(cond_content))
        logits_rec, e_rec = self._logits_and_evidence(decoder, available_fuse, z_hat)
        # reliability signal r(x) feeding the discount β_rec = β·exp(−r/τ):
        #   "learned" → head's predicted error (primary);  else σ² (ablation).
        r_map = None
        if self.reliability_source == "learned":
            # h = reliability logit (higher ⇒ more error); gate = sigmoid(−h/τ).
            r_map = self.recovery_flow.predict_reliability(z_hat, list(cond_content), sigma2_map)
            # DETACH into the gate: the head is a pure read-out predictor trained
            # ONLY by L_rel (rec["r_map"] stays connected for that loss).  Letting
            # ce_dst_rec backprop through the gate makes seg-accuracy and error-
            # fidelity fight: a fresh rel-only head probes ρ=0.51 on these exact
            # features, but the leaky-gate head collapsed to ρ=0.23.  Detaching
            # restores the head to its pure-predictor role (probe-proven).
            rel = r_map.detach()
            if rel.shape[-2:] != e_rec.shape[-2:]:
                rel = F.interpolate(rel, size=e_rec.shape[-2:], mode="bilinear", align_corners=False)
            gate = self._reliability_gate(rel)
        else:
            # seed-variance σ² (ablation): r ≥ 0, gate = exp(−σ²/τ).
            rel = sigma2_map if self.flow_spatial_sigma else sigma2
            if rel.dim() == 4 and rel.shape[-2:] != e_rec.shape[-2:]:
                rel = F.interpolate(rel, size=e_rec.shape[-2:], mode="bilinear", align_corners=False)
            gate = torch.exp(-rel / self.flow_tau)
        # Eval-time ablation toggle (claim-2/3 harness): gate_enabled=False makes
        # β_rec_eff = β_rec (const-β arm, no reliability discount).  Default True
        # ⇒ no change to training or normal eval.
        beta_rec_eff = beta_rec * gate if getattr(self, "gate_enabled", True) else beta_rec
        if not getattr(self, "recovery_discount", True):   # full-recovery baseline
            beta_rec_eff = torch.ones_like(beta_rec_eff)
        return {
            "logits": logits_rec,
            "evidence": e_rec,
            "beta_rec": beta_rec_eff,
            "sigma2": sigma2,            # global scalar (logging / last_restore_sigma2)
            "sigma2_map": sigma2_map,    # per-position σ²(x)
            "z_hat": z_hat,             # restored content pyramid (for L_rel target)
            "r_map": r_map,             # learned reliability prediction (or None)
        }

    def _recover_content(self, decoder, available_fuse, available_content, beta_rec):
        """Recovery-slot dispatcher for the Phase-1 method-breadth sweep.

        ``self.recovery_method`` (getattr-default ``'auto'`` ⇒ unchanged behaviour)
        selects how the *missing* modality's content pyramid is produced before the
        decoder turns it into evidence — so every recovery method shares the SAME
        encoder / evidential decoder / Dempster eval machinery and only the recovery
        differs:
            'auto' : flow if use_flow_recovery else det   (preserves prior eval/train)
            'flow' : latent rectified-flow restoration     (learned recovery)
            'det'  : deterministic — reuse available content (MetaRS-style passthrough)
            'zero' : trivial zero-fill of the missing content
            'mean' : trivial spatial-mean-fill (per-image, per-level)
        Returns the same dict shape as ``_flow_recover`` (sigma2 / r_map None for the
        non-flow methods).
        """
        method = getattr(self, "recovery_method", "auto")
        if method == "auto":
            method = "flow" if self.use_flow_recovery else "det"
        if method == "flow":
            return self._flow_recover(decoder, available_fuse, available_content, beta_rec)
        if method == "ddpm":
            z_hat, sigma2, sigma2_map = self.recovery_ddpm.restore(list(available_content))
            _, e_rec = self._logits_and_evidence(decoder, available_fuse, z_hat)
            return {"evidence": e_rec, "beta_rec": beta_rec, "sigma2": sigma2,
                    "sigma2_map": sigma2_map, "z_hat": z_hat, "r_map": None}
        if method == "det":
            content = list(available_content)
        elif method == "zero":
            content = [torch.zeros_like(c) for c in available_content]
        elif method == "mean":
            content = [c.mean(dim=(2, 3), keepdim=True).expand_as(c) for c in available_content]
        else:
            raise ValueError(f"unknown recovery_method {method!r}")
        _, e_rec = self._logits_and_evidence(decoder, available_fuse, content)
        return {"evidence": e_rec, "beta_rec": beta_rec, "sigma2": None,
                "sigma2_map": None, "z_hat": content, "r_map": None}

    def _reliability_gate(self, rel):
        """Map reliability signal r(x) → multiplicative gate on β_rec.

        Eval-time swappable knob for the claim-2 *calibration-aware gate* sweep.
        Training and normal eval use the default ``gate_form='sigmoid'`` which is
        bit-identical to the original ``sigmoid(−r/τ)`` (discount-only), so this
        method changes nothing unless the eval harness overrides the attributes.

        The trained reliability head's output ``r`` is ≈ N(0,1) (the distilled
        head folds its standardisation into the final conv), so ``gate_c`` shifts
        the operating point and ``flow_tau`` sets the sharpness.

          ``sigmoid``  : ``sigmoid((c−r)/τ)``               ∈ (0,1)   discount-only (default)
          ``sigmoid2`` : ``gmax·sigmoid((c−r)/τ)``          ∈ (0,gmax) two-sided, centre gmax/2
          ``exp2``     : ``exp((c−r)/τ).clamp(gmin,gmax)``   two-sided, centre exp(c/τ)
          ``scale``    : constant ``g`` (ignores r)         global-temperature ablation

        A discount-only gate can only *lower* confidence; when the base model is
        already under-confident (meanconf < acc on SAR-only) that *raises* ECE.
        The two-sided forms let reliable positions *raise* confidence, and the
        ``scale`` form isolates how much of any gain is a plain global rescale
        vs. the reliability signal redistributing confidence per-position.
        """
        form = getattr(self, "gate_form", "sigmoid")
        tau = self.flow_tau
        c = float(getattr(self, "gate_c", 0.0))
        if form == "sigmoid":
            return torch.sigmoid((c - rel) / tau)
        if form == "sigmoid2":
            g = float(getattr(self, "gate_gmax", 2.0))
            return g * torch.sigmoid((c - rel) / tau)
        if form == "exp2":
            gmin = float(getattr(self, "gate_gmin", 0.3))
            gmax = float(getattr(self, "gate_gmax", 3.0))
            return torch.exp((c - rel) / tau).clamp(gmin, gmax)
        if form == "scale":
            g = float(getattr(self, "gate_gmax", 1.0))
            return torch.full_like(rel, g)
        raise ValueError(f"unknown gate_form {form!r}")

    def _ce_loss(self, logits, cls_true):
        """Standard cross-entropy with optional class weights + OHEM."""
        cw = None
        if self.class_weights is not None:
            cw = self.class_weights.to(logits.device).float()
        if self.ohem_keep_ratio < 1.0:
            # OHEM at sample-pixel level: compute per-pixel CE, keep topk
            losses = F.cross_entropy(
                logits, cls_true.long(),
                ignore_index=self.ignore_index,
                weight=cw,
                reduction='none',
                label_smoothing=self.label_smoothing,
            )
            valid_mask = (cls_true.long() != self.ignore_index)
            flat_loss = losses[valid_mask]
            if flat_loss.numel() == 0:
                return torch.zeros((), device=logits.device, requires_grad=True)
            n_keep = max(1, int(flat_loss.numel() * self.ohem_keep_ratio))
            topk, _ = torch.topk(flat_loss, n_keep)
            return topk.mean()
        return F.cross_entropy(
            logits, cls_true.long(),
            ignore_index=self.ignore_index,
            weight=cw,
            label_smoothing=self.label_smoothing,
        )

    def _ce_on_prob(self, prob, cls_true):
        """NLL on an already-normalised probability (the Dempster-fused expected
        categorical α_dst/S).  Mirrors ``_ce_loss`` (class weights + OHEM +
        ignore_index) but consumes probabilities, not logits — so the output we
        *read out* at eval under ``missing_readout='dempster'`` is the same
        thing we *train*.
        """
        log_p = torch.log(prob.clamp_min(1e-8))
        cw = None
        if self.class_weights is not None:
            cw = self.class_weights.to(prob.device).float()
        if self.ohem_keep_ratio < 1.0:
            losses = F.nll_loss(
                log_p, cls_true.long(),
                ignore_index=self.ignore_index,
                weight=cw,
                reduction='none',
            )
            valid_mask = (cls_true.long() != self.ignore_index)
            flat_loss = losses[valid_mask]
            if flat_loss.numel() == 0:
                return torch.zeros((), device=prob.device, requires_grad=True)
            n_keep = max(1, int(flat_loss.numel() * self.ohem_keep_ratio))
            topk, _ = torch.topk(flat_loss, n_keep)
            return topk.mean()
        return F.nll_loss(
            log_p, cls_true.long(),
            ignore_index=self.ignore_index,
            weight=cw,
        )

    # ------------------------------------------------------------------
    # main forward
    # ------------------------------------------------------------------
    def forward(self, x, y=None):
        if self.training:
            self._bump_global_step()
            if self.apply_mmr and self.global_step.item() == self.config.begin_mmr_iter:
                self.conduct_mask_matrix()
            # Keep β-ramp in sync with training progress.
            self.cov_reliability.set_ramp_step(int(self.global_step.item()))
            return self._forward_train(x, y)
        return self._forward_eval(x, y)

    # ------------------------------------------------------------------
    def _bump_global_step(self):
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                self.global_step += 1
            dist.broadcast(self.global_step, src=0)
        else:
            self.global_step += 1

    # ------------------------------------------------------------------
    def _forward_train(self, x, y):
        sar = x[:, : self.config.sar_in_channels, :, :]
        rgb = x[
            :,
            self.config.sar_in_channels : self.config.sar_in_channels
            + self.config.rgb_in_channels,
            :,
            :,
        ]
        cls_true = y["cls"]
        K = self.config.num_classes
        occlusion = None
        cls_rgb = cls_true
        self._spatial_keep = None
        if self.spatial_dropout_prob > 0:
            chosen = torch.rand(rgb.shape[0], device=rgb.device) < self.spatial_dropout_prob
            if bool(chosen.any()):
                occlusion = pxs.blob_masks(rgb.shape[0], rgb.shape[2], rgb.shape[3], rgb.device)
                occlusion = occlusion * chosen.view(-1, 1, 1, 1).float()
                rgb_clean = rgb
                if self.occlusion_mode == "cloud":
                    # Plain augmentation: opaque-to-thick cloud, no mask information.
                    opacity = 0.6 + 0.4 * torch.rand(rgb.shape[0], 1, 1, 1, device=rgb.device)
                    alpha = occlusion * opacity
                    rgb = rgb * (1.0 - alpha) + self.cloud_value.view(1, -1, 1, 1) * alpha
                    occlusion = None
                else:
                    rgb = rgb * (1.0 - occlusion)          # remove occluded optical content
                    self._spatial_keep = 1.0 - occlusion
                    cls_rgb = cls_true.clone()
                    cls_rgb[occlusion.squeeze(1) > 0.5] = self.ignore_index

        # Always encode both during training so MMR/L_mse have signal.
        sar_content, sar_specific, sar_fuse, sar_w = self._encode_sar(sar)
        rgb_content, rgb_specific, rgb_fuse, rgb_w = self._encode_rgb(rgb)

        # Primary per-modality heads (logits) + auxiliary evidence -----------
        sar_logits, e_sar = self._logits_and_evidence(
            self.decoder_sar, sar_fuse, sar_content
        )
        rgb_logits, e_rgb = self._logits_and_evidence(
            self.decoder_rgb, rgb_fuse, rgb_content
        )

        # Cross-modal fused content features → fused decoder (primary path) ---
        balance_loss_val = None
        fused_logits = None
        if self.decoder_fused is not None:
            checkpoint_clean_fusion = bool(
                getattr(
                    self.config,
                    "complete_checkpoint_clean_fusion",
                    False,
                )
            )
            if checkpoint_clean_fusion:
                if self.fusion_type != "moar":
                    raise ValueError(
                        "complete_checkpoint_clean_fusion currently requires "
                        "fusion_type='moar' (whose balance loss is None)"
                    )
                fused_logits = self._fused_logits_from_content(
                    self._fusion_streams(sar_content, sar_fuse),
                    self._fusion_streams(rgb_content, rgb_fuse),
                    checkpointed=True,
                )
            elif self.use_moe:
                fused_pyramid, _router_weights, balance_loss_val = self.MoE_Fuse(
                    list(self._fusion_streams(sar_content, sar_fuse)),
                    self._mask_stream(self._fusion_streams(rgb_content, rgb_fuse), occlusion),
                )
            else:
                # Fusespfnv2 signature: (unimodal_list, share_list) → fused_list
                fused_pyramid = self.MoE_Fuse(
                    list(self._fusion_streams(sar_content, sar_fuse)),
                    list(self._fusion_streams(rgb_content, rgb_fuse)),
                )
            if not checkpoint_clean_fusion:
                fused_logits, _ = self._logits_and_evidence_fused(fused_pyramid)

        # Pillar-2 discount on per-modality evidence -------------------------
        beta_sar = self._compute_beta(
            recovered=False, features=sar_w, modality="sar"
        )
        beta_rgb = self._compute_beta(
            recovered=False, features=rgb_w, modality="rgb"
        )
        alpha_sar = evidence_to_alpha(discount_evidence(e_sar, beta_sar))
        alpha_rgb = evidence_to_alpha(discount_evidence(e_rgb, beta_rgb))
        # Pillar-3 Dempster on the two evidence streams (auxiliary uncertainty)
        alpha_dst = self._combine_evidence(alpha_sar, alpha_rgb, K)

        # KL annealing schedule for the EDL auxiliary --------------------------
        kl_lambda = kl_annealing(
            int(self.global_step.item()), self.kl_ramp_iters, self.kl_max_lambda
        )

        def _edl(alpha):
            return edl_loss(
                alpha, cls_true, K, kl_lambda, self.ignore_index,
                form=self.edl_form, ce_aux_weight=self.ce_aux_weight,
                class_weights=self.class_weights,
                ohem_keep_ratio=self.ohem_keep_ratio,
            )

        loss_dict = {}
        # ---- Primary classification losses (CE on softmax) -----------------
        loss_dict["ce_sar_loss"] = self.lam_ce_sar * self._ce_loss(sar_logits, cls_true)
        loss_dict["ce_rgb_loss"] = self.lam_ce_rgb * self._ce_loss(rgb_logits, cls_rgb)
        if fused_logits is not None:
            loss_dict["ce_fuse_loss"] = self.lam_ce_fuse * self._ce_loss(
                fused_logits, cls_true
            )
            if self.lam_spill > 0 and occlusion is not None:
                target = self._clean_fused_prob(sar_content, sar_fuse, rgb_clean)
                # non-occluded pixels of occluded samples only
                keep = (occlusion < 0.5).float() * chosen.view(-1, 1, 1, 1).float()
                log_p = F.log_softmax(fused_logits.float(), dim=1)
                kl = (target * (torch.log(target.clamp_min(1e-8)) - log_p)).sum(1, keepdim=True)
                loss_dict["spill_loss"] = self.lam_spill * (kl * keep).sum() / keep.sum().clamp_min(1.0)

        # ---- Auxiliary EDL losses (uncertainty path) -----------------------
        if self.lam_evi_sar > 0:
            loss_dict["evi_sar_loss"] = self.lam_evi_sar * _edl(alpha_sar)
        if self.lam_evi_rgb > 0:
            loss_dict["evi_rgb_loss"] = self.lam_evi_rgb * _edl(alpha_rgb)
        if self.lam_evi_fuse > 0:
            loss_dict["evi_dst_loss"] = self.lam_evi_fuse * _edl(alpha_dst)

        # ---- Boundary dice on softmax(fused_logits) -----------------------
        if self.lam_dice > 0 and fused_logits is not None:
            loss_dict["dice_loss"] = self.lam_dice * soft_dice_loss(
                F.softmax(fused_logits, dim=1), cls_true, K, self.ignore_index
            )

        # ---- MoE balance loss (push toward uniform expert usage) -----------
        if balance_loss_val is not None and self.lam_balance > 0:
            loss_dict["balance_loss"] = self.lam_balance * balance_loss_val

        # ---- Missing-modality branch ---------------------------------------
        if self.missing_modality_training:
            mode = self._sample_missing()
            if mode == [1, 0]:  # RGB missing — recover from SAR (eval-equivalent path)
                complete_recon_loss = None
                if self.complete_recovery_mode != "none":
                    (
                        rgb_content_rec,
                        rgb_specific_rec,
                        rgb_fuse_rec,
                        rgb_image_rec,
                    ) = self._complete_recover(
                        "sar_to_rgb", sar, sar_content, sar_specific
                    )
                    rgb_logits_rec, e_rgb_rec = self._logits_and_evidence(
                        self.decoder_rgb, rgb_fuse_rec, rgb_content_rec
                    )
                    beta_rec = torch.ones_like(beta_sar)
                    complete_recon_loss = self._complete_reconstruction_loss(
                        rgb_content_rec,
                        rgb_specific_rec,
                        rgb_content,
                        rgb_specific,
                        recovered_image=rgb_image_rec,
                        target_image=rgb,
                    )
                else:
                    beta_rec = self._compute_beta(
                        recovered=True, features=sar_w, modality="sar"
                    )
                if (
                    self.complete_recovery_mode == "none"
                    and self.use_flow_recovery
                ):
                    rec = self._flow_recover(
                        self.decoder_rgb, sar_fuse, sar_content, beta_rec
                    )
                    rgb_logits_rec, e_rgb_rec, beta_rec = (
                        rec["logits"], rec["evidence"], rec["beta_rec"])
                    if self.lam_flow > 0:
                        loss_dict["flow_loss"] = self.lam_flow * \
                            self.recovery_flow.flow_matching_loss(
                                list(rgb_content), list(sar_content)
                            )
                    if self.lam_rel > 0 and rec["r_map"] is not None:
                        err_t = self._restoration_err_map(
                            rec["z_hat"], rgb_content, rec["r_map"].shape[-2:])
                        loss_dict["rel_loss"] = self.lam_rel * F.smooth_l1_loss(
                            rec["r_map"], err_t)
                elif self.complete_recovery_mode == "none":
                    rgb_logits_rec, e_rgb_rec = self._logits_and_evidence(
                        self.decoder_rgb, sar_fuse, sar_content
                    )
                if self.zero_recovered_evidence:
                    e_rgb_rec = torch.zeros_like(e_rgb_rec)
                alpha_rec = evidence_to_alpha(discount_evidence(e_rgb_rec, beta_rec))
                if self.lam_ce_missing > 0:
                    loss_dict["ce_missing_loss"] = self.lam_ce_missing * self._ce_loss(
                        rgb_logits_rec, cls_true
                    )
                if self.lam_evi_missing > 0:
                    loss_dict["evi_missing_loss"] = self.lam_evi_missing * _edl(alpha_rec)
                if self.lam_evi_fuse_rec > 0 or self.lam_ce_dst_rec > 0:
                    alpha_dst_rec = self._combine_recovered(alpha_sar, alpha_rec, beta_rec, K)
                    if self.lam_evi_fuse_rec > 0:
                        loss_dict["evi_dst_rec_loss"] = self.lam_evi_fuse_rec * _edl(
                            alpha_dst_rec
                        )
                    if self.lam_ce_dst_rec > 0:
                        loss_dict["ce_dst_rec_loss"] = self.lam_ce_dst_rec * \
                            self._ce_on_prob(alpha_to_prob(alpha_dst_rec), cls_true)
                if complete_recon_loss is not None:
                    loss_dict["recon_loss"] = (
                        self.lam_recon * complete_recon_loss
                    )
                else:
                    loss_dict["recon_loss"] = self.lam_recon * self._recon_loss(
                        [sar_content], rgb_content
                    )
            elif mode == [0, 1]:  # SAR missing — recover from RGB
                complete_recon_loss = None
                if self.complete_recovery_mode != "none":
                    (
                        sar_content_rec,
                        sar_specific_rec,
                        sar_fuse_rec,
                        sar_image_rec,
                    ) = self._complete_recover(
                        "rgb_to_sar", rgb, rgb_content, rgb_specific
                    )
                    sar_logits_rec, e_sar_rec = self._logits_and_evidence(
                        self.decoder_sar, sar_fuse_rec, sar_content_rec
                    )
                    beta_rec = torch.ones_like(beta_rgb)
                    complete_recon_loss = self._complete_reconstruction_loss(
                        sar_content_rec,
                        sar_specific_rec,
                        sar_content,
                        sar_specific,
                        recovered_image=sar_image_rec,
                        target_image=sar,
                    )
                else:
                    beta_rec = self._compute_beta(
                        recovered=True, features=rgb_w, modality="rgb"
                    )
                if (
                    self.complete_recovery_mode == "none"
                    and self.use_flow_recovery
                ):
                    rec = self._flow_recover(
                        self.decoder_sar, rgb_fuse, rgb_content, beta_rec
                    )
                    sar_logits_rec, e_sar_rec, beta_rec = (
                        rec["logits"], rec["evidence"], rec["beta_rec"])
                    if self.lam_flow > 0:
                        loss_dict["flow_loss"] = self.lam_flow * \
                            self.recovery_flow.flow_matching_loss(
                                list(sar_content), list(rgb_content)
                            )
                    if self.lam_rel > 0 and rec["r_map"] is not None:
                        err_t = self._restoration_err_map(
                            rec["z_hat"], sar_content, rec["r_map"].shape[-2:])
                        loss_dict["rel_loss"] = self.lam_rel * F.smooth_l1_loss(
                            rec["r_map"], err_t)
                elif self.complete_recovery_mode == "none":
                    sar_logits_rec, e_sar_rec = self._logits_and_evidence(
                        self.decoder_sar, rgb_fuse, rgb_content
                    )
                if self.zero_recovered_evidence:
                    e_sar_rec = torch.zeros_like(e_sar_rec)
                alpha_rec = evidence_to_alpha(discount_evidence(e_sar_rec, beta_rec))
                if self.lam_ce_missing > 0:
                    loss_dict["ce_missing_loss"] = self.lam_ce_missing * self._ce_loss(
                        sar_logits_rec, cls_rgb  # read from (possibly occluded) optical
                    )
                if self.lam_evi_missing > 0:
                    loss_dict["evi_missing_loss"] = self.lam_evi_missing * _edl(alpha_rec)
                if self.lam_evi_fuse_rec > 0 or self.lam_ce_dst_rec > 0:
                    alpha_dst_rec = self._combine_recovered(alpha_rgb, alpha_rec, beta_rec, K)
                    if self.lam_evi_fuse_rec > 0:
                        loss_dict["evi_dst_rec_loss"] = self.lam_evi_fuse_rec * _edl(
                            alpha_dst_rec
                        )
                    if self.lam_ce_dst_rec > 0:
                        loss_dict["ce_dst_rec_loss"] = self.lam_ce_dst_rec * \
                            self._ce_on_prob(alpha_to_prob(alpha_dst_rec), cls_true)
                if complete_recon_loss is not None:
                    loss_dict["recon_loss"] = (
                        self.lam_recon * complete_recon_loss
                    )
                else:
                    loss_dict["recon_loss"] = self.lam_recon * self._recon_loss(
                        [rgb_content], sar_content
                    )
            # complete mode: nothing extra

        # MMR / MSE (pillar-1 anchors, kept from MetaRS) ---------------------
        if "mmr" in self.config.loss and self.apply_mmr and self.global_step.item() > self.config.begin_mmr_iter:
            if any(layer.mask_matrix is None for layer in self.cov_matrix_layer):
                self.conduct_mask_matrix()
            loss_dict["mmr_loss"] = self.lam_mmr * self._mmr_loss(sar_w, rgb_w).mean()
            if "mse" in self.config.loss:
                alpha_mse = float(self.config.loss.get("alpha", 1.0))
                loss_dict["mse_loss"] = self.lam_mse * alpha_mse * (
                    sum(self._aligned_mse(s, r) for s, r in zip(sar_content[2:], rgb_content[2:])) / 2.0
                )
        elif "mse" in self.config.loss:
            # warmup phase: still anchor meta features
            loss_dict["mse_loss"] = self.lam_mse * 0.1 * (
                sum(self._aligned_mse(s, r) for s, r in zip(sar_content[2:], rgb_content[2:])) / 2.0
            )

        loss_dict["mem"] = torch.from_numpy(
            np.array([torch.cuda.max_memory_allocated() // 1024 // 1024], dtype=np.float32)
        ).to(x.device)
        return loss_dict

    # ------------------------------------------------------------------
    def _sample_missing(self):
        return self.missing_modality_sampler([], None, None)

    def _valid_mask(self, cls_true):
        v = (cls_true.long() != self.ignore_index)
        return cls_true, v

    def _restoration_err_map(self, z_hat_list, true_content, ref_hw):
        """#22 L_rel target: the per-image **z-scored** restoration error on the
        σ²_map grid (raw err = mean_l ‖(ẑ−z_true)/std‖²).  Standardising per image
        is load-bearing: regressing the *raw* (heavy-tailed) error makes the Huber
        head collapse to the median ⇒ an inert gate (τ-sweep, 2026-06-07); the
        z-scored target preserves rank/spread (matches the ρ≈0.53 probe).
        Detached — a fixed target so L_rel trains *only* the head."""
        b = z_hat_list[0].shape[0]
        err = torch.zeros(b, 1, *ref_hw, device=z_hat_list[0].device)
        for zh, zt in zip(z_hat_list, true_content):
            std = zt.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
            se = (((zh.detach() - zt.detach()) / std) ** 2).mean(1, keepdim=True)
            if se.shape[-2:] != ref_hw:
                se = F.interpolate(se, size=ref_hw, mode="bilinear", align_corners=False)
            err = err + se
        err = err / max(len(z_hat_list), 1)
        mu = err.mean(dim=(1, 2, 3), keepdim=True)
        sd = err.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        return ((err - mu) / sd).detach()

    def _recon_loss(self, available_meta_features_list, ground_truth_meta_features):
        # mean across available modalities at each FPN level, MSE vs truth.
        recovered = []
        for layer_feats in zip(*available_meta_features_list):
            recovered.append(sum(layer_feats) / len(layer_feats))
        return sum(
            self._aligned_mse(rec, gt) for rec, gt in zip(recovered, ground_truth_meta_features)
        ) / max(len(recovered), 1)

    # ------------------------------------------------------------------
    def _forward_eval(self, x, y):
        """Eval primary classification + auxiliary uncertainty.

        Primary output (used by the mIoU metric):
            mode='sar'/'rgb' → missing_readout='head'    : softmax(available head)
                               missing_readout='dempster': α_dst/S (σ²-gated fusion)
            mode='fuse'      → softmax(fused_logits)   (uses MoE_Fuse + decoder_fused)

        Auxiliary output (cached for ECE / §5.4 heatmaps):
            self.last_uncertainty = K / S_alpha
            where α comes from the matching evidence head; in 'sar' mode the
            Dempster combination of (α_sar, α_rgb_recovered) is also computed
            so that the uncertainty map reflects the full evidential path.
        """
        mode = self.infer_mode
        K = self.config.num_classes

        if mode == "fuse":
            sar = x[:, : self.config.sar_in_channels, :, :]
            rgb = x[
                :,
                self.config.sar_in_channels : self.config.sar_in_channels
                + self.config.rgb_in_channels,
                :,
                :,
            ]
            rgb_observed = rgb
            occlusion = self._estimate_mask(rgb)
            self.last_mask_cover = None
            if occlusion is not None:
                self.last_mask_cover = occlusion.flatten(1).mean(1)
                rgb = rgb * (1.0 - occlusion)
            sar_content, _, sar_fuse, sar_w = self._encode_sar(sar)
            rgb_content, _, rgb_fuse, rgb_w = self._encode_rgb(rgb)
            # Primary: fused decoder (MoE or Fusespfnv2)
            if self.decoder_fused is not None:
                sar_stream = list(self._fusion_streams(sar_content, sar_fuse))
                rgb_stream = self._mask_stream(self._fusion_streams(rgb_content, rgb_fuse), occlusion)
                if self.use_moe:
                    fused_pyramid, _, _ = self.MoE_Fuse(sar_stream, rgb_stream)
                else:
                    fused_pyramid = self.MoE_Fuse(sar_stream, rgb_stream)
                fused_logits, _ = self._logits_and_evidence_fused(fused_pyramid)
                prob = F.softmax(fused_logits, dim=1)
            else:
                # Fallback (ablation): average of per-mod softmaxes
                sar_logits, _ = self._logits_and_evidence(self.decoder_sar, sar_fuse, sar_content)
                rgb_logits, _ = self._logits_and_evidence(self.decoder_rgb, rgb_fuse, rgb_content)
                prob = 0.5 * (F.softmax(sar_logits, dim=1) + F.softmax(rgb_logits, dim=1))
            # Auxiliary uncertainty via Dempster of evidence streams
            sar_logits_aux, e_sar = self._logits_and_evidence(self.decoder_sar, sar_fuse, sar_content)
            rgb_logits_aux, e_rgb = self._logits_and_evidence(self.decoder_rgb, rgb_fuse, rgb_content)
            beta_sar = self._compute_beta(
                recovered=False, features=sar_w, modality="sar"
            )
            beta_rgb = self._compute_beta(
                recovered=False, features=rgb_w, modality="rgb"
            )
            alpha_sar = evidence_to_alpha(discount_evidence(e_sar, beta_sar))
            alpha_rgb = evidence_to_alpha(discount_evidence(e_rgb, beta_rgb))
            alpha_dst = self._combine_evidence(alpha_sar, alpha_rgb, K)
            batch = prob.shape[0]
            route = torch.zeros(batch, dtype=torch.long, device=prob.device)
            # Always-fuse readout, kept so an evaluator can score the gated and
            # ungated readouts on the same forward pass.
            self.last_prob_fused = prob.clone()
            self.last_alpha_fused = alpha_dst.clone()
            # Single-sensor head outputs on the same forward pass (these are
            # exactly the missing-optical / missing-SAR read-outs).
            self.last_prob_sar_head = F.softmax(sar_logits_aux, dim=1)
            self.last_prob_rgb_head = F.softmax(rgb_logits_aux, dim=1)
            if self.reliability_fallback:
                b_sar = beta_sar.reshape(-1).expand(batch) if beta_sar.numel() == 1 else beta_sar.reshape(batch)
                b_rgb = beta_rgb.reshape(-1).expand(batch) if beta_rgb.numel() == 1 else beta_rgb.reshape(batch)
                bad_rgb = b_rgb < self.fallback_tau[1]
                if self.fallback_rule == "pix_or_conf":
                    # Model-free image statistics OR head confidence, per sensor.
                    dev_s, dev_o = self._image_deviation(sar, rgb_observed)
                    msp = lambda logits: F.softmax(logits.float(), dim=1).max(dim=1).values.flatten(1).mean(1)
                    bad_sar = (dev_s > self.rel_img_tau[0]) | (msp(sar_logits_aux) < self.fallback_msp_tau[0])
                    if self.last_mask_cover is not None:
                        # Hierarchical: a pixel mask already removes local damage, so
                        # the whole optical image is dropped only when the mask covers
                        # (nearly) all of it -- which is what global degradations do.
                        bad_rgb = self.last_mask_cover > self.mask_max_cover
                    else:
                        bad_rgb = (dev_o > self.rel_img_tau[1]) | (msp(rgb_logits_aux) < self.fallback_msp_tau[1])
                elif self.fallback_rule == "hybrid":
                    msp_sar = F.softmax(sar_logits_aux.float(), dim=1).max(dim=1).values.flatten(1).mean(1)
                    bad_sar = msp_sar < self.fallback_msp_tau[0]
                else:
                    bad_sar = b_sar < self.fallback_tau[0]
                route[bad_rgb & ~bad_sar] = 1   # optical unreliable -> SAR head
                route[bad_sar & ~bad_rgb] = 2   # SAR unreliable -> optical head
                for code, logits, alpha_obs, decoder, fuse, content, beta_obs in (
                    (1, sar_logits_aux, alpha_sar, self.decoder_rgb, sar_fuse, sar_content, b_sar),
                    (2, rgb_logits_aux, alpha_rgb, self.decoder_sar, rgb_fuse, rgb_content, b_rgb),
                ):
                    sel = route == code
                    if not bool(sel.any()):
                        continue
                    # Single-modality readout for the routed images, exactly as
                    # when the other modality is absent.
                    prob[sel] = F.softmax(logits[sel], dim=1)
                    modality = "sar" if code == 1 else "rgb"
                    features = sar_w if code == 1 else rgb_w
                    beta_rec = self._compute_beta(recovered=True, features=features, modality=modality)
                    rec = self._recover_content(decoder, fuse, content, beta_rec)
                    beta_rec = rec["beta_rec"]
                    alpha_rec = evidence_to_alpha(discount_evidence(rec["evidence"], beta_rec))
                    alpha_single = self._combine_recovered(alpha_obs, alpha_rec, beta_rec, K)
                    alpha_dst[sel] = alpha_single[sel]
            self.last_route = route
            self.last_uncertainty = alpha_to_uncertainty(alpha_dst, K, dim=1).squeeze(1)
            self.last_evi_prob = alpha_to_prob(alpha_dst, dim=1)  # evidential predictive prob α/S
            self.last_alpha = alpha_dst  # Dirichlet params, for post-hoc temperature scaling
            self.last_beta_obs = 0.5 * (beta_sar.detach() + beta_rgb.detach())
            self.last_beta_rec = None
            self.last_beta_sar = beta_sar.detach()
            self.last_beta_rgb = beta_rgb.detach()
            return prob

        if mode == "full":
            # Full-modality evidential control for the cross-domain calibration
            # audit: identical Dempster readout as `sar`/`rgb`, but with the REAL
            # RGB evidence instead of the flow-recovered one → isolates how much
            # of the per-city calibration regime-flip is *caused by* the missing
            # modality (recovered RGB) vs. plain geographic domain shift.
            sar = x[:, : self.config.sar_in_channels, :, :]
            rgb = x[:, self.config.sar_in_channels:
                    self.config.sar_in_channels + self.config.rgb_in_channels, :, :]
            sar_content, _, sar_fuse, sar_w = self._encode_sar(sar)
            rgb_content, _, rgb_fuse, rgb_w = self._encode_rgb(rgb)
            _, e_sar = self._logits_and_evidence(self.decoder_sar, sar_fuse, sar_content)
            _, e_rgb = self._logits_and_evidence(self.decoder_rgb, rgb_fuse, rgb_content)
            beta_sar = self._compute_beta(
                recovered=False, features=sar_w, modality="sar"
            )
            beta_rgb = self._compute_beta(
                recovered=False, features=rgb_w, modality="rgb"
            )
            alpha_sar = evidence_to_alpha(discount_evidence(e_sar, beta_sar))
            alpha_rgb = evidence_to_alpha(discount_evidence(e_rgb, beta_rgb))
            alpha_dst = self._combine_evidence(alpha_sar, alpha_rgb, K)
            self.last_uncertainty = alpha_to_uncertainty(alpha_dst, K, dim=1).squeeze(1)
            self.last_evi_prob = alpha_to_prob(alpha_dst, dim=1)  # evidential predictive prob α/S
            self.last_alpha = alpha_dst  # Dirichlet params, for post-hoc temperature scaling
            self.last_r_map = None
            self.last_beta_obs = 0.5 * (beta_sar.detach() + beta_rgb.detach())
            self.last_beta_rec = None
            self.last_beta_sar = beta_sar.detach()
            self.last_beta_rgb = beta_rgb.detach()
            return alpha_to_prob(alpha_dst, dim=1)

        if mode == "sar":
            sar = x if x.size(1) == self.config.sar_in_channels else x[:, : self.config.sar_in_channels, :, :]
            sar_content, sar_specific, sar_fuse, sar_w = self._encode_sar(sar)
            # Primary: SAR head (MetaRS-equivalent path)
            sar_logits, e_sar = self._logits_and_evidence(self.decoder_sar, sar_fuse, sar_content)
            head_prob = F.softmax(sar_logits, dim=1)
            # σ²-gated Dempster fusion of (α_sar, α_rgb_rec) — drives both the
            # uncertainty map and (when missing_readout='dempster') the prediction.
            beta_in = self._compute_beta(
                recovered=False, features=sar_w, modality="sar"
            )
            beta_rec = self._compute_beta(
                recovered=True, features=sar_w, modality="sar"
            )
            if self.complete_recovery_mode != "none":
                (
                    rgb_content_rec,
                    _,
                    rgb_fuse_rec,
                    _,
                ) = self._complete_recover(
                    "sar_to_rgb", sar, sar_content, sar_specific
                )
                complete_logits = self._fused_logits_from_content(
                    self._fusion_streams(sar_content, sar_fuse),
                    self._fusion_streams(rgb_content_rec, rgb_fuse_rec),
                )
                head_prob = F.softmax(complete_logits, dim=1)
                _, e_rgb_rec = self._logits_and_evidence(
                    self.decoder_rgb, rgb_fuse_rec, rgb_content_rec
                )
                beta_rec = torch.ones_like(beta_in)
                self.last_restore_sigma2 = None
                self.last_r_map = None
            else:
                rec = self._recover_content(
                    self.decoder_rgb, sar_fuse, sar_content, beta_rec
                )
                e_rgb_rec, beta_rec = rec["evidence"], rec["beta_rec"]
                if self.force_beta_rec is not None:
                    # hallucination harness: full-weight shared recovery
                    beta_rec = torch.full_like(
                        beta_rec, float(self.force_beta_rec)
                    )
                self.last_restore_sigma2 = (
                    rec["sigma2"].detach()
                    if rec["sigma2"] is not None
                    else None
                )
                self.last_r_map = (
                    rec["r_map"].detach()
                    if rec["r_map"] is not None
                    else None
                )
            if getattr(self, "zero_recovered_evidence", False):
                e_rgb_rec = torch.zeros_like(e_rgb_rec)   # vacuous 2nd stream → avail-only
            alpha_sar = evidence_to_alpha(discount_evidence(e_sar, beta_in))
            alpha_rgb_rec = evidence_to_alpha(discount_evidence(e_rgb_rec, beta_rec))
            alpha_dst = self._combine_recovered(alpha_sar, alpha_rgb_rec, beta_rec, K)
            self.last_uncertainty = alpha_to_uncertainty(alpha_dst, K, dim=1).squeeze(1)
            self.last_evi_prob = alpha_to_prob(alpha_dst, dim=1)  # evidential predictive prob α/S
            self.last_alpha = alpha_dst  # Dirichlet params, for post-hoc temperature scaling
            self.last_beta_obs = beta_in.detach()
            self.last_beta_rec = beta_rec.detach()
            self.last_beta_sar = beta_in.detach()
            self.last_beta_rgb = None
            if (
                self.complete_recovery_mode == "none"
                and self.missing_readout == "dempster"
            ):
                return alpha_to_prob(alpha_dst, dim=1)
            return head_prob

        if mode == "rgb":
            rgb = x if x.size(1) == self.config.rgb_in_channels else x[
                :,
                self.config.sar_in_channels : self.config.sar_in_channels
                + self.config.rgb_in_channels,
                :,
                :,
            ]
            rgb_content, rgb_specific, rgb_fuse, rgb_w = self._encode_rgb(rgb)
            rgb_logits, e_rgb = self._logits_and_evidence(self.decoder_rgb, rgb_fuse, rgb_content)
            head_prob = F.softmax(rgb_logits, dim=1)
            beta_in = self._compute_beta(
                recovered=False, features=rgb_w, modality="rgb"
            )
            beta_rec = self._compute_beta(
                recovered=True, features=rgb_w, modality="rgb"
            )
            if self.complete_recovery_mode != "none":
                (
                    sar_content_rec,
                    _,
                    sar_fuse_rec,
                    _,
                ) = self._complete_recover(
                    "rgb_to_sar", rgb, rgb_content, rgb_specific
                )
                complete_logits = self._fused_logits_from_content(
                    self._fusion_streams(sar_content_rec, sar_fuse_rec),
                    self._fusion_streams(rgb_content, rgb_fuse),
                )
                head_prob = F.softmax(complete_logits, dim=1)
                _, e_sar_rec = self._logits_and_evidence(
                    self.decoder_sar, sar_fuse_rec, sar_content_rec
                )
                beta_rec = torch.ones_like(beta_in)
                self.last_restore_sigma2 = None
                self.last_r_map = None
            else:
                rec = self._recover_content(
                    self.decoder_sar, rgb_fuse, rgb_content, beta_rec
                )
                e_sar_rec, beta_rec = rec["evidence"], rec["beta_rec"]
                if self.force_beta_rec is not None:
                    # hallucination harness: full-weight shared recovery
                    beta_rec = torch.full_like(
                        beta_rec, float(self.force_beta_rec)
                    )
                self.last_restore_sigma2 = (
                    rec["sigma2"].detach()
                    if rec["sigma2"] is not None
                    else None
                )
                self.last_r_map = (
                    rec["r_map"].detach()
                    if rec["r_map"] is not None
                    else None
                )
            if getattr(self, "zero_recovered_evidence", False):
                e_sar_rec = torch.zeros_like(e_sar_rec)   # vacuous 2nd stream → avail-only
            alpha_rgb = evidence_to_alpha(discount_evidence(e_rgb, beta_in))
            alpha_sar_rec = evidence_to_alpha(discount_evidence(e_sar_rec, beta_rec))
            alpha_dst = self._combine_recovered(alpha_rgb, alpha_sar_rec, beta_rec, K)
            self.last_uncertainty = alpha_to_uncertainty(alpha_dst, K, dim=1).squeeze(1)
            self.last_evi_prob = alpha_to_prob(alpha_dst, dim=1)  # evidential predictive prob α/S
            self.last_alpha = alpha_dst  # Dirichlet params, for post-hoc temperature scaling
            self.last_beta_obs = beta_in.detach()
            self.last_beta_rec = beta_rec.detach()
            self.last_beta_sar = None
            self.last_beta_rgb = beta_in.detach()
            if (
                self.complete_recovery_mode == "none"
                and self.missing_readout == "dempster"
            ):
                return alpha_to_prob(alpha_dst, dim=1)
            return head_prob

        raise ValueError(f"Unknown infer_mode: {mode}")

    @torch.no_grad()
    def extract_probe_features(self, x):
        """Frozen features for the probe-based 𝒱-information sweep (RGB-missing path).

        Returns three feature pyramids that differ ONLY in the missing-modality
        content (absent / flow-recovered / real) — so a fixed external probe
        trained on each isolates the *usable* information M̂ vs M add over A:
            'A'     : decoder-input features from the available modality only
                      (= Concat(sar_fuse, sar_content), what decoder_sar sees)
            'AMhat' : 'A' ⊕ recovered missing content  z_hat = g(A)
            'AM'    : 'A' ⊕ real missing content        rgb_content
        AMhat and AM share channel structure (both append a content pyramid) so
        the only difference is recovered-vs-real.
        """
        sar = x[:, : self.config.sar_in_channels]
        rgb = x[:, self.config.sar_in_channels:
                self.config.sar_in_channels + self.config.rgb_in_channels]
        sar_content, _, sar_fuse, _ = self._encode_sar(sar)
        rgb_content, _, _, _ = self._encode_rgb(rgb)
        z_hat, _, _ = self.recovery_flow.restore(list(sar_content))
        base = list(self.Concat(sar_fuse, sar_content))         # pyramid (list)
        return {
            "A": base,
            "AMhat": [torch.cat([b, z], 1) for b, z in zip(base, z_hat)],
            "AM": [torch.cat([b, r], 1) for b, r in zip(base, rgb_content)],
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def restoration_diagnostics(self, x):
        """Validation harness (M5a claim 1): does the restoration uncertainty
        σ² track the true restoration error?

        Restores the RGB content pyramid from SAR via the flow, and compares it
        to the *real* RGB content (both modalities are present in ``x`` at
        validation time).  Returns per-image::

            sigma2 : [B]  restoration uncertainty (seed variance, scale-free)
            err    : [B]  restoration error  mean_l ||(z_hat - z_true)/std_true||²

        ``err`` is normalised by the per-image target std so it lives in the
        same scale-free space as σ² — a positive σ²↔err correlation (high
        Spearman ρ, low AUSE) is the evidence that σ² is a faithful reliability
        signal.  Requires ``use_flow_recovery``.
        """
        assert self.use_flow_recovery, "restoration_diagnostics needs use_flow_recovery=True"
        sar = x[:, : self.config.sar_in_channels, :, :]
        rgb = x[
            :,
            self.config.sar_in_channels : self.config.sar_in_channels
            + self.config.rgb_in_channels,
            :, :,
        ]
        sar_content, _, _, _ = self._encode_sar(sar)
        rgb_content, _, _, _ = self._encode_rgb(rgb)        # ground-truth content
        z_hat, sigma2, _sigma2_map = self.recovery_flow.restore(list(sar_content))
        err = torch.zeros(x.shape[0], device=x.device)
        for zh, zt in zip(z_hat, rgb_content):
            std = zt.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
            err = err + (((zh - zt) / std) ** 2).mean(dim=(1, 2, 3))
        err = err / max(len(z_hat), 1)
        return sigma2.flatten(), err

    @torch.no_grad()
    def restoration_diagnostics_spatial(self, x, max_points: int = 20000):
        """#18 claim-1 (spatial): does **per-position** σ²(x) track per-position
        restoration error?

        Restores the RGB content from SAR, and at the finest pyramid grid (where
        σ²(x) lives) compares the per-position seed variance σ²(x) to the
        per-position standardised restoration error err(x) = channel-mean
        ‖(ẑ − z_true)/std‖².  Returns flattened, sub-sampled ``(σ²(x), err(x))``
        across all images×positions — feed to the same Spearman/AUSE metrics.

        Caveat: neighbouring positions are spatially correlated ⇒ the effective
        sample size is below ``max_points``; the rank metrics are still valid
        faithfulness measures, just not i.i.d.
        """
        assert self.use_flow_recovery, "needs use_flow_recovery=True"
        sar = x[:, : self.config.sar_in_channels, :, :]
        rgb = x[
            :,
            self.config.sar_in_channels : self.config.sar_in_channels
            + self.config.rgb_in_channels,
            :, :,
        ]
        sar_content, _, _, _ = self._encode_sar(sar)
        rgb_content, _, _, _ = self._encode_rgb(rgb)
        z_hat, _sig, sigma2_map = self.recovery_flow.restore(list(sar_content))
        ref_hw = sigma2_map.shape[-2:]
        err_map = torch.zeros(x.shape[0], 1, *ref_hw, device=x.device)
        for zh, zt in zip(z_hat, rgb_content):
            std = zt.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
            se = (((zh - zt) / std) ** 2).mean(dim=1, keepdim=True)  # [B,1,H_l,W_l]
            if se.shape[-2:] != ref_hw:
                se = F.interpolate(se, size=ref_hw, mode="bilinear", align_corners=False)
            err_map = err_map + se
        err_map = err_map / max(len(z_hat), 1)
        s2 = sigma2_map.flatten()
        er = err_map.flatten()
        if s2.numel() > max_points:
            idx = torch.randperm(s2.numel(), device=s2.device)[:max_points]
            s2 = s2[idx]
            er = er[idx]
        return s2, er

    def restoration_diagnostics_learned(self, x, max_points: int = 20000):
        """#22 claim-1 (learned head): does the **trained reliability head**'s
        per-position prediction r(x) track per-position restoration error?

        Identical setup to :meth:`restoration_diagnostics_spatial`, but the
        reliability *signal* is the learned head output ``r(x) =
        predict_reliability(ẑ, sar_content, σ²(x))`` (higher ⇒ more predicted
        error), not the raw seed variance.  This is the headline faithfulness
        test for the learned gate; the seed-variance method is the ablation.

        Returns flattened, sub-sampled ``(r(x), err(x))`` — feed to the same
        Spearman/AUSE metrics.  ``ρ>0`` means the head ranks error correctly.
        """
        assert self.use_flow_recovery, "needs use_flow_recovery=True"
        assert self.reliability_source == "learned", \
            "ckpt has no trained reliability_head (reliability_source != 'learned')"
        sar = x[:, : self.config.sar_in_channels, :, :]
        rgb = x[
            :,
            self.config.sar_in_channels : self.config.sar_in_channels
            + self.config.rgb_in_channels,
            :, :,
        ]
        sar_content, _, _, _ = self._encode_sar(sar)
        rgb_content, _, _, _ = self._encode_rgb(rgb)
        z_hat, _sig, sigma2_map = self.recovery_flow.restore(list(sar_content))
        r_map = self.recovery_flow.predict_reliability(
            z_hat, list(sar_content), sigma2_map)        # [B,1,H0,W0]
        ref_hw = r_map.shape[-2:]
        err_map = torch.zeros(x.shape[0], 1, *ref_hw, device=x.device)
        for zh, zt in zip(z_hat, rgb_content):
            std = zt.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
            se = (((zh - zt) / std) ** 2).mean(dim=1, keepdim=True)
            if se.shape[-2:] != ref_hw:
                se = F.interpolate(se, size=ref_hw, mode="bilinear", align_corners=False)
            err_map = err_map + se
        err_map = err_map / max(len(z_hat), 1)
        rr = r_map.flatten()
        er = err_map.flatten()
        if rr.numel() > max_points:
            idx = torch.randperm(rr.numel(), device=rr.device)[:max_points]
            rr = rr[idx]
            er = er[idx]
        return rr, er

    # ------------------------------------------------------------------
    # MMR plumbing — identical to MetaRS, kept here so CoDER is self-contained
    # ------------------------------------------------------------------
    def set_mask_matrix(self):
        for layer in self.cov_matrix_layer:
            layer.set_mask_matrix()

    def reset_mask_matrix(self):
        for layer in self.cov_matrix_layer:
            layer.reset_mask_matrix()

    def conduct_mask_matrix(self):
        mask_dataloader = make_dataloader(self.config.data)
        en_content_was_training = self.en_content.training
        en_sar_was_training = self.en_sar.training
        en_rgb_was_training = self.en_rgb.training
        self.en_content.eval()
        self.reset_mask_matrix()
        try:
            with torch.no_grad():
                for img, _ in tqdm(mask_dataloader):
                    img = img.to(self.cov_reliability.gamma_rec_raw.device)
                    sar = img[:, : self.config.sar_in_channels, :, :]
                    rgb = img[:, self.config.sar_in_channels :, :, :]
                    with self._en_content_as_modality("sar"):
                        _, sar_w_arr = self.en_content(sar)
                    with self._en_content_as_modality("rgb"):
                        _, rgb_w_arr = self.en_content(rgb)
                    for index, (sar_f_map, rgb_f_map) in enumerate(
                        zip(sar_w_arr, rgb_w_arr)
                    ):
                        B, C, H, W = sar_f_map.shape
                        HW = H * W
                        sar_f_map = sar_f_map.contiguous().view(B, C, -1)
                        rgb_f_map = rgb_f_map.contiguous().view(B, C, -1)
                        eye, reverse_eye = self.cov_matrix_layer[index].get_eye_matrix()
                        for b in range(B):
                            f_map = torch.concat(
                                [
                                    torch.unsqueeze(sar_f_map[b], dim=0),
                                    torch.unsqueeze(rgb_f_map[b], dim=0),
                                ],
                                dim=0,
                            )
                            f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW) + (
                                self.eps * eye
                            )
                            off_diag_elements = f_cor * reverse_eye
                            self.cov_matrix_layer[index].set_variance_of_covariance(
                                torch.var(off_diag_elements, dim=0)
                            )
                self.set_mask_matrix()
        finally:
            self.en_content.train(en_content_was_training)
            self.en_sar.train(en_sar_was_training)
            self.en_rgb.train(en_rgb_was_training)

    def _mmr_loss(self, sar_w_arr, rgb_w_arr):
        wt_loss = torch.zeros(1, device=self.cov_reliability.gamma_rec_raw.device)
        for index, (sar_f_map, rgb_f_map) in enumerate(zip(sar_w_arr, rgb_w_arr)):
            eye, mask_matrix, _, num_sensitive = self.cov_matrix_layer[
                index
            ].get_mask_matrix()
            wt_loss = (
                wt_loss
                + instance_whitening_loss(sar_f_map, eye, mask_matrix, num_sensitive)
                + instance_whitening_loss(rgb_f_map, eye, mask_matrix, num_sensitive)
            )
        return wt_loss / (len(sar_w_arr) * 2)

    # ------------------------------------------------------------------
    def set_default_config(self):
        self.config.update(
            dict(
                encoder=dict(
                    resnet_type="resnet50",
                    include_conv5=True,
                    batchnorm_trainable=True,
                    pretrained=True,
                    load_rgb=0,
                    freeze_at=0,
                    output_stride=32,
                    with_cp=(False, False, False, False),
                    stem3_3x3=False,
                    norm_layer=nn.BatchNorm2d,
                    wt=[0, 0, 0, 0, 0, 0, 0],
                ),
                fpn=dict(
                    in_channels_list=(256, 512, 1024, 256),
                    out_channels=256,
                    top_blocks=None,
                ),
                decoder=dict(
                    in_channels=256,
                    out_channels=128,
                    in_feat_output_strides=(4, 8, 16, 32),
                    out_feat_output_stride=4,
                    norm_fn=nn.BatchNorm2d,
                    num_groups_gn=None,
                ),
                loss=dict(),
            )
        )
