"""MoAR + DMFIA fusion, ported from the CMNeXtSGSLN clean-SOTA baseline
(`earthmiss_baseline/model.py`) into CoDER as a drop-in replacement for FpnMoE.

Self-contained (torch only).  `MoARFpnFuse` matches FpnMoE's call contract:
    forward(sar_list, rgb_list) -> (fused_list, router_weights_or_None, balance_or_None)
with `len == len(inchan_list)` and per-level output channels == input channels,
so CoDER's `decoder_fused` (FPN in_channels_list = fuse_inchan_list) is unchanged.

Design note (unification principle: improve clean WITHOUT hurting SAR-only):
CoDER's SAR-only eval path does NOT use this module, so swapping it cannot
*directly* affect SAR-only.  DMFIA refinement here is applied to COPIES of the
content pyramids used only for the fused path, so the per-modality EDL/Dempster
path is left untouched; the only coupling left is the shared encoder's gradient,
which we verify empirically by evaluating SAR-only before/after.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


# --------------------------------------------------------------------------- #
# basic blocks
# --------------------------------------------------------------------------- #
class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class PointwiseNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class ChannelGate(nn.Sequential):
    """Squeeze-and-excitation channel attention to refine aggregated/fused features."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        hidden = max(channels // reduction, 8)
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )


# --------------------------------------------------------------------------- #
# MoAR experts (each proposes a cross-modal rectification residual pair)
# --------------------------------------------------------------------------- #
class ChannelAttentionExpert(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.channels = channels
        hidden = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels * 2, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, rgb: torch.Tensor, sar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.gate(torch.cat([rgb, sar], dim=1))
        rgb_w, sar_w = torch.split(weight, self.channels, dim=1)
        return sar * sar_w, rgb * rgb_w


class SpatialAttentionExpert(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, 2, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, rgb: torch.Tensor, sar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.gate(torch.cat([rgb, sar], dim=1))
        rgb_w, sar_w = torch.split(weight, 1, dim=1)
        return sar * sar_w, rgb * rgb_w


class CrossTokenAttentionExpert(nn.Module):
    def __init__(self, channels: int, pooled: int = 16) -> None:
        super().__init__()
        self.pooled = pooled
        self.scale = channels ** -0.5
        self.q_rgb = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.q_sar = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.kv_rgb = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.kv_sar = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)

    def _attend(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        b, c, h, w = query.shape
        q = query.flatten(2).transpose(1, 2)
        k, v = kv.flatten(2).chunk(2, dim=1)
        attn = torch.softmax(q @ k * self.scale, dim=-1)
        return (attn @ v.transpose(1, 2)).transpose(1, 2).reshape(b, c, h, w)

    def forward(self, rgb: torch.Tensor, sar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        size = rgb.shape[2:]
        pooled = (min(self.pooled, size[0]), min(self.pooled, size[1]))
        rgb_small = F.adaptive_avg_pool2d(rgb, pooled)
        sar_small = F.adaptive_avg_pool2d(sar, pooled)
        rgb_delta = self._attend(self.q_rgb(rgb_small), self.kv_sar(sar_small))
        sar_delta = self._attend(self.q_sar(sar_small), self.kv_rgb(rgb_small))
        rgb_delta = F.interpolate(rgb_delta, size=size, mode="bilinear", align_corners=False)
        sar_delta = F.interpolate(sar_delta, size=size, mode="bilinear", align_corners=False)
        return rgb_delta, sar_delta


class NullExpert(nn.Module):
    def forward(self, rgb: torch.Tensor, sar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros_like(rgb), torch.zeros_like(sar)


class MoARectifyModule(nn.Module):
    """MoAR: Mixture-of-Attention Rectification with scene-adaptive top-k routing
    and DeepSeek-V3 aux-loss-free load balancing (bias affects selection only)."""

    def __init__(self, channels: int, num_experts: int = 3, top_k: int = 2,
                 reduction: int = 4, router_noise: float = 0.0,
                 bias_update_speed: float = 0.001, gate_mode: str = "renorm",
                 null_expert: bool = False) -> None:
        super().__init__()
        expert_factories = [
            lambda: ChannelAttentionExpert(channels),
            lambda: SpatialAttentionExpert(channels),
            lambda: CrossTokenAttentionExpert(channels),
            lambda: NullExpert(),
        ]
        if not 1 <= num_experts <= len(expert_factories):
            raise ValueError(f"num_experts must be in [1, {len(expert_factories)}], got {num_experts}")
        # gate_mode='renorm' (legacy): softmax over the selected logits only; with
        #   top_k=1 the gate is identically 1, so the router receives no task gradient.
        # gate_mode='switch': gate = softmax over ALL logits, read at the selected
        #   indices (Switch Transformer), so the task loss trains the router.
        if gate_mode not in ("renorm", "switch"):
            raise ValueError(f"unknown gate_mode {gate_mode!r}")
        self.gate_mode = gate_mode
        # null_expert=True appends an explicit zero-residual option after the
        # num_experts attention experts (the legacy path only had it when
        # num_experts=4).
        if null_expert and num_experts < len(expert_factories):
            factories = expert_factories[:num_experts] + [expert_factories[-1]]
        else:
            factories = expert_factories[:num_experts]
        num_experts = len(factories)
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")
        self.top_k = top_k
        self.num_experts = num_experts
        self.router_noise = router_noise
        self.bias_update_speed = bias_update_speed
        hidden = max(channels // reduction, 8)
        self.proj = nn.Sequential(nn.Linear(channels, hidden), nn.GELU())
        self.keys = nn.Parameter(torch.randn(num_experts, hidden) * hidden ** -0.5)
        self.experts = nn.ModuleList(factory() for factory in factories)
        self.register_buffer("expert_bias", torch.zeros(num_experts))

    def forward(self, rgb: torch.Tensor, sar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = F.adaptive_avg_pool2d(rgb + sar, 1).flatten(1)
        logits = self.proj(context) @ self.keys.t()
        if self.training and self.router_noise > 0:
            logits = logits + torch.randn_like(logits) * self.router_noise
        select_scores = logits + self.expert_bias
        top_indices = select_scores.topk(self.top_k, dim=-1).indices
        gates = logits.new_zeros(logits.shape)
        if self.gate_mode == "switch":
            top_probs = torch.softmax(logits.float(), dim=-1).gather(1, top_indices)
        else:
            # softmax runs in fp32 under bf16 autocast; cast back so scatter_'s
            # self/src dtypes match (scatter_ requires exact dtype equality).
            top_probs = torch.softmax(logits.gather(1, top_indices), dim=-1)
        gates.scatter_(1, top_indices, top_probs.to(gates.dtype))
        self.last_top_indices = top_indices.detach()
        if self.training:
            with torch.no_grad():
                load = torch.zeros_like(self.expert_bias)
                load.scatter_add_(0, top_indices.flatten(),
                                  torch.ones_like(top_indices.flatten(), dtype=load.dtype))
                load = load / top_indices.numel()
                self.expert_bias += self.bias_update_speed * torch.sign(1.0 / self.num_experts - load)
        rgb_out, sar_out = rgb, sar
        for index, expert in enumerate(self.experts):
            gate = gates[:, index]
            if not torch.any(gate > 0):
                continue
            rgb_delta, sar_delta = expert(rgb, sar)
            gate = gate.view(-1, 1, 1, 1)
            rgb_out = rgb_out + gate * rgb_delta
            sar_out = sar_out + gate * sar_delta
        return rgb_out, sar_out


class CrossModalFusionBlock(nn.Module):
    """Rectify both modalities (MoAR), then fuse and refine to one feature."""

    def __init__(self, channels: int, moa_top_k: int = 1, moa_num_experts: int = 3,
                 gate_mode: str = "renorm", null_expert: bool = False) -> None:
        super().__init__()
        self.rectify = MoARectifyModule(channels, num_experts=moa_num_experts, top_k=moa_top_k,
                                        gate_mode=gate_mode, null_expert=null_expert)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.refine = ChannelGate(channels)

    def forward(self, rgb: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        rgb_rectified, sar_rectified = self.rectify(rgb, sar)
        fused = self.fuse(torch.cat([rgb_rectified, sar_rectified], dim=1))
        return fused * self.refine(fused)


# --------------------------------------------------------------------------- #
# DMFIA: dual-modal multi-level feature interaction + aggregation
# --------------------------------------------------------------------------- #
class CrossLevelInteraction(nn.Module):
    """Project every *other* level onto a target level, attention-weighted aggregate."""

    def __init__(self, channels: tuple[int, ...], target_level: int) -> None:
        super().__init__()
        self.target_level = target_level
        self.target_channels = channels[target_level]
        self.projectors = nn.ModuleList(
            PointwiseNormAct(source_channels, self.target_channels)
            if level != target_level else nn.Identity()
            for level, source_channels in enumerate(channels)
        )
        hidden = max(self.target_channels // 4, 8)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.target_channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, feats: list[torch.Tensor], target_size: tuple[int, int]) -> torch.Tensor:
        projected = []
        for level, feat in enumerate(feats):
            if level == self.target_level:
                continue
            context = self.projectors[level](feat)
            if context.shape[2:] != target_size:
                context = F.interpolate(context, size=target_size, mode="bilinear", align_corners=False)
            projected.append(context)
        if not projected:
            reference = feats[self.target_level]
            return reference.new_zeros(reference.shape[0], self.target_channels, *target_size)
        stacked = torch.stack(projected, dim=0)
        scores = torch.stack([self.attention(context) for context in projected], dim=0)
        weights = torch.softmax(scores, dim=0)
        return (stacked * weights).sum(dim=0)


class DMFIAAggregator(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.refine = ChannelGate(channels)
        self.act = nn.GELU()

    def forward(self, residual, same_modal_same_level, same_modal_cross_level,
                cross_modal_same_level, cross_modal_cross_level) -> torch.Tensor:
        context = torch.cat([same_modal_same_level, same_modal_cross_level,
                             cross_modal_same_level, cross_modal_cross_level], dim=1)
        mixed = self.mix(context)
        mixed = mixed * self.refine(mixed)
        return self.act(residual + mixed)


class DMFIAUnit(nn.Module):
    """Dual-Modal Multi-Level Feature Interaction and Aggregation unit."""

    def __init__(self, channels: tuple[int, ...]) -> None:
        super().__init__()
        if not channels:
            raise ValueError("DMFIAUnit requires at least one feature level")
        self.channels = channels
        num_levels = len(channels)
        self.rgb_same_level = nn.ModuleList(ConvNormAct(ch, ch) for ch in channels)
        self.sar_same_level = nn.ModuleList(ConvNormAct(ch, ch) for ch in channels)
        self.rgb_same_modal_cross_level = nn.ModuleList(
            CrossLevelInteraction(channels, level) for level in range(num_levels))
        self.sar_same_modal_cross_level = nn.ModuleList(
            CrossLevelInteraction(channels, level) for level in range(num_levels))
        self.rgb_from_sar_same_level = nn.ModuleList(PointwiseNormAct(ch, ch) for ch in channels)
        self.sar_from_rgb_same_level = nn.ModuleList(PointwiseNormAct(ch, ch) for ch in channels)
        self.rgb_from_sar_cross_level = nn.ModuleList(
            CrossLevelInteraction(channels, level) for level in range(num_levels))
        self.sar_from_rgb_cross_level = nn.ModuleList(
            CrossLevelInteraction(channels, level) for level in range(num_levels))
        self.rgb_aggregators = nn.ModuleList(DMFIAAggregator(ch) for ch in channels)
        self.sar_aggregators = nn.ModuleList(DMFIAAggregator(ch) for ch in channels)

    def forward(self, rgb_feats, sar_feats):
        if len(rgb_feats) != len(self.channels) or len(sar_feats) != len(self.channels):
            raise ValueError(
                f"DMFIAUnit expected {len(self.channels)} levels, "
                f"got {len(rgb_feats)} RGB and {len(sar_feats)} SAR")
        next_rgb, next_sar = [], []
        for level, (rgb_feat, sar_feat) in enumerate(zip(rgb_feats, sar_feats)):
            target_size = rgb_feat.shape[2:]
            rgb_same_cross = self.rgb_same_modal_cross_level[level](rgb_feats, target_size)
            sar_same_cross = self.sar_same_modal_cross_level[level](sar_feats, target_size)
            rgb_cross_cross = self.rgb_from_sar_cross_level[level](sar_feats, target_size)
            sar_cross_cross = self.sar_from_rgb_cross_level[level](rgb_feats, target_size)
            next_rgb.append(self.rgb_aggregators[level](
                rgb_feat, self.rgb_same_level[level](rgb_feat), rgb_same_cross,
                self.rgb_from_sar_same_level[level](sar_feat), rgb_cross_cross))
            next_sar.append(self.sar_aggregators[level](
                sar_feat, self.sar_same_level[level](sar_feat), sar_same_cross,
                self.sar_from_rgb_same_level[level](rgb_feat), sar_cross_cross))
        return next_rgb, next_sar


# --------------------------------------------------------------------------- #
# drop-in replacement for FpnMoE
# --------------------------------------------------------------------------- #
class MoARFpnFuse(nn.Module):
    """CMNeXtSGSLN-style fusion as a drop-in for FpnMoE.

    forward(sar_list, rgb_list) -> (fused_list, None, None)
    matching FpnMoE's 3-tuple contract.  DMFIA refines copies of the pyramids
    used only here (the per-modality EDL path is untouched), then per-level
    MoAR rectify+fuse produces the fused pyramid (channels == inchan_list).
    """

    def __init__(self, inchan_list, num_dmfia_units: int = 1,
                 moa_num_experts: int = 3, moa_top_k: int = 1,
                 gate_mode: str = "renorm", null_expert: bool = False) -> None:
        super().__init__()
        channels = tuple(inchan_list)
        self.dmfia_units = nn.ModuleList(DMFIAUnit(channels) for _ in range(num_dmfia_units))
        self.fusion_blocks = nn.ModuleList(
            CrossModalFusionBlock(ch, moa_top_k=moa_top_k, moa_num_experts=moa_num_experts,
                                  gate_mode=gate_mode, null_expert=null_expert)
            for ch in channels
        )

    def forward(self, sar_list, rgb_list):
        # CoDER calls MoE_Fuse(sar_content, rgb_content); CMNeXtSGSLN convention is (rgb, sar).
        rgb = list(rgb_list)
        sar = list(sar_list)
        for unit in self.dmfia_units:
            rgb, sar = unit(rgb, sar)
        fused = [blk(r, s) for blk, r, s in zip(self.fusion_blocks, rgb, sar)]
        return fused, None, None


# --------------------------------------------------------------------------- #
# SegFormer-style all-level decode head (ported from CMNeXtSGSLN, the clean-SOTA
# baseline) — the decode head that reaches 0.5779 on WHU, as a drop-in for
# CoDER's FPN-based `decoder_fused`.
# --------------------------------------------------------------------------- #
class MultiLevelDecoder(nn.Module):
    """Project every level to a common width, upsample to `fuse_level`'s
    resolution, concat, fuse, classify, then resize to `output_size`."""

    def __init__(self, in_channels, decoder_channels: int, num_classes: int,
                 fuse_level: int = 0, dropout: float = 0.0) -> None:
        super().__init__()
        self.fuse_level = fuse_level
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.projectors = nn.ModuleList(PointwiseNormAct(ch, decoder_channels) for ch in in_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_channels * len(in_channels), decoder_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

    def forward(self, feats, output_size):
        target_size = feats[self.fuse_level].shape[2:]
        projected = []
        for projector, feat in zip(self.projectors, feats):
            context = projector(feat)
            if context.shape[2:] != target_size:
                context = F.interpolate(context, size=target_size, mode="bilinear", align_corners=False)
            projected.append(context)
        decoded = self.fuse(torch.cat(projected, dim=1))
        logits = self.classifier(self.dropout(decoded))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class MultiLevelFusedDecoder(nn.Module):
    """Drop-in for CoDER's `decoder_fused`: callable on the fused pyramid list,
    returns full-resolution logits (fuse at `fuse_level`, upsample x`scale_factor`
    to match the original FPN+AssymetricDecoder stride-4 -> full-res behaviour),
    so no call site (`_logits_and_evidence_fused`) needs to change."""

    def __init__(self, in_channels, decoder_channels: int, num_classes: int,
                 fuse_level: int = 0, dropout: float = 0.0, scale_factor: int = 4) -> None:
        super().__init__()
        self.core = MultiLevelDecoder(tuple(in_channels), decoder_channels, num_classes, fuse_level, dropout)
        self.scale_factor = scale_factor

    def forward(self, feats):
        h, w = feats[self.core.fuse_level].shape[2:]
        return self.core(feats, output_size=(h * self.scale_factor, w * self.scale_factor))
