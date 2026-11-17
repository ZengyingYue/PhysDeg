"""CoDER v3 — Mixture-of-Experts cross-modal fusion.

Stage S1 (this file): `BasicMoE` per FPN level.  Four experts:
    E1: private SAR expert   (sar_feat            → out_channels)
    E2: private RGB expert   (rgb_feat            → out_channels)
    E3: collaborative shallow (concat[sar, rgb]   → out_channels)
    E4: collaborative deep    (concat[sar, rgb]   → out_channels, larger hidden)

A single global router (1x1 GAP → MLP → 4-way softmax) produces per-sample
expert weights.  The fused output is the weighted sum of expert outputs.
A balance loss (negative entropy of mean expert usage) is returned alongside
the fused feature so the training loop can prevent expert collapse.

Stage S2 will extend `BasicMoE` to consume a reliability vector (β-driven
routing).  Stage S3 will add modality-bridge experts.  Stage S4 will add a
modality-aware BiFPN feeding into this module.  See `V3_PLAN.md`.

The pyramidal wrapper `FpnMoE` matches the `Fusespfnv2(sar_content,
rgb_content)` call signature so it can be a drop-in replacement in the
CoDER (and MetaRS) forward.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU1x1(nn.Module):
    """Lightweight 1x1 conv + BN + ReLU.  Used inside ExpertBlock."""

    def __init__(self, in_channels: int, out_channels: int, use_bn: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ExpertBlock(nn.Module):
    """1x1 → BN → ReLU → 1x1 expert.  Channels: in → hidden → out."""

    def __init__(self, in_channels: int, out_channels: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden if hidden is not None else in_channels
        self.net = nn.Sequential(
            ConvBNReLU1x1(in_channels, hidden),
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.net(x)


class BasicMoE(nn.Module):
    """4-expert MoE at a single FPN level.

    Returns:
        fused: [B, C_out, H, W] weighted sum of expert outputs
        router_weights: [B, 4] softmax over experts
        balance_loss: scalar tensor — negative entropy of mean expert usage
                      (subtract this from total loss to encourage uniform usage)
    """

    NUM_EXPERTS = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        router_hidden: int = 128,
        router_dropout: float = 0.1,
    ):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        # 2 private experts
        self.private_sar = ExpertBlock(in_channels, out_channels)
        self.private_rgb = ExpertBlock(in_channels, out_channels)
        # 2 collaborative experts on concat[sar, rgb] -- different capacities
        self.collab_shallow = ExpertBlock(
            2 * in_channels, out_channels, hidden=in_channels
        )
        self.collab_deep = ExpertBlock(
            2 * in_channels, out_channels, hidden=2 * in_channels
        )
        # Router: input = global avg-pooled concat[sar, rgb]
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2 * in_channels, router_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(router_dropout),
            nn.Linear(router_hidden, self.NUM_EXPERTS),
        )

    def forward(self, sar_feat: torch.Tensor, rgb_feat: torch.Tensor):
        B, C, H, W = sar_feat.shape
        joint = torch.cat([sar_feat, rgb_feat], dim=1)  # [B, 2C, H, W]

        e_sar = self.private_sar(sar_feat)
        e_rgb = self.private_rgb(rgb_feat)
        e_collab_s = self.collab_shallow(joint)
        e_collab_d = self.collab_deep(joint)
        experts = torch.stack(
            [e_sar, e_rgb, e_collab_s, e_collab_d], dim=1
        )  # [B, 4, C_out, H, W]

        router_logits = self.router(joint)  # [B, 4]
        router_weights = F.softmax(router_logits, dim=1)  # [B, 4]
        w = router_weights.view(B, self.NUM_EXPERTS, 1, 1, 1)
        fused = (experts * w).sum(dim=1)  # [B, C_out, H, W]

        # Balance loss: maximise entropy of *batch-mean* usage, so the
        # router uses all experts on average.  Returns a value in [-log(K), 0];
        # add `lam_balance * balance_loss` to total loss to push it toward
        # -log(K) (uniform usage), i.e. minimise this scalar.
        usage = router_weights.mean(dim=0).clamp_min(1e-8)
        entropy = -(usage * usage.log()).sum()
        balance_loss = -entropy  # minimising balance_loss = maximising entropy

        return fused, router_weights, balance_loss


class FpnMoE(nn.Module):
    """Per-level wrapper: one BasicMoE per FPN scale.

    Matches the call signature of MetaRS's `Fusespfnv2.forward(sar_list,
    rgb_list) -> fused_list` so it can be used as a drop-in replacement.
    Additionally returns the stacked router-weights tensor and the mean
    balance loss across levels, which the training loop can use for
    regularisation and visualisation.
    """

    def __init__(self, inchan_list, out_chan_list=None, **moe_kwargs):
        super().__init__()
        if out_chan_list is None:
            out_chan_list = list(inchan_list)
        assert len(inchan_list) == len(out_chan_list)
        self.modules_per_level = nn.ModuleList(
            [
                BasicMoE(in_c, out_c, **moe_kwargs)
                for in_c, out_c in zip(inchan_list, out_chan_list)
            ]
        )

    def forward(self, sar_list, rgb_list):
        assert len(sar_list) == len(rgb_list) == len(self.modules_per_level), (
            f"FpnMoE expected {len(self.modules_per_level)} levels, got "
            f"sar={len(sar_list)} rgb={len(rgb_list)}"
        )
        fused_list = []
        router_weights_per_level = []
        total_balance = 0.0
        for moe, sar_f, rgb_f in zip(self.modules_per_level, sar_list, rgb_list):
            fused, w, bl = moe(sar_f, rgb_f)
            fused_list.append(fused)
            router_weights_per_level.append(w)
            total_balance = total_balance + bl
        mean_balance = total_balance / float(len(self.modules_per_level))
        return fused_list, router_weights_per_level, mean_balance
