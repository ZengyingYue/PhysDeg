"""Complete-recovery baselines for the ReBound central comparison.

These modules deliberately reconstruct *all* missing-modality content instead
of limiting recovery to a shared subspace:

* ``FullFeaturePyramidRecoverer`` predicts both the content and the
  modality-specific feature pyramid of the missing modality.
* ``FullImageRecoverer`` predicts the missing raw (normalised) sensor image,
  which is then passed through the ordinary missing-modality encoders.

Neither module produces or consumes a reliability discount.  Their recovered
outputs are treated exactly like observed inputs by the downstream fusion head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FeatureLevelMapper(nn.Module):
    """Memory-conscious residual mapper for one feature-pyramid level."""

    def __init__(self, channels: int, bottleneck_ratio: int = 8):
        super().__init__()
        hidden = max(32, channels // max(int(bottleneck_ratio), 1))
        groups = min(32, hidden)
        while hidden % groups:
            groups -= 1
        self.mapper = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 2 * channels, kernel_size=1),
        )
        # Start from the defensible copy baseline, then learn cross-modal
        # residuals.  Zero initialisation also keeps the first optimisation
        # steps numerically stable at the 2048-channel pyramid level.
        nn.init.zeros_(self.mapper[-1].weight)
        nn.init.zeros_(self.mapper[-1].bias)

    def forward(self, content: torch.Tensor, specific: torch.Tensor):
        delta_content, delta_specific = self.mapper(
            torch.cat([content, specific], dim=1)
        ).chunk(2, dim=1)
        return content + delta_content, specific + delta_specific


class FullFeaturePyramidRecoverer(nn.Module):
    """Predict complete missing-modality content and specific pyramids."""

    def __init__(self, channels, bottleneck_ratio: int = 8):
        super().__init__()
        self.levels = nn.ModuleList(
            [_FeatureLevelMapper(int(c), bottleneck_ratio) for c in channels]
        )

    def forward(self, content, specific):
        if len(content) != len(self.levels) or len(specific) != len(self.levels):
            raise ValueError(
                "feature pyramid length does not match complete recoverer"
            )
        recovered_content, recovered_specific = [], []
        for mapper, content_level, specific_level in zip(
            self.levels, content, specific
        ):
            rec_content, rec_specific = mapper(content_level, specific_level)
            recovered_content.append(rec_content)
            recovered_specific.append(rec_specific)
        return recovered_content, recovered_specific


class _ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        groups = min(16, out_channels)
        while out_channels % groups:
            groups -= 1
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )


class FullImageRecoverer(nn.Module):
    """Small U-Net that reconstructs the complete normalised sensor image."""

    def __init__(self, in_channels: int, out_channels: int, width: int = 32):
        super().__init__()
        width = int(width)
        self.enc1 = nn.Sequential(
            _ConvNormAct(in_channels, width),
            _ConvNormAct(width, width),
        )
        self.enc2 = nn.Sequential(
            _ConvNormAct(width, 2 * width, stride=2),
            _ConvNormAct(2 * width, 2 * width),
        )
        self.bottleneck = nn.Sequential(
            _ConvNormAct(2 * width, 4 * width, stride=2),
            _ConvNormAct(4 * width, 4 * width),
        )
        self.dec2 = nn.Sequential(
            _ConvNormAct(6 * width, 2 * width),
            _ConvNormAct(2 * width, 2 * width),
        )
        self.dec1 = nn.Sequential(
            _ConvNormAct(3 * width, width),
            _ConvNormAct(width, width),
        )
        self.out = nn.Conv2d(width, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        skip1 = self.enc1(x)
        skip2 = self.enc2(skip1)
        latent = self.bottleneck(skip2)
        latent = F.interpolate(
            latent, size=skip2.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded2 = self.dec2(torch.cat([latent, skip2], dim=1))
        decoded2 = F.interpolate(
            decoded2, size=skip1.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.out(self.dec1(torch.cat([decoded2, skip1], dim=1)))
