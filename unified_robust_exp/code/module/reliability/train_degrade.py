"""Random training-time degradations in raw sensor units (DPD screening).

Seen types follow the parameter ranges of the evaluation corruptions in
``experiments/revision/evaluate.py`` (haze, noise, blur) plus the blob clouds
of ``moddrop_cloudaug``; dark, stripe and localnoise are never applied here,
so they stay held out at evaluation.  Inputs and outputs are normalised
tensors; ``mean``/``std`` are the per-band normalisation statistics.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from module.reliability import pixel_stats as pxs

OPTICAL_TYPES = ("haze", "noise", "blur", "cloud")


def _blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(math.ceil(3 * sigma)))
    axis = torch.arange(-radius, radius + 1, dtype=img.dtype, device=img.device)
    kernel = torch.exp(-axis ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    c = img.shape[1]
    k2 = (kernel[:, None] * kernel[None, :]).expand(c, 1, -1, -1)
    return F.conv2d(F.pad(img, [radius] * 4, mode="reflect"), k2, groups=c)


def degrade_optical(rgb, mean, std, prob, types=OPTICAL_TYPES, max_dn=255.0, unit=1.0):
    """Degrade each sample's optical image with probability ``prob``.

    DN constants are for 8-bit imagery; ``unit`` rescales them (SEN12MS-CR reflectance).
    """
    chosen = torch.rand(rgb.shape[0], device=rgb.device) < prob
    if not bool(chosen.any()):
        return rgb
    mean, std = mean.view(1, -1, 1, 1), std.view(1, -1, 1, 1)
    raw = rgb.float() * std + mean
    out = []
    for i in range(rgb.shape[0]):
        img = raw[i:i + 1]
        if chosen[i]:
            kind = types[int(torch.randint(len(types), (1,)))]
            u = float(torch.rand(1))
            if kind == "haze":
                w = 0.10 + 0.60 * u
                airlight = unit * (200.0 + 55.0 * float(torch.rand(1)))
                img = img * (1 - w) + airlight * w
            elif kind == "noise":
                img = img + torch.randn_like(img) * unit * (5.0 + 45.0 * u)
            elif kind == "blur":
                img = _blur(img, 0.5 + 3.5 * u)
            elif kind == "cloud":
                mask = pxs.blob_masks(1, img.shape[2], img.shape[3], img.device)
                opacity = 0.6 + 0.4 * u
                value = unit * (230.0 + 25.0 * float(torch.rand(1)))
                img = img * (1 - mask * opacity) + value * mask * opacity
            img = img.clamp(0, max_dn)
        out.append(img)
    return ((torch.cat(out, 0) - mean) / std).to(rgb.dtype)


def degrade_sar(sar, mean, std, prob, max_dn=255.0, domain="amp"):
    """Gamma speckle (1..16 looks) with probability ``prob``; additive in dB for ``domain='db'``."""
    chosen = torch.rand(sar.shape[0], device=sar.device) < prob
    if not bool(chosen.any()):
        return sar
    mean, std = mean.view(1, -1, 1, 1), std.view(1, -1, 1, 1)
    raw = sar.float() * std + mean
    looks = 1.0 + 15.0 * torch.rand(sar.shape[0], 1, 1, 1, device=sar.device)
    speckle = torch._standard_gamma(looks.expand_as(raw).contiguous()) / looks
    if domain == "db":
        noisy = raw + 10 * torch.log10(speckle.clamp_min(1e-6))
    else:
        noisy = (raw * speckle).clamp(0, max_dn)
    raw = torch.where(chosen.view(-1, 1, 1, 1), noisy, raw)
    return ((raw - mean) / std).to(sar.dtype)


__all__ = ["OPTICAL_TYPES", "degrade_optical", "degrade_sar"]
