"""Label-free input statistics for fallback and masked fusion (ReBound-v4).

All statistics are z-scored against clean training images, so they are
invariant to the per-channel affine input normalisation: they can be computed
on normalised inputs with the same result as on raw sensor values.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_LAPLACE = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)


def image_stats(x: torch.Tensor) -> torch.Tensor:
    """[B,C,H,W] -> [B,3C]: per-channel mean, std and mean |Laplacian|."""
    b, c = x.shape[:2]
    lap = F.conv2d(x.reshape(b * c, 1, *x.shape[2:]).float(), _LAPLACE.to(x.device), padding=1)
    flat = x.reshape(b, c, -1).float()
    return torch.cat([flat.mean(-1), flat.std(-1), lap.abs().reshape(b, c, -1).mean(-1)], 1)


def local_stats(x: torch.Tensor, window: int = 15) -> torch.Tensor:
    """[B,C,H,W] -> [B,2C,H,W]: local mean and std in a window x window box."""
    x = x.float()
    pad = window // 2
    mean = F.avg_pool2d(x, window, 1, pad, count_include_pad=False)
    sq = F.avg_pool2d(x * x, window, 1, pad, count_include_pad=False)
    return torch.cat([mean, (sq - mean * mean).clamp_min(0).sqrt()], 1)


def max_abs_z(stats: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    """Largest |z| over the statistic dimension (dim 1)."""
    shape = (1, -1) + (1,) * (stats.dim() - 2)
    return ((stats - mu.view(shape)) / sd.view(shape)).abs().max(1).values


def dilate(mask: torch.Tensor, kernel: int) -> torch.Tensor:
    """Binary dilation of a [B,1,H,W] mask by a square kernel."""
    if kernel <= 1:
        return mask
    return F.max_pool2d(mask, kernel, 1, kernel // 2)


def blob_masks(batch: int, height: int, width: int, device, generator=None,
               coverage=(0.1, 0.9)) -> torch.Tensor:
    """Random cloud-like occlusion masks [B,1,H,W] in [0,1] (training).

    A smooth random field at a random scale (1/16 .. 1/64 of the image) is
    thresholded at the quantile giving a coverage drawn uniformly from
    ``coverage``; edges are softened over ~10% of the field's spread.
    """
    masks = []
    for _ in range(batch):
        scale = 2 ** torch.randint(4, 7, (1,), generator=generator).item()
        low = torch.randn((1, 1, max(height // scale, 2), max(width // scale, 2)),
                          generator=generator)
        field = F.interpolate(low.to(device), size=(height, width), mode="bicubic",
                              align_corners=False)
        cover = coverage[0] + (coverage[1] - coverage[0]) * torch.rand((1,), generator=generator).item()
        thr = torch.quantile(field.flatten(), 1.0 - cover)
        soft = 0.1 * field.std()
        masks.append(((field - thr) / soft + 0.5).clamp(0, 1))
    return torch.cat(masks, 0)


def pool_to(mask: torch.Tensor, size) -> torch.Tensor:
    """Fraction of masked area in each cell of a feature map of spatial ``size``."""
    return F.adaptive_avg_pool2d(mask, size)


__all__ = ["image_stats", "local_stats", "max_abs_z", "dilate", "blob_masks", "pool_to"]
