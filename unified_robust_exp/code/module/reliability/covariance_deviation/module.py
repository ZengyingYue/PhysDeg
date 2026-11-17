"""Input-adaptive observed-modality reliability from covariance deviation.

The module is deliberately separate from MetaRS's ``CovMatrix_mmr``.  MMR owns
a training-time whitening mask and clears its temporary variance accumulator;
this module owns persistent, per-modality clean-domain reference statistics.
All tensors below are registered buffers and therefore survive checkpoints.

Calibration is two-pass:

1. Welford accumulation estimates clean per-entry covariance mean/variance.
2. Clean deviation scores set a high clean threshold (q90) and scale (q99-q90).

Inference computes covariance independently for every image over spatial
positions, so beta never depends on the other samples in a batch.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


def _feature_covariance(feature: torch.Tensor, eps: float) -> torch.Tensor:
    """Return per-image channel covariance, shape [B,C,C]."""
    if feature.ndim != 4:
        raise ValueError(f"expected BCHW feature, got {tuple(feature.shape)}")
    b, c, h, w = feature.shape
    flat = feature.float().reshape(b, c, h * w)
    flat = flat - flat.mean(dim=2, keepdim=True)
    denom = max(h * w - 1, 1)
    cov = torch.bmm(flat, flat.transpose(1, 2)) / float(denom)
    eye = torch.eye(c, device=cov.device, dtype=cov.dtype).unsqueeze(0)
    return cov + eps * eye


class _ReferenceLevel(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        shape = (channels, channels)
        for modality in ("sar", "rgb"):
            self.register_buffer(f"{modality}_mean", torch.zeros(shape))
            self.register_buffer(f"{modality}_m2", torch.zeros(shape))
            self.register_buffer(f"{modality}_var", torch.ones(shape))
            self.register_buffer(f"{modality}_mask", torch.zeros(shape, dtype=torch.bool))
            self.register_buffer(f"{modality}_count", torch.zeros((), dtype=torch.long))
            self.register_buffer(f"{modality}_threshold", torch.zeros(()))
            self.register_buffer(f"{modality}_scale", torch.ones(()))
            self.register_buffer(f"{modality}_ready", torch.zeros((), dtype=torch.bool))
            # Diagonal (per-channel variance) deviation calibration.  Independent
            # sensor noise shows up in the diagonal while the masked off-diagonal
            # score can move toward the clean reference.
            self.register_buffer(f"{modality}_diag_threshold", torch.zeros(()))
            self.register_buffer(f"{modality}_diag_scale", torch.ones(()))
            self.register_buffer(f"{modality}_diag_ready", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def reset(self, modality: str) -> None:
        getattr(self, f"{modality}_mean").zero_()
        getattr(self, f"{modality}_m2").zero_()
        getattr(self, f"{modality}_var").fill_(1.0)
        getattr(self, f"{modality}_mask").zero_()
        getattr(self, f"{modality}_count").zero_()
        getattr(self, f"{modality}_threshold").zero_()
        getattr(self, f"{modality}_scale").fill_(1.0)
        getattr(self, f"{modality}_ready").zero_()
        getattr(self, f"{modality}_diag_threshold").zero_()
        getattr(self, f"{modality}_diag_scale").fill_(1.0)
        getattr(self, f"{modality}_diag_ready").zero_()

    @torch.no_grad()
    def update(self, cov: torch.Tensor, modality: str) -> None:
        """Parallel Welford update over a [B,C,C] covariance batch."""
        cov = cov.detach()
        batch_n = cov.shape[0]
        if batch_n == 0:
            return
        mean = getattr(self, f"{modality}_mean")
        m2 = getattr(self, f"{modality}_m2")
        count = getattr(self, f"{modality}_count")
        batch_mean = cov.mean(0)
        batch_m2 = ((cov - batch_mean) ** 2).sum(0)
        old_n = int(count.item())
        new_n = old_n + batch_n
        if old_n == 0:
            mean.copy_(batch_mean)
            m2.copy_(batch_m2)
        else:
            delta = batch_mean - mean
            mean.add_(delta * (batch_n / float(new_n)))
            m2.add_(batch_m2 + delta.square() * (old_n * batch_n / float(new_n)))
        count.fill_(new_n)

    @torch.no_grad()
    def finalize(self, modality: str, mask_fraction: float, eps: float) -> None:
        count = int(getattr(self, f"{modality}_count").item())
        if count < 2:
            raise RuntimeError(f"need >=2 clean samples for {modality}, got {count}")
        m2 = getattr(self, f"{modality}_m2")
        var = getattr(self, f"{modality}_var")
        var.copy_(m2 / float(count - 1))

        channels = var.shape[0]
        upper = torch.triu(torch.ones_like(var, dtype=torch.bool), diagonal=1)
        candidates = var[upper]
        keep = max(1, int(round(candidates.numel() * mask_fraction)))
        threshold = torch.topk(candidates, k=keep, largest=True).values.min()
        mask = upper & (var >= threshold)
        mask = mask | mask.transpose(0, 1)
        # Never include diagonal variance in the quality score.
        getattr(self, f"{modality}_mask").copy_(mask)
        getattr(self, f"{modality}_var").clamp_min_(eps)

    def scores(self, cov: torch.Tensor, modality: str, eps: float) -> torch.Tensor:
        mean = getattr(self, f"{modality}_mean").to(cov.device)
        var = getattr(self, f"{modality}_var").to(cov.device)
        mask = getattr(self, f"{modality}_mask").to(cov.device)
        if not bool(mask.any()):
            return torch.zeros(cov.shape[0], device=cov.device)
        # A relative floor prevents near-constant clean entries from dominating.
        positive = var[mask]
        variance_floor = positive.median().clamp_min(eps) * 1e-2
        std = var.clamp_min(variance_floor).sqrt()
        z = (cov - mean).abs() / (std + eps)
        return z[:, mask].mean(dim=1)

    def diag_scores(self, cov: torch.Tensor, modality: str, eps: float) -> torch.Tensor:
        """Mean |z| of the per-channel variances against the clean reference."""
        mean = torch.diagonal(getattr(self, f"{modality}_mean").to(cov.device))
        var = torch.diagonal(getattr(self, f"{modality}_var").to(cov.device))
        positive = var[var > 0]
        floor = (positive.median().clamp_min(eps) * 1e-2 if positive.numel()
                 else torch.tensor(eps, device=cov.device))
        diagonal = torch.diagonal(cov, dim1=1, dim2=2)
        return ((diagonal - mean).abs() / (var.clamp_min(floor).sqrt() + eps)).mean(dim=1)


class CovarianceDeviationReliability(nn.Module):
    """Persistent, modality-specific, per-image covariance quality gate."""

    def __init__(
        self,
        channels: Sequence[int],
        beta_floor: float = 0.1,
        mask_fraction: float = 0.1,
        clean_quantile: float = 0.90,
        scale_quantile: float = 0.99,
        tau: float = 1.0,
        eps: float = 1e-6,
        diag_weight: float = 0.0,
    ):
        super().__init__()
        if not channels:
            raise ValueError("at least one covariance-tracked level is required")
        if not 0.0 < beta_floor < 1.0:
            raise ValueError("beta_floor must be in (0,1)")
        if not 0.0 < mask_fraction <= 1.0:
            raise ValueError("mask_fraction must be in (0,1]")
        if not 0.0 < clean_quantile < scale_quantile < 1.0:
            raise ValueError("require 0 < clean_quantile < scale_quantile < 1")
        self.levels = nn.ModuleList([_ReferenceLevel(int(c)) for c in channels])
        self.beta_floor = float(beta_floor)
        self.mask_fraction = float(mask_fraction)
        self.clean_quantile = float(clean_quantile)
        self.scale_quantile = float(scale_quantile)
        self.tau = float(tau)
        self.eps = float(eps)
        self.diag_weight = float(diag_weight)

    def _validate(self, features: Sequence[torch.Tensor], modality: str) -> None:
        if modality not in ("sar", "rgb"):
            raise ValueError(f"unknown modality {modality!r}")
        if len(features) != len(self.levels):
            raise ValueError(
                f"expected {len(self.levels)} feature levels, got {len(features)}"
            )

    @torch.no_grad()
    def reset_calibration(self) -> None:
        for level in self.levels:
            for modality in ("sar", "rgb"):
                level.reset(modality)

    @torch.no_grad()
    def update(self, features: Sequence[torch.Tensor], modality: str) -> None:
        self._validate(features, modality)
        for feature, level in zip(features, self.levels):
            level.update(_feature_covariance(feature, self.eps), modality)

    @torch.no_grad()
    def finalize_statistics(self) -> None:
        for level in self.levels:
            for modality in ("sar", "rgb"):
                level.finalize(modality, self.mask_fraction, self.eps)

    @torch.no_grad()
    def deviation_scores(
        self, features: Sequence[torch.Tensor], modality: str
    ) -> list[torch.Tensor]:
        self._validate(features, modality)
        return [
            level.scores(_feature_covariance(feature, self.eps), modality, self.eps)
            for feature, level in zip(features, self.levels)
        ]

    @torch.no_grad()
    def set_score_calibration(
        self, scores: Sequence[torch.Tensor], modality: str
    ) -> None:
        if len(scores) != len(self.levels):
            raise ValueError("one score tensor is required per feature level")
        for score, level in zip(scores, self.levels):
            score = score.detach().float().flatten()
            if score.numel() < 2:
                raise RuntimeError("need >=2 clean deviation scores")
            q_clean = torch.quantile(score, self.clean_quantile)
            q_scale = torch.quantile(score, self.scale_quantile)
            scale = (q_scale - q_clean).clamp_min(self.eps)
            getattr(level, f"{modality}_threshold").copy_(q_clean)
            getattr(level, f"{modality}_scale").copy_(scale)
            getattr(level, f"{modality}_ready").fill_(True)

    @torch.no_grad()
    def diagonal_scores(
        self, features: Sequence[torch.Tensor], modality: str
    ) -> list[torch.Tensor]:
        self._validate(features, modality)
        return [
            level.diag_scores(_feature_covariance(feature, self.eps), modality, self.eps)
            for feature, level in zip(features, self.levels)
        ]

    @torch.no_grad()
    def set_diag_calibration(
        self, scores: Sequence[torch.Tensor], modality: str
    ) -> None:
        if len(scores) != len(self.levels):
            raise ValueError("one score tensor is required per feature level")
        for score, level in zip(scores, self.levels):
            score = score.detach().float().flatten()
            q_clean = torch.quantile(score, self.clean_quantile)
            q_scale = torch.quantile(score, self.scale_quantile)
            getattr(level, f"{modality}_diag_threshold").copy_(q_clean)
            getattr(level, f"{modality}_diag_scale").copy_((q_scale - q_clean).clamp_min(self.eps))
            getattr(level, f"{modality}_diag_ready").fill_(True)

    def is_ready(self, modality: str | None = None) -> bool:
        modalities = (modality,) if modality else ("sar", "rgb")
        return all(
            bool(getattr(level, f"{name}_ready"))
            for level in self.levels
            for name in modalities
        )

    def forward(
        self, features: Sequence[torch.Tensor], modality: str
    ) -> torch.Tensor:
        self._validate(features, modality)
        batch = features[0].shape[0]
        device = features[0].device
        if not self.is_ready(modality):
            return torch.ones(batch, 1, 1, 1, device=device)
        covs = [_feature_covariance(feature, self.eps) for feature in features]
        level_betas = []
        for cov, level in zip(covs, self.levels):
            score = level.scores(cov, modality, self.eps)
            threshold = getattr(level, f"{modality}_threshold").to(score.device)
            scale = getattr(level, f"{modality}_scale").to(score.device)
            excess = torch.relu(score - threshold) / (scale * self.tau + self.eps)
            if self.diag_weight > 0 and bool(getattr(level, f"{modality}_diag_ready")):
                diag = level.diag_scores(cov, modality, self.eps)
                d_thr = getattr(level, f"{modality}_diag_threshold").to(diag.device)
                d_scale = getattr(level, f"{modality}_diag_scale").to(diag.device)
                excess = excess + self.diag_weight * torch.relu(diag - d_thr) / (d_scale + self.eps)
            beta = self.beta_floor + (1.0 - self.beta_floor) * torch.exp(-excess)
            level_betas.append(beta.clamp(self.beta_floor, 1.0))
        stacked = torch.stack(level_betas, dim=1)
        beta = torch.exp(torch.log(stacked.clamp_min(self.eps)).mean(dim=1))
        return beta[:, None, None, None]

