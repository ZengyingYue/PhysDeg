"""Latent conditional DDPM restoration — the generative-imputer comparison point
for the Stage-2 method-breadth benchmark.

A sibling of `recovery_flow.LatentRectifiedFlow`: restores the missing modality's
content-feature pyramid from the available one, but with a **denoising-diffusion**
objective (ε-prediction, cosine β schedule) and DDIM few-step sampling, instead of
rectified-flow velocity matching.  Reuses the *same* per-level conditional network
(`recovery_flow._LevelVelocity`) and cond-standardised working space, so the only
difference vs flow is the generative process — the fair "swap only the recovery"
comparison.  `restore()` is drop-in with the flow's signature
(`z_hat_list, sigma2, sigma2_map`).

Trained standalone on FROZEN CoDER features (restore rgb_content from sar_content)
so it plugs into the identical encoder/decoder/Dempster eval machinery as the
trivial fills and the flow — see `tests/train_ddpm_recovery.py`.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from module.baseline.recovery_flow import _LevelVelocity, _sinusoidal_time_embedding


def _cosine_alpha_bar(T: int, s: float = 0.008) -> torch.Tensor:
    """Nichol-Dhariwal cosine schedule: returns ᾱ_t for t=0..T (length T+1)."""
    steps = torch.arange(T + 1, dtype=torch.float32)
    f = torch.cos(((steps / T) + s) / (1 + s) * (torch.pi / 2)) ** 2
    return (f / f[0]).clamp(1e-6, 1.0)


class LatentDDPM(nn.Module):
    """Conditional latent DDPM over a content-feature pyramid (ε-prediction)."""

    def __init__(
        self,
        channels: Sequence[int],
        t_dim: int = 64,
        T: int = 1000,
        n_steps: int = 50,        # DDIM sampling steps
        n_samples: int = 4,       # seeds for restoration σ²
        sigma_detach: bool = True,
        x0_clip: float = 3.0,     # static thresholding (standardised space) — stops DDIM x0 blow-up
    ):
        super().__init__()
        self.channels = list(channels)
        self.t_dim = int(t_dim)
        self.T = int(T)
        self.n_steps = int(n_steps)
        self.n_samples = int(n_samples)
        self.sigma_detach = bool(sigma_detach)
        self.x0_clip = float(x0_clip)
        # reuse the flow's per-level conditional net as the ε-predictor
        # (no zero-init: DDPM ε-prediction has no identity-start rationale and
        # converges much faster with standard init)
        self.levels = nn.ModuleList(
            [_LevelVelocity(c, self.t_dim, zero_init=False) for c in self.channels])
        self.register_buffer("alpha_bar", _cosine_alpha_bar(self.T))   # [T+1]

    @staticmethod
    def _stats(x: torch.Tensor, eps: float = 1e-5):
        mu = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
        return mu.detach(), std.detach()

    def _eps(self, z_list, t_norm, condn_list):
        """Per-level ε prediction. ``t_norm`` is ``[B]`` in [0,1]."""
        t_emb = _sinusoidal_time_embedding(t_norm, self.t_dim)
        return [lv(z, c, t_emb) for lv, z, c in zip(self.levels, z_list, condn_list)]

    # ---- training objective ----------------------------------------------
    def ddpm_loss(self, target_list: Sequence[torch.Tensor],
                  cond_list: Sequence[torch.Tensor]) -> torch.Tensor:
        """ε-prediction MSE in cond-standardised space (matches flow's space)."""
        assert len(target_list) == len(self.channels), "level count mismatch"
        b = target_list[0].shape[0]
        device = target_list[0].device
        ti = torch.randint(1, self.T + 1, (b,), device=device)        # 1..T
        ab = self.alpha_bar[ti][:, None, None, None]                   # ᾱ_t
        t_norm = ti.float() / self.T
        z_t_list, eps_list, condn_list = [], [], []
        for z1, cond in zip(target_list, cond_list):
            mu_c, std_c = self._stats(cond)
            z0 = (z1 - mu_c) / std_c                                   # standardised target = x0
            condn = (cond - mu_c) / std_c
            eps = torch.randn_like(z0)
            z_t_list.append(ab.sqrt() * z0 + (1 - ab).sqrt() * eps)
            eps_list.append(eps)
            condn_list.append(condn)
        eps_pred = self._eps(z_t_list, t_norm, condn_list)
        loss = sum(F.mse_loss(ep, et) for ep, et in zip(eps_pred, eps_list))
        return loss / max(len(self.channels), 1)

    # ---- DDIM sampling + uncertainty -------------------------------------
    def _ddim_sample(self, condn_list, device, b):
        """One DDIM trajectory (deterministic given the initial noise) → x0 per level."""
        ts = torch.linspace(self.T, 1, self.n_steps, device=device).round().long()
        z_list = [torch.randn(b, c, *cn.shape[-2:], device=device)
                  for c, cn in zip(self.channels, condn_list)]
        for i in range(self.n_steps):
            ti = ts[i]
            ab_t = self.alpha_bar[ti]
            t_norm = torch.full((b,), float(ti) / self.T, device=device)
            eps_list = self._eps(z_list, t_norm, condn_list)
            ab_prev = self.alpha_bar[ts[i + 1]] if i + 1 < self.n_steps else self.alpha_bar[0]
            new_z = []
            for z, eps in zip(z_list, eps_list):
                x0 = (z - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
                x0 = x0.clamp(-self.x0_clip, self.x0_clip)   # static thresholding: stop x0 blow-up
                new_z.append(ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps)
            z_list = new_z
        return z_list   # ≈ x0 (standardised)

    @torch.no_grad()
    def restore(self, cond_list: Sequence[torch.Tensor]
                ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """K-seed DDIM restoration → (z_hat raw-scale pyramid, sigma2, sigma2_map),
        signature-compatible with LatentRectifiedFlow.restore."""
        device = cond_list[0].device
        b = cond_list[0].shape[0]
        stats = [self._stats(c) for c in cond_list]
        condn_list = [(c - mu) / std for c, (mu, std) in zip(cond_list, stats)]
        per_level = [[] for _ in self.channels]
        for _k in range(self.n_samples):
            x0_list = self._ddim_sample(condn_list, device, b)
            for li, z in enumerate(x0_list):
                per_level[li].append(z)
        z_hat_list, var_terms, map_terms = [], [], []
        for li, (mu, std) in enumerate(stats):
            stack = torch.stack(per_level[li], 0)                     # [K,B,C,H,W]
            mean = stack.mean(0)
            var = stack.var(0, unbiased=False)                        # standardised space
            z_hat_list.append(mean * std + mu)                       # de-standardise
            var_terms.append(var.mean(dim=(1, 2, 3), keepdim=True))   # [B,1,1,1]
            vm = var.mean(1, keepdim=True)                            # [B,1,H,W]
            map_terms.append(F.interpolate(vm, size=per_level[0][0].shape[-2:],
                                           mode="bilinear", align_corners=False))
        sigma2 = torch.stack(var_terms, 0).mean(0)
        sigma2_map = torch.stack(map_terms, 0).mean(0)
        if self.sigma_detach:
            sigma2 = sigma2.detach()
        return z_hat_list, sigma2, sigma2_map
