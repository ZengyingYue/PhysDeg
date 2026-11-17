"""Latent rectified-flow restoration for missing-modality recovery (CoDER M3→M5a).

The headline of the method is **not** "use a flow to recover the missing
modality"; it is **calibrated restoration uncertainty**.  This module restores
the *content latent pyramid* of a missing modality from the available one with a
conditional rectified flow (flow matching), and reports a restoration
uncertainty ``sigma2`` (per-image) obtained as the spread of the ODE endpoint
over K noise seeds.  ``sigma2`` is what downstream CoDER turns into an evidential
reliability discount::

    beta_rec = beta_in * gamma_rec * exp(-sigma2 / tau)

so that an *uncertain* restoration discounts the recovered Dirichlet evidence
before Dempster combination.

Design notes
------------
* **Rectified flow / flow matching** (straight-path ``z_t = (1-t) z0 + t z1``,
  velocity target ``z1 - z0``) was chosen over conditional DDPM / consistency
  models: simplest + stablest training, few-step ODE inference, and the cleanest
  uncertainty story (seed-ensemble variance, upgradable to the closed-form
  divergence identity later).
* **Per-level**: ``content`` is a 4-level pyramid (channels = ``fuse_inchan_list``
  = (128,256,512,1024) for the ConvNeXt backbone).  One small conv velocity head
  per level, all params created in ``__init__`` (DDP-safe, no lazy modules).
* **Uncertainty granularity (v1)**: ``sigma2`` is per-image (``[B,1,1,1]``),
  aggregated over channels/positions/levels.  Per-position / per-channel sigma2
  is a v2 extension (mirrors the existing per-position-beta cliff note).
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal embedding of a scalar time ``t`` in [0, 1].

    Args:
        t: shape ``[B]`` (one time per batch element).
        dim: embedding dimension (even).
    Returns:
        ``[B, dim]`` embedding.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:  # pad if odd
        emb = F.pad(emb, (0, 1))
    return emb


class _LevelVelocity(nn.Module):
    """Conditional velocity head for one pyramid level.

    Predicts ``v_theta(z_t, t | cond)`` ~ ``z1 - z0`` for a feature map of
    ``channels`` channels, conditioned on the available-modality feature
    ``cond`` (same shape) and a time embedding.
    """

    def __init__(self, channels: int, t_dim: int, hidden: Optional[int] = None,
                 zero_init: bool = True):
        super().__init__()
        h = hidden if hidden is not None else max(channels // 2, 32)
        self.t_proj = nn.Linear(t_dim, h)
        in_ch = 2 * channels + h  # [z_t, cond, time(broadcast)]
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, h, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, h), num_channels=h),
            nn.SiLU(),
            nn.Conv2d(h, h, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.out = nn.Conv2d(h, channels, kernel_size=1)
        # zero-init the output conv → the flow starts as the identity transport
        # (velocity 0 ⇒ z stays Gaussian); the field grows in during training,
        # analogous to the ViT-Adapter gamma-0 gate.  DDPM ε-prediction converges
        # faster WITHOUT zero-init (no identity-start rationale there).
        if zero_init:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, z_t: torch.Tensor, cond: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        b, _, hgt, wid = z_t.shape
        t_map = self.t_proj(t_emb)[:, :, None, None].expand(b, -1, hgt, wid)
        x = torch.cat([z_t, cond, t_map], dim=1)
        return self.out(self.net(x))


class _ReliabilityHead(nn.Module):
    """Tiny per-position regressor: restore-time feature stack → predicted
    restoration error r(x) ≥ 0.

    Motivation (2026-06-07 finding): the K-seed ensemble variance σ² is a *weak*
    per-position reliability signal (within-image ρ(σ²,err)≈0.19) — ensemble
    variance sees *disagreement* not *bias*.  An offline probe showed a tiny
    head on restore-time features (σ²(x), per-level mean|ẑ|, mean|cond|,
    ‖ẑ−cond‖) regressed against the true restoration error reaches ρ≈0.53 on
    held-out test cities.  The head is the primary reliability source; σ² is an
    ablation baseline.

    Output = a raw **reliability logit** h(x) (higher ⇒ more predicted error),
    trained against the *per-image-standardised* error (z-scored) — regressing to
    raw error makes a Huber head collapse to the median (inert gate); the
    z-scored target preserves rank/spread.  The gate is then
    ``β_rec = β·sigmoid(−h/τ)`` ∈ (0,1) (monotone↓ in predicted error), which
    needs no sign constraint on h.
    """

    def __init__(self, in_ch: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # raw reliability logit (no activation)


class LatentRectifiedFlow(nn.Module):
    """Conditional rectified flow over a content-feature pyramid.

    Args:
        channels: per-level channel counts of the content pyramid
            (e.g. ``fuse_inchan_list`` = (128,256,512,1024)).
        t_dim: sinusoidal time-embedding dimension.
        n_steps: Euler ODE steps for sampling (few-step).
        n_samples: noise seeds K for the restoration-uncertainty estimate.
        sigma_detach: if True, sigma2 is detached from the graph before being
            returned (so the uncertainty gate does not directly backprop into
            making the flow *more* uncertain — beta_rec uses sigma2 only as a
            stop-grad reliability signal).  The mean restoration ``z_hat`` keeps
            its graph.
    """

    def __init__(
        self,
        channels: Sequence[int],
        t_dim: int = 64,
        n_steps: int = 4,
        n_samples: int = 4,
        sigma_detach: bool = True,
        learned_reliability: bool = False,
    ):
        super().__init__()
        self.channels = list(channels)
        self.t_dim = int(t_dim)
        self.n_steps = int(n_steps)
        self.n_samples = int(n_samples)
        self.sigma_detach = bool(sigma_detach)
        self.levels = nn.ModuleList(
            [_LevelVelocity(c, self.t_dim) for c in self.channels]
        )
        # learned reliability head (primary σ² source; see _ReliabilityHead).
        # feature stack = σ²(x) map + 3 channel-summary maps per pyramid level.
        self.learned_reliability = bool(learned_reliability)
        if self.learned_reliability:
            self.reliability_head = _ReliabilityHead(1 + 3 * len(self.channels))
        else:
            self.reliability_head = None

    # ---- core field -------------------------------------------------------
    @staticmethod
    def _stats(x: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-image (over C,H,W) mean/std, detached — used to standardise the
        flow's working space so velocity targets are O(1) regardless of the
        raw feature magnitude (ConvNeXt content features are large)."""
        mu = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
        return mu.detach(), std.detach()

    def _velocity(self, z_list: List[torch.Tensor], t: torch.Tensor,
                  cond_n_list: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """Per-level velocity. ``t`` is ``[B]`` in [0,1]; ``cond_n`` standardised."""
        t_emb = _sinusoidal_time_embedding(t, self.t_dim)
        return [lv(z, c, t_emb) for lv, z, c in zip(self.levels, z_list, cond_n_list)]

    # ---- training objective ----------------------------------------------
    def flow_matching_loss(self, target_list: Sequence[torch.Tensor],
                           cond_list: Sequence[torch.Tensor]) -> torch.Tensor:
        """Rectified-flow velocity-matching loss in *cond-standardised* space.

        Per level, both target and cond are standardised by the **cond**
        statistics (so train matches the restore path, which only has cond
        stats).  Then ``z0 ~ N(0,I)``, ``t ~ U(0,1)``, ``z_t = (1-t) z0 + t z1n``
        with ``z1n`` the standardised target; regress ``v_theta(z_t, t | condn)``
        onto ``z1n - z0``.
        """
        assert len(target_list) == len(self.channels), "level count mismatch"
        b = target_list[0].shape[0]
        device = target_list[0].device
        t = torch.rand(b, device=device)
        t_b = t[:, None, None, None]
        z_t_list, vtarg_list, condn_list = [], [], []
        for z1, cond in zip(target_list, cond_list):
            mu_c, std_c = self._stats(cond)
            z1n = (z1 - mu_c) / std_c
            condn = (cond - mu_c) / std_c
            z0 = torch.randn_like(z1n)
            z_t_list.append((1.0 - t_b) * z0 + t_b * z1n)
            vtarg_list.append(z1n - z0)
            condn_list.append(condn)
        vpred_list = self._velocity(z_t_list, t, condn_list)
        loss = sum(F.mse_loss(vp, vt) for vp, vt in zip(vpred_list, vtarg_list))
        return loss / max(len(self.channels), 1)

    # ---- sampling + uncertainty ------------------------------------------
    def restore(
        self, cond_list: Sequence[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Few-step ODE restoration with K-seed uncertainty.

        Integrates in cond-standardised space from ``z0 ~ N(0,I)`` and
        de-standardises the endpoint back to raw feature scale (so the restored
        content can feed the decoder directly).

        Returns:
            z_hat_list: mean restored content pyramid (raw scale, keeps graph).
            sigma2: per-image restoration uncertainty ``[B,1,1,1]`` — seed
                variance of the ODE endpoint in *standardised* (scale-free)
                space, averaged over C/H/W and levels.  Dimensionless ⇒ flow_tau
                is a clean knob in ``exp(-sigma2/flow_tau)``.
            sigma2_map: **per-position** restoration uncertainty
                ``[B,1,H0,W0]`` (#18) — the same seed variance averaged over C
                and seeds but **kept spatial**, levels bilinearly upsampled to
                the finest pyramid resolution and averaged.  Global aggregation
                (``sigma2``) washes out almost all dynamic range across images
                (≈ a constant by the law of large numbers); the per-position
                map retains the signal needed for a selective β_rec(x) gate.
        """
        dt = 1.0 / self.n_steps
        b = cond_list[0].shape[0]
        device = cond_list[0].device
        stats = [self._stats(c) for c in cond_list]
        condn_list = [(c - mu) / std for c, (mu, std) in zip(cond_list, stats)]

        # per_level_samples[level] -> list of [B,C,H,W] standardised endpoints
        per_level_samples: List[List[torch.Tensor]] = [[] for _ in self.channels]
        for _k in range(self.n_samples):
            z_list = [torch.randn_like(c) for c in condn_list]
            for step in range(self.n_steps):
                t = torch.full((b,), (step + 0.5) * dt, device=device)
                v_list = self._velocity(z_list, t, condn_list)
                z_list = [z + dt * v for z, v in zip(z_list, v_list)]
            for li, z in enumerate(z_list):
                per_level_samples[li].append(z)

        z_hat_list: List[torch.Tensor] = []
        sigma2_levels: List[torch.Tensor] = []        # [B,1,1,1] per level (global)
        sigma2_maps: List[torch.Tensor] = []          # [B,1,H0,W0] per level (spatial)
        ref_hw = tuple(per_level_samples[0][0].shape[-2:])  # finest level resolution
        for li, (mu, std) in enumerate(stats):
            stack = torch.stack(per_level_samples[li], dim=0)  # [K,B,C,H,W] standardised
            mean_n = stack.mean(dim=0)                          # [B,C,H,W]
            z_hat_list.append(mean_n * std + mu)               # de-standardise → raw scale
            if self.n_samples > 1:
                var = stack.var(dim=0, unbiased=False)         # [B,C,H,W] (scale-free)
                sigma2_levels.append(var.mean(dim=(1, 2, 3), keepdim=True))  # [B,1,1,1]
                vmap = var.mean(dim=1, keepdim=True)           # [B,1,H_l,W_l] per-position
                if tuple(vmap.shape[-2:]) != ref_hw:
                    vmap = F.interpolate(vmap, size=ref_hw, mode="bilinear", align_corners=False)
                sigma2_maps.append(vmap)
            else:
                sigma2_levels.append(torch.zeros(b, 1, 1, 1, device=device))
                sigma2_maps.append(torch.zeros(b, 1, *ref_hw, device=device))
        sigma2 = torch.stack(sigma2_levels, dim=0).mean(dim=0)    # [B,1,1,1]
        sigma2_map = torch.stack(sigma2_maps, dim=0).mean(dim=0)  # [B,1,H0,W0]
        if self.sigma_detach:
            sigma2 = sigma2.detach()
            sigma2_map = sigma2_map.detach()
        return z_hat_list, sigma2, sigma2_map

    # ---- learned reliability ---------------------------------------------
    @staticmethod
    def _reliability_features(
        z_hat_list: Sequence[torch.Tensor],
        cond_list: Sequence[torch.Tensor],
        sigma2_map: torch.Tensor,
    ) -> torch.Tensor:
        """Scale-free restore-time feature stack at the σ²_map grid: the σ²(x)
        map plus, per level, mean|ẑ|, mean|cond|, and the cross-modal residual
        ‖ẑ−cond‖ (channel-reduced).  No z_true ⇒ usable at eval."""
        ref = sigma2_map.shape[-2:]

        def _to_ref(t):
            return t if t.shape[-2:] == ref else F.interpolate(
                t, size=ref, mode="bilinear", align_corners=False)

        feats = [sigma2_map]
        for zh, cd in zip(z_hat_list, cond_list):
            feats.append(_to_ref(zh.abs().mean(1, keepdim=True)))
            feats.append(_to_ref(cd.abs().mean(1, keepdim=True)))
            feats.append(_to_ref(((zh - cd) ** 2).mean(1, keepdim=True).clamp_min(1e-12).sqrt()))
        return torch.cat(feats, dim=1)

    def predict_reliability(
        self,
        z_hat_list: Sequence[torch.Tensor],
        cond_list: Sequence[torch.Tensor],
        sigma2_map: torch.Tensor,
    ) -> torch.Tensor:
        """Per-position reliability logit h(x) (higher ⇒ more predicted error),
        ``[B,1,H0,W0]``.

        Features are **detached** — the head is a pure read-out predictor; it is
        trained only by its own regression loss (``L_rel``) against the
        per-image-standardised restoration error, and must not reshape the
        flow/encoders to make error easier to rate.  ``h(x)`` then drives
        ``β_rec(x)=β·sigmoid(−h(x)/τ)``.
        """
        assert self.reliability_head is not None, "learned_reliability=False"
        feat = self._reliability_features(
            [z.detach() for z in z_hat_list],
            [c.detach() for c in cond_list],
            sigma2_map.detach(),
        )
        return self.reliability_head(feat)
