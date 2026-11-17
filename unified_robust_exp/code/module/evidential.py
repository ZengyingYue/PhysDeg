"""CoDER evidential utilities.

Implements the three pillars of the CoDER proposal:

  1. Dirichlet evidence head (Sensoy et al., NeurIPS'18) — replaces softmax.
       evidence e_k >= 0  →  Dirichlet alpha_k = e_k + 1
       p_k = alpha_k / S,    u = K / S       (S = sum_k alpha_k)

  2. Covariance → reliability discount  beta in [0, 1].
       Per cov-tracked encoder layer l we own a [C_l, C_l] variance-of-
       covariance matrix T_l accumulated by MMR.  We aggregate its row sums
       into a per-channel sensitivity score and then average across channels
       into a single scalar beta_l per layer.  High variance => unreliable
       => beta_l small.  An extra learnable discount gamma_rec scales beta
       further when the modality is the *recovered* (not in-place) branch.
       The discount is applied as  e_k <- beta * e_k  (Yager / DST style:
       discounted mass goes to the vacuous u).

  3. Dempster-Shafer combination of two evidence streams (per-pixel form,
       following the Trusted Multi-View Classification setup, Han et al.
       ICLR'21).  Replaces MetaRS's uniform mean over recovered features.

Also provides the EDL training loss (Sensoy Bayesian risk + annealed KL
to the uniform prior) and an Expected Calibration Error utility used for
the §5.4 calibration analysis.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pillar 1 — evidence / Dirichlet utilities
# ---------------------------------------------------------------------------

def logits_to_evidence(logits: torch.Tensor, activation: str = "softplus") -> torch.Tensor:
    """Map raw network outputs to non-negative evidence e_k.

    Args:
        logits: [B, K, H, W] (or [B, K]) raw outputs.
        activation: "softplus" (default, smooth) or "relu" or "exp".
    """
    if activation == "softplus":
        return F.softplus(logits)
    if activation == "relu":
        return F.relu(logits)
    if activation == "exp":
        return torch.exp(torch.clamp(logits, max=10.0))
    raise ValueError(f"Unknown evidence activation: {activation}")


def evidence_to_alpha(evidence: torch.Tensor) -> torch.Tensor:
    return evidence + 1.0


def alpha_to_prob(alpha: torch.Tensor, dim: int = 1) -> torch.Tensor:
    S = alpha.sum(dim=dim, keepdim=True)
    return alpha / S


def alpha_to_uncertainty(alpha: torch.Tensor, num_classes: int, dim: int = 1) -> torch.Tensor:
    """Vacuous-belief mass u = K / S, shape matches alpha with `dim` collapsed."""
    S = alpha.sum(dim=dim, keepdim=True)
    return num_classes / S


# ---------------------------------------------------------------------------
# Evidential Deep Learning loss (Sensoy 2018)
# ---------------------------------------------------------------------------

def _one_hot(y: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    """y: [B, H, W] (any int dtype); returns [B, K, H, W] float, zeros at ignore.

    Cast to int64 internally because F.one_hot requires LongTensor — the
    dataset hands us int8 masks.
    """
    y_long = y.long()
    valid = (y_long != ignore_index)
    y_safe = torch.where(valid, y_long, torch.zeros_like(y_long))
    onehot = F.one_hot(y_safe, num_classes=num_classes)        # [B, H, W, K]
    onehot = onehot.permute(0, 3, 1, 2).float()                # [B, K, H, W]
    onehot = onehot * valid.unsqueeze(1).float()
    return onehot, valid


def edl_loss(
    alpha: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: int,
    kl_lambda: float = 1.0,
    ignore_index: int = -1,
    form: str = "mse",
    ce_aux_weight: float = 0.5,
    class_weights: torch.Tensor | None = None,
    ohem_keep_ratio: float = 1.0,
) -> torch.Tensor:
    """Sensoy EDL loss with two formulations + optional CE auxiliary.

    form="risk" — original Bayesian risk:
        sum_k y_k * (psi(S) - psi(alpha_k))
      (slow convergence, easily trapped in vacuous region for K>=8)

    form="mse" (default, Sensoy 2018 §3.2, recommended for segmentation):
        sum_k [(y_k - p_k)^2 + p_k * (1 - p_k) / (S + 1)],  p_k = alpha_k / S
      Strong gradient even at low evidence; the variance term encourages
      sharp Dirichlets only when justified.

    Plus a small CE auxiliary  -log(p_y)  on the expected probability, which
    gives the same gradient direction as cross-entropy and breaks the
    "vacuous attractor" early in training.  Total:

        L = L_form + kl_lambda * KL[Dir(alpha_tilde) || Dir(1)]
            + ce_aux_weight * (-log p_y)
    """
    y_onehot, valid = _one_hot(y_true, num_classes, ignore_index)
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S  # expected probabilities

    if form == "risk":
        data_term = (y_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1)
    elif form == "mse":
        squared = (y_onehot - p).pow(2).sum(dim=1)
        variance = (p * (1.0 - p) / (S + 1.0)).sum(dim=1)
        data_term = squared + variance
    else:
        raise ValueError(f"Unknown edl_loss form: {form}")

    # KL[ Dir(alpha_tilde) || Dir(1) ] -- correct-class evidence removed
    alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
    kl = _kl_dirichlet_uniform(alpha_tilde, num_classes)

    per_pixel = data_term + kl_lambda * kl
    if ce_aux_weight > 0.0:
        # numerically stable CE on expected prob; p has no zeros (alpha>=1)
        log_p_y = (y_onehot * torch.log(p.clamp_min(1e-8))).sum(dim=1)
        per_pixel = per_pixel + ce_aux_weight * (-log_p_y)

    valid_f = valid.float()

    # Per-pixel class weight — boost tail classes (Agricultural, Playground)
    # via `class_weights[k]` indexed by ground-truth class.  Background-only
    # pixels (ignore_index) get weight 0 through the valid_f mask.
    if class_weights is not None:
        cw = class_weights.to(per_pixel.device).float()  # [K]
        y_long = y_true.long().clamp_min(0)
        pixel_w = cw[y_long]  # [B, H, W]
        per_pixel = per_pixel * pixel_w
        valid_f = valid_f * pixel_w  # so the denominator matches

    # OHEM: keep the hardest-loss `ohem_keep_ratio` fraction of valid pixels.
    # Attacks long-tail classes whose pixels are usually high-loss.
    if ohem_keep_ratio < 1.0:
        flat_loss = (per_pixel * valid_f).view(-1)
        flat_valid = valid_f.view(-1) > 0
        valid_loss = flat_loss[flat_valid]
        if valid_loss.numel() == 0:
            return torch.zeros((), device=per_pixel.device, requires_grad=True)
        n_keep = max(1, int(valid_loss.numel() * ohem_keep_ratio))
        topk_loss, _ = torch.topk(valid_loss, n_keep)
        return topk_loss.mean()

    denom = valid_f.sum().clamp_min(1.0)
    return (per_pixel * valid_f).sum() / denom


def dirichlet_kl_distillation(
    alpha_student: torch.Tensor,
    alpha_teacher: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL[ Dir(student) || Dir(teacher.detach()) ] per-pixel, mean over valid.

    Used to align the recovered-modality evidence  decoder_X(other_features)
    with the real-modality evidence  decoder_X(self_features).  Teacher is
    detached so gradients only flow into the recovery path.
    """
    alpha_teacher = alpha_teacher.detach()
    S_s = alpha_student.sum(dim=1, keepdim=True)
    S_t = alpha_teacher.sum(dim=1, keepdim=True)
    # KL = lgamma(S_s) - sum lgamma(alpha_s)
    #    - lgamma(S_t) + sum lgamma(alpha_t)
    #    + sum (alpha_s - alpha_t) (psi(alpha_s) - psi(S_s))
    term_a = torch.lgamma(S_s).squeeze(1) - torch.lgamma(alpha_student).sum(dim=1)
    term_b = -torch.lgamma(S_t).squeeze(1) + torch.lgamma(alpha_teacher).sum(dim=1)
    term_c = ((alpha_student - alpha_teacher)
              * (torch.digamma(alpha_student) - torch.digamma(S_s))).sum(dim=1)
    per_pixel = term_a + term_b + term_c
    if valid_mask is not None:
        valid_f = valid_mask.float()
        denom = valid_f.sum().clamp_min(1.0)
        return (per_pixel * valid_f).sum() / denom
    return per_pixel.mean()


def soft_dice_loss(
    prob: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Multi-class soft Dice on probabilities (e.g. alpha/S).

    prob:   [B, K, H, W] probabilities (sum-to-1 over K).
    y_true: [B, H, W] long.
    Skips ignore_index pixels and channels with no positive samples.
    """
    y_onehot, valid = _one_hot(y_true, num_classes, ignore_index)
    valid_f = valid.unsqueeze(1).float()
    prob_v = prob * valid_f
    onehot_v = y_onehot
    # per-class Dice
    inter = (prob_v * onehot_v).sum(dim=(0, 2, 3))
    union = prob_v.sum(dim=(0, 2, 3)) + onehot_v.sum(dim=(0, 2, 3))
    dice = (2.0 * inter + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def _kl_dirichlet_uniform(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Per-pixel KL[ Dir(alpha) || Dir(1, ..., 1) ] (Sensoy 2018 Eq. 12).

    alpha: [B, K, H, W]; returns [B, H, W].
    """
    S = alpha.sum(dim=1, keepdim=True)
    K = num_classes
    # KL = log(Gamma(S)/Gamma(K)) - sum_k log(Gamma(alpha_k))
    #    + sum_k (alpha_k - 1) * (psi(alpha_k) - psi(S))
    term1 = torch.lgamma(S).squeeze(1) - torch.lgamma(torch.tensor(float(K), device=alpha.device))
    term2 = torch.lgamma(alpha).sum(dim=1)
    term3 = ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S))).sum(dim=1)
    return term1 - term2 + term3


def kl_annealing(step: int, ramp_steps: int, max_lambda: float = 1.0) -> float:
    """Linear warmup of the KL regularizer."""
    if ramp_steps <= 0:
        return max_lambda
    return float(max_lambda * min(1.0, step / float(ramp_steps)))


# ---------------------------------------------------------------------------
# Pillar 3 — Dempster combination of two Dirichlet evidence streams
# ---------------------------------------------------------------------------

def dempster_combine(alpha_a: torch.Tensor, alpha_b: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Combine two Dirichlets at every pixel using the rule of Han et al. (TMC).

    Inputs:  [B, K, H, W] per-pixel Dirichlet alphas (alpha = e + 1).
    Output:  [B, K, H, W] combined Dirichlet alpha.

    Procedure per pixel:
        b^m_k = (alpha^m_k - 1) / S^m,    u^m = K / S^m
        C = sum_{i != j} b^a_i b^b_j                   (conflict mass)
        b_k = (b^a_k b^b_k + b^a_k u^b + b^b_k u^a) / (1 - C)
        u   = u^a u^b / (1 - C)
        S_combined = K / u,    alpha_k = b_k * S_combined + 1
    """
    K = num_classes
    S_a = alpha_a.sum(dim=1, keepdim=True)
    S_b = alpha_b.sum(dim=1, keepdim=True)
    b_a = (alpha_a - 1.0) / S_a
    b_b = (alpha_b - 1.0) / S_b
    u_a = K / S_a
    u_b = K / S_b

    # conflict C = sum_i sum_{j!=i} b_a_i b_b_j = (sum_i b_a_i)(sum_j b_b_j) - sum_k b_a_k b_b_k
    bb = b_a * b_b
    sum_a = b_a.sum(dim=1, keepdim=True)
    sum_b = b_b.sum(dim=1, keepdim=True)
    conflict = sum_a * sum_b - bb.sum(dim=1, keepdim=True)
    one_minus_C = (1.0 - conflict).clamp_min(1e-6)

    b_comb = (bb + b_a * u_b + b_b * u_a) / one_minus_C
    u_comb = (u_a * u_b / one_minus_C).clamp_min(1e-6)
    S_comb = K / u_comb
    alpha_comb = b_comb * S_comb + 1.0
    return alpha_comb


# ---------------------------------------------------------------------------
# Pillar 2 — covariance-derived reliability discount  beta
# ---------------------------------------------------------------------------

class CovarianceReliability(nn.Module):
    """Maps the variance-of-covariance matrices owned by MMR into a single
    scalar reliability coefficient beta in (0, 1].

    For each cov-tracked layer l with off-diagonal variance matrix T_l
    (shape [C_l, C_l]), we compute the per-channel sensitivity
        s_l_c = sum_{c'} T_l[c, c'] + sum_{c'} T_l[c', c]
    Normalize to [0, 1] across c, take the mean, and pass through
        beta_l = sigmoid( -alpha * (s_mean - 0.5) ) * (1 - eps) + eps
    so that high mean sensitivity drives beta toward eps (= floor).
    The final reliability for in-modality evidence is the mean of beta_l
    across layers; for the recovered modality it is further discounted by
    a learnable gamma_rec in (0, 1).

    The module also exposes a "stale" mode: until the MMR mask matrices
    have been computed (early training), beta defaults to 1.0 (full trust),
    which makes the evidential head behave like plain EDL.
    """

    def __init__(self, scale: float = 6.0, eps: float = 0.05,
                 gamma_rec_init: float = 0.7,
                 ramp_iters: int = 0,
                 ramp_begin_iter: int = 0):
        super().__init__()
        self.scale = float(scale)
        self.eps = float(eps)
        # gamma_rec stored in raw param-space and squashed to (0, 1)
        self.gamma_rec_raw = nn.Parameter(
            torch.tensor(_inverse_sigmoid(gamma_rec_init), dtype=torch.float32)
        )
        # Linear blend between β=1.0 (full evidence) and the computed β over
        # `ramp_iters` steps starting at `ramp_begin_iter`.  Lets the EDL head
        # mature before MMR's covariance-derived discount kicks in fully.
        # The blend factor is set externally via `set_ramp_progress(step)`.
        self.ramp_iters = int(ramp_iters)
        self.ramp_begin_iter = int(ramp_begin_iter)
        self.register_buffer("_ramp_step", torch.tensor(0, dtype=torch.long))

    @property
    def gamma_rec(self) -> torch.Tensor:
        return torch.sigmoid(self.gamma_rec_raw)

    def _beta_from_var(self, var_matrix: torch.Tensor) -> torch.Tensor:
        # var_matrix: [C, C].  Aggregate to per-channel sensitivity then mean.
        row_sum = var_matrix.sum(dim=1) + var_matrix.sum(dim=0)
        # robust min/max normalize
        vmin = row_sum.min()
        vmax = row_sum.max()
        denom = (vmax - vmin).clamp_min(1e-8)
        normed = (row_sum - vmin) / denom
        s_mean = normed.mean()
        beta = torch.sigmoid(-self.scale * (s_mean - 0.5))
        return beta * (1.0 - self.eps) + self.eps

    def _ramp_factor(self) -> float:
        """Linear ramp coefficient r in [0, 1] applied to the discount strength.
        r=0 means β=1.0 (no discount); r=1 means full MMR-derived β.
        """
        if self.ramp_iters <= 0:
            return 1.0
        step = int(self._ramp_step.item())
        if step < self.ramp_begin_iter:
            return 0.0
        progress = (step - self.ramp_begin_iter) / float(self.ramp_iters)
        return float(min(1.0, max(0.0, progress)))

    def set_ramp_step(self, step: int) -> None:
        self._ramp_step.fill_(int(step))

    def compute(self, cov_matrix_layers, recovered: bool = False) -> torch.Tensor:
        """Return scalar beta tensor on the same device as the cov mats.

        cov_matrix_layers: list of CovMatrix_mmr objects from MetaRS/CoDER.
        recovered: if True, multiply by gamma_rec for extra discounting.
        """
        device = self.gamma_rec_raw.device
        betas = []
        for layer in cov_matrix_layers:
            vm = getattr(layer, "var_matrix", None)
            if vm is None or not torch.is_tensor(vm):
                # MMR hasn't accumulated yet -- treat as fully reliable.
                betas.append(torch.tensor(1.0, device=device))
                continue
            betas.append(self._beta_from_var(vm.detach().to(device)))
        if not betas:
            return torch.tensor(1.0, device=device)
        beta_mmr = torch.stack(betas).mean()
        # Blend with β=1.0 to soften the post-MMR transition.
        r = self._ramp_factor()
        beta = r * beta_mmr + (1.0 - r) * torch.ones_like(beta_mmr)
        if recovered:
            beta = beta * self.gamma_rec
        else:
            # Keep gamma_rec_raw in the autograd graph even on
            # complete-modality steps so DDP doesn't flag it as unused.
            beta = beta + 0.0 * self.gamma_rec_raw
        return beta


def discount_evidence(evidence: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Yager-style discount: e_k <- beta * e_k.  The mass (1-beta) flows
    into the vacuous u via the alpha = e + 1 transform."""
    if beta.dim() == 0:
        return evidence * beta
    # broadcast: beta can be [B, 1, H, W] or [B, 1, 1, 1]
    return evidence * beta


# ---------------------------------------------------------------------------
# Calibration metric (§5.4)
# ---------------------------------------------------------------------------

def expected_calibration_error(
    prob: torch.Tensor,
    y_true: torch.Tensor,
    num_bins: int = 15,
    ignore_index: int = -1,
) -> torch.Tensor:
    """Standard ECE on per-pixel max-class predictions.

    prob:    [B, K, H, W]   probabilities (alpha/S for evidential).
    y_true:  [B, H, W]      labels.
    """
    conf, pred = prob.max(dim=1)
    conf = conf.flatten()
    pred = pred.flatten()
    gt = y_true.flatten()
    mask = gt != ignore_index
    conf, pred, gt = conf[mask], pred[mask], gt[mask]
    if conf.numel() == 0:
        return torch.tensor(0.0, device=prob.device)
    correct = (pred == gt).float()
    ece = torch.zeros((), device=prob.device)
    bin_edges = torch.linspace(0.0, 1.0, num_bins + 1, device=prob.device)
    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        n = in_bin.sum()
        if n == 0:
            continue
        avg_conf = conf[in_bin].mean()
        acc = correct[in_bin].mean()
        ece = ece + (n.float() / conf.numel()) * (avg_conf - acc).abs()
    return ece


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inverse_sigmoid(p: float) -> float:
    p = max(min(p, 0.999), 0.001)
    import math
    return math.log(p / (1.0 - p))
