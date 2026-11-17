"""Baselines re-implemented under the revision protocol (registered models).

All three share ResNet-50 encoders, the FPN + asymmetric decoder, the
segmentation loss (class-weighted, label-smoothed, OHEM cross-entropy + Dice)
and per-sample modality dropout (each sample independently complete /
SAR-missing / optical-missing with probability 1/3).  Only the fusion and the
method-specific objectives differ.

ModalityDropoutFusion  per-level 1x1-conv fusion of the concatenated streams;
                       a missing modality is zero-filled (mean in normalised
                       units).
PADFusionR50           PAD (Zheng et al., TGRS 2025) phase-amplitude
                       decoupling fusion at every level, with its amplitude
                       penalty; zero-filled missing input.
ShaSpec                Wang et al., CVPR 2023: shared encoder r, specific
                       encoders s, f = proj([r, s]) + r; a missing modality's
                       feature is the mean shared feature of the available
                       ones; domain-classification loss on s (beta = 0.02) and
                       L1 distribution alignment of r (alpha = 0.1).  Fusion is
                       applied at every pyramid level.

``infer_mode``: 'fuse' (both), 'sar' (optical missing), 'rgb' (SAR missing).
"""
from __future__ import annotations

import copy
import logging
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from ever import registry
from ever.interface import ERModule
from ever.module import FPN

from module.baseline.base import AssymetricDecoder, default_conv_block
from module.baseline.base_resnet.resnet import ResNetEncoder
from module.evidential import soft_dice_loss
from module.reliability import pixel_stats as pxs
from module.reliability import phys_degrade as phys
from module.reliability import train_degrade as tdeg

logger = logging.getLogger(__name__)

COMPLETE, SAR_MISSING, RGB_MISSING = 0, 1, 2


class _DualSegBase(ERModule):
    """Shared scaffolding: encoders, decoder, loss, modality dropout."""

    fuse_channels = (256, 512, 1024, 2048)

    def __init__(self, config):
        super().__init__(config)
        self.infer_mode = self.config.infer_mode
        # Spatial occlusion of the optical input during training (revision
        # ablation): 'hole' removes the occluded content (spatial modality
        # dropout), 'cloud' blends in a synthetic cloud (plain augmentation).
        self.occlusion_mode = str(self.config.get("occlusion_mode", "none"))
        self.occlusion_prob = float(self.config.get("occlusion_prob", 0.5))
        self.register_buffer("cloud_value", torch.tensor(
            list(self.config.get("cloud_value_norm", [3.0, 3.0, 3.0])), dtype=torch.float32),
            persistent=False)
        # Degradation-posterior distillation (DPD screening): random seen-type
        # degradations of the student input, and a KL term toward an EMA
        # teacher's prediction on the clean complete pair.
        deg = dict(self.config.get("degrade", None) or {})
        self.deg_prob = float(deg.get("prob", 0.0))
        self.deg_sar_prob = float(deg.get("sar_prob", 0.0))
        self.deg_types = tuple(deg.get("types", tdeg.OPTICAL_TYPES))
        # Physical degradations (PHYS_PLAN.md): optical / sar = 'generic' | 'phys'
        # ('both' for optical), co-registration shifts, dataset radiometry.
        self.deg_optical = str(deg.get("optical", "generic"))
        self.deg_sar = str(deg.get("sar", "generic"))
        self.deg_coreg_prob = float(deg.get("coreg_prob", 0.0))
        self.deg_coreg_max = float(deg.get("coreg_max", 12.0))
        self.deg_unit = float(deg.get("unit", 1.0))
        self.deg_max_dn = float(deg.get("max_dn", 255.0))
        self.deg_sar_domain = str(deg.get("sar_domain", "amp"))
        self.deg_gsd = float(deg.get("gsd", 5.0))
        # Component ablations of the physical optical families (PHYS_PLAN.md section 5).
        self.deg_ablate = tuple(deg.get("ablate", ()))
        self.deg_optical_types = tuple(deg.get("optical_types", phys.OPTICAL_TYPES))
        self.deg_coverage = deg.get("coverage", "uniform")
        self._rel_valid = None
        # Reliability-supervised subclasses set want_rel: the physical optical
        # degradation then also yields a per-pixel reliability target.
        self.want_rel = False
        self._rel_target = None
        for name in ("rgb_mean", "rgb_std", "sar_mean", "sar_std"):
            self.register_buffer(name, torch.tensor(list(deg.get(name, [0.0])),
                                                    dtype=torch.float32), persistent=False)
        kd = dict(self.config.get("kd", None) or {})
        self.kd_lam = float(kd.get("lam", 0.0))
        self.kd_start = int(kd.get("start", 2000))
        self.kd_ramp = int(kd.get("ramp", 4000))
        self.kd_ema = float(kd.get("ema", 0.999))
        self._teacher = []  # held in a list so it is neither a submodule nor saved
        self._kd_step = 0
        cw = self.config.loss.get("class_weights", None)
        self.register_buffer("class_weights",
                             torch.tensor(list(cw), dtype=torch.float32) if cw else torch.ones(0),
                             persistent=False)
        self.decoder = nn.Sequential(
            FPN(in_channels_list=self.fuse_channels, out_channels=256,
                conv_block=default_conv_block, top_blocks=None),
            AssymetricDecoder(in_channels=256, out_channels=128,
                              in_feat_output_strides=(4, 8, 16, 32), out_feat_output_stride=4),
            nn.Conv2d(128, self.config.num_classes, 1),
            nn.UpsamplingBilinear2d(scale_factor=4),
        )

    # ------------------------------------------------------------------ utils
    def _encoder(self, in_channels):
        cfg = copy.deepcopy(self.config.encoder)
        cfg.in_channels = in_channels
        cfg.wt_layer = [0] * 7
        return ResNetEncoder(cfg)

    def _split(self, x):
        s = self.config.sar_in_channels
        return x[:, :s], x[:, s:s + self.config.rgb_in_channels]

    def _sample_modes(self, batch, device):
        return torch.randint(0, 3, (batch,), device=device)

    def _eval_modes(self, batch, device):
        code = {"fuse": COMPLETE, "sar": RGB_MISSING, "rgb": SAR_MISSING}[self.infer_mode]
        return torch.full((batch,), code, dtype=torch.long, device=device)

    def _seg_loss(self, logits, y):
        loss = self.config.loss
        cw = self.class_weights if self.class_weights.numel() else None
        ce = F.cross_entropy(logits, y.long(), weight=cw, ignore_index=-1, reduction="none",
                             label_smoothing=float(loss.get("label_smoothing", 0.0)))
        valid = y.long() != -1
        flat = ce[valid]
        keep = float(loss.get("ohem_keep_ratio", 1.0))
        if keep < 1.0 and flat.numel():
            flat = torch.topk(flat, max(1, int(flat.numel() * keep))).values
        out = {"ce_loss": flat.mean() if flat.numel() else logits.sum() * 0}
        lam_dice = float(loss.get("lam_dice", 0.0))
        if lam_dice > 0:
            out["dice_loss"] = lam_dice * soft_dice_loss(
                F.softmax(logits, dim=1), y, self.config.num_classes, -1)
        return out

    def custom_param_groups(self):
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if p.requires_grad:
                (no_decay if p.ndim <= 1 else decay).append(p)
        return [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]

    @torch.no_grad()
    def _teacher_probs(self, sar, rgb):
        """EMA teacher on the clean complete pair (created lazily on first use)."""
        if not self._teacher:
            teacher = copy.deepcopy(self)
            teacher._teacher = []
            teacher.kd_lam = 0.0
            teacher.eval().requires_grad_(False)
            self._teacher.append(teacher)
        teacher = self._teacher[0]
        m = min(self.kd_ema, (1.0 + self._kd_step) / (10.0 + self._kd_step))
        for pt, ps in zip(teacher.parameters(), self.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1 - m)
        for bt, bs in zip(teacher.buffers(), self.buffers()):
            bt.copy_(bs)
        modes = torch.full((sar.shape[0],), COMPLETE, dtype=torch.long, device=sar.device)
        logits, _ = teacher.segment(sar, rgb, modes)
        return F.softmax(logits.float(), dim=1)

    def _kd_loss(self, logits, teacher, y):
        valid = (y.long() != -1).unsqueeze(1).to(logits.dtype)
        logp = F.log_softmax(logits.float(), dim=1)
        kl = (teacher * (torch.log(teacher.clamp_min(1e-8)) - logp)).sum(1, keepdim=True)
        return (kl * valid).sum() / valid.sum().clamp_min(1.0)

    def forward(self, x, y=None):
        sar, rgb = self._split(x)
        modes = (self._sample_modes(x.shape[0], x.device) if self.training
                 else self._eval_modes(x.shape[0], x.device))
        sar_clean, rgb_clean = sar, rgb
        self._rel_target = None
        if self.training and self.deg_prob > 0:
            rgb = self._degrade_rgb(rgb)
        if self.training and self.deg_sar_prob > 0:
            if self.deg_sar == "phys":
                sar = phys.random_sar(sar, self.sar_mean, self.sar_std, self.deg_sar_prob,
                                      self.deg_sar_domain)
            else:
                sar = tdeg.degrade_sar(sar, self.sar_mean, self.sar_std, self.deg_sar_prob,
                                       domain=self.deg_sar_domain)
        if self.training and self.deg_coreg_prob > 0:
            sar = phys.random_coreg(sar, self.deg_coreg_prob, self.deg_coreg_max)
        occlusion = None
        if self.training and self.occlusion_mode != "none":
            chosen = torch.rand(x.shape[0], device=x.device) < self.occlusion_prob
            mask = pxs.blob_masks(x.shape[0], x.shape[2], x.shape[3], x.device)
            mask = mask * chosen.view(-1, 1, 1, 1).float()
            if self.occlusion_mode == "cloud":
                opacity = 0.6 + 0.4 * torch.rand(x.shape[0], 1, 1, 1, device=x.device)
                rgb = rgb * (1 - mask * opacity) + self.cloud_value.view(1, -1, 1, 1) * mask * opacity
            else:
                rgb = rgb * (1 - mask)
                occlusion = mask
        logits, aux = self.segment(sar, rgb, modes, occlusion) if occlusion is not None \
            else self.segment(sar, rgb, modes)
        if not self.training:
            return F.softmax(logits.float(), dim=1)
        losses = self._seg_loss(logits, y["cls"])
        losses.update(aux)
        if self.kd_lam > 0:
            self._kd_step += 1
            if self._kd_step >= self.kd_start:
                ramp = min(1.0, (self._kd_step - self.kd_start + 1) / max(self.kd_ramp, 1))
                teacher = self._teacher_probs(sar_clean, rgb_clean)
                losses["kd_loss"] = self.kd_lam * ramp * self._kd_loss(logits, teacher, y["cls"])
        losses["mem"] = torch.tensor(float(torch.cuda.max_memory_allocated() // 2 ** 20),
                                     device=x.device)
        return losses

    def _degrade_rgb(self, rgb):
        args = (self.rgb_mean, self.rgb_std)
        p = self.deg_prob
        if self.deg_optical == "both":  # each family independently; P(any) = prob
            p = 1 - (1 - p) ** 0.5
        generic_hit = None
        if self.deg_optical in ("generic", "both"):
            before = rgb.clone() if self.want_rel else None
            rgb = tdeg.degrade_optical(rgb, *args, p, self.deg_types, self.deg_max_dn,
                                       self.deg_unit)
            if self.want_rel:  # generic families carry no reliability target: exclude them
                # (untouched images still pass a float de/re-normalisation: compare with a tolerance)
                generic_hit = (rgb.float() - before.float()).abs().flatten(1).amax(1) > 1e-3
        if self.deg_optical in ("phys", "both"):
            out = phys.random_optical(rgb, *args, p, self.deg_unit, self.deg_max_dn,
                                      self.deg_gsd, self.deg_optical_types, self.deg_ablate,
                                      return_rel=self.want_rel,
                                      coverage_mode=self.deg_coverage)
            rgb, self._rel_target = out if self.want_rel else (out, None)
        self._rel_valid = None if generic_hit is None else ~generic_hit
        return rgb

    def segment(self, sar, rgb, modes):  # pragma: no cover - abstract
        raise NotImplementedError

    def set_default_config(self):
        self.config.update(dict(
            num_classes=7, sar_in_channels=1, rgb_in_channels=3, infer_mode="fuse",
            encoder=dict(resnet_type="resnet50", include_conv5=True, batchnorm_trainable=True,
                         pretrained=True, freeze_at=0, output_stride=32,
                         wt_layer=[0] * 7, num_classes=1),
            loss=dict(class_weights=None, label_smoothing=0.05, ohem_keep_ratio=0.5,
                      lam_dice=0.3),
        ))


class HarmonizedSegLoss(nn.Module):
    """The revision's shared segmentation loss with MetaRS's call signature.

    ``loss(pred, y, prefix)`` returns ``{prefix + 'ce_loss': ...}`` (plus
    ``{prefix}dice_loss`` for the fused head), matching ``SegmentationLoss``.
    """

    def __init__(self, cfg, num_classes):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        cw = cfg.get("class_weights", None)
        self.register_buffer("class_weights",
                             torch.tensor(list(cw), dtype=torch.float32) if cw else torch.ones(0),
                             persistent=False)

    def forward(self, y_pred, y_true, pre="", Lamada=1):
        cw = self.class_weights if self.class_weights.numel() else None
        ce = F.cross_entropy(y_pred, y_true.long(), weight=cw, ignore_index=-1, reduction="none",
                             label_smoothing=float(self.cfg.get("label_smoothing", 0.0)))
        flat = ce[y_true.long() != -1]
        keep = float(self.cfg.get("ohem_keep_ratio", 1.0))
        if keep < 1.0 and flat.numel():
            flat = torch.topk(flat, max(1, int(flat.numel() * keep))).values
        out = {f"{pre}ce_loss": flat.mean() if flat.numel() else y_pred.sum() * 0}
        lam_dice = float(self.cfg.get("lam_dice", 0.0))
        if pre == "fuse" and lam_dice > 0:
            out[f"{pre}dice_loss"] = lam_dice * soft_dice_loss(
                F.softmax(y_pred, dim=1), y_true, self.num_classes, -1)
        return out


def _zero_missing(sar, rgb, modes):
    """Zero-fill (dataset mean after normalisation) the missing stream per sample."""
    keep_sar = (modes != SAR_MISSING).view(-1, 1, 1, 1).to(sar.dtype)
    keep_rgb = (modes != RGB_MISSING).view(-1, 1, 1, 1).to(rgb.dtype)
    return sar * keep_sar, rgb * keep_rgb


@registry.MODEL.register("ModalityDropoutFusion")
class ModalityDropoutFusion(_DualSegBase):
    def __init__(self, config):
        super().__init__(config)
        self.en_sar = self._encoder(self.config.sar_in_channels)
        self.en_rgb = self._encoder(self.config.rgb_in_channels)
        self.fuse = nn.ModuleList(
            nn.Sequential(nn.Conv2d(2 * c, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(True))
            for c in self.fuse_channels)

    def segment(self, sar, rgb, modes, occlusion=None):
        sar, rgb = _zero_missing(sar, rgb, modes)
        fs, fr = self.en_sar(sar), self.en_rgb(rgb)
        if occlusion is not None:  # zero the optical stream where it was removed
            fr = [f * (1 - pxs.pool_to(occlusion, f.shape[-2:]).to(f.dtype)) for f in fr]
        fused = [f(torch.cat([a, b], 1)) for f, a, b in zip(self.fuse, fs, fr)]
        return self.decoder(fused), {}


def _resize(x, size):
    """Bilinear up-sampling or area down-sampling of a [B,1,h,w] map to ``size``."""
    if x.shape[-2] >= size[0]:
        return F.adaptive_avg_pool2d(x, size)
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


@registry.MODEL.register("ModalityDropoutFusionRel")
class ModalityDropoutFusionRel(ModalityDropoutFusion):
    """ModDrop + physically supervised optical reliability (PHYS_PLAN.md section 6).

    The physical optical degradation gives, for free, the per-pixel contrast
    transmittance r = dI/dJ of every training image (1 = clean surface signal,
    0 = fully hidden by cloud).  A small head on the stride-16 SAR + optical
    features predicts r (BCE on area-pooled targets; optical-missing samples
    excluded).  With ``gate`` the optical features at every level are
    interpolated per pixel between their value and the features of a missing
    (zero-filled) optical input, f' = f0 + r_hat (f - f0): r_hat = 0 reproduces
    exactly the optical-missing state that modality dropout trains, so the head
    turns sample-level modality dropout into a per-pixel, physically supervised
    one.  SAR-missing samples are never gated (no fallback exists).  Without
    ``gate`` the head is an auxiliary task only (control).
    """

    def __init__(self, config):
        super().__init__(config)
        rel = self.config.rel
        self.rel_lam = float(rel.lam)
        self.rel_gate = bool(rel.gate)
        self.rel_level = int(rel.level)
        if self.deg_prob <= 0 or self.deg_optical not in ("phys", "both"):
            raise ValueError("reliability supervision needs physical optical degradations")
        self.want_rel = True
        c = self.fuse_channels[self.rel_level]
        self.rel_head = nn.Sequential(
            nn.Conv2d(2 * c, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 1, 1))
        self.last_reliability = None

    def segment(self, sar, rgb, modes, occlusion=None):
        sar, rgb = _zero_missing(sar, rgb, modes)
        b = rgb.shape[0]
        fs = self.en_sar(sar)
        if self.rel_gate:  # one pass: the batch plus a missing (zero-filled) optical input
            both = self.en_rgb(torch.cat([rgb, torch.zeros_like(rgb[:1])], 0))
            fr, f0 = [f[:b] for f in both], [f[b:] for f in both]
        else:
            fr = self.en_rgb(rgb)
        level = self.rel_level
        logit = self.rel_head(torch.cat([fr[level], fs[level]], 1)).float()
        r = torch.sigmoid(logit)
        self.last_reliability = r.detach()
        if self.rel_gate:
            g = torch.where((modes == SAR_MISSING).view(-1, 1, 1, 1), torch.ones_like(r), r)
            fr = [z + _resize(g, f.shape[-2:]).to(f.dtype) * (f - z) for f, z in zip(fr, f0)]
        fused = [f(torch.cat([a, c], 1)) for f, a, c in zip(self.fuse, fs, fr)]
        aux = {}
        if self.training and self._rel_target is not None:
            target = F.adaptive_avg_pool2d(self._rel_target.float(), logit.shape[-2:])
            keep = modes != RGB_MISSING
            if self._rel_valid is not None:  # 'both': images hit by a generic family excluded
                keep = keep & self._rel_valid.to(keep.device)
            aux["rel_loss"] = self.rel_lam * F.binary_cross_entropy_with_logits(
                logit[keep], target[keep]) if bool(keep.any()) else logit.sum() * 0
        return self.decoder(fused), aux

    def set_default_config(self):
        super().set_default_config()
        self.config.update(dict(rel=dict(lam=1.0, gate=True, level=2)))


# ---------------------------------------------------------------------------
# PAD fusion module, ported from the official implementation
# (baselines/PAD/src/PAD_module.py) without the mmengine dependency.
# ---------------------------------------------------------------------------
class _PSC(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        hidden = dim // reduction if dim > reduction else 1
        self.attention = nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.LeakyReLU(),
                                       nn.Conv2d(hidden, 1, 1))

    def forward(self, phase):
        return phase * (1 + torch.sigmoid(self.attention(phase)))


class _MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers, last = [], in_dim
        for hidden in hidden_list:
            layers += [nn.Linear(last, hidden), nn.LeakyReLU()]
            last = hidden
        layers.append(nn.Linear(last, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PADModule(nn.Module):
    def __init__(self, dim, penalty_weight=0.1, reduction=8, radius=0.1):
        super().__init__()
        self.penalty_weight = penalty_weight
        self.residual = nn.Conv2d(2 * dim, dim, 1)
        self.fc = nn.Conv2d(2 * dim, 1, kernel_size=1)
        self.catconv1 = nn.Conv2d(2 * dim, dim, kernel_size=1)
        self.psc = _PSC(dim, reduction)
        self.temperature = 10
        self.radius = nn.Parameter(torch.tensor(radius))
        self.asf = _MLP(dim, dim, [dim, dim * 2])
        self.mix_conv = nn.Conv2d(2 * dim, dim, 1)

    def forward(self, x1, x2):
        # FFT ops run in fp32 regardless of the autocast context.
        with torch.autocast(device_type="cuda", enabled=False):
            x1, x2 = x1.float(), x2.float()
            h, w = x1.shape[-2:]
            x = torch.cat([x1, x2], dim=1)
            residual = self.residual(x)
            attention = torch.sigmoid(self.fc(x))
            x = self.catconv1(torch.cat([x1 * attention, x2 * (1 - attention)], dim=1)) + residual
            x = torch.fft.fftshift(torch.fft.rfft2(x, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
            mag = torch.abs(x)
            mag_residual = mag
            center = (x.shape[-2] // 2, x.shape[-1] // 2)
            yy, xx = torch.meshgrid(torch.arange(x.shape[-2], device=x.device),
                                    torch.arange(x.shape[-1], device=x.device), indexing="ij")
            dist = ((xx - center[1]) ** 2 + (yy - center[0]) ** 2).float().sqrt()
            high = torch.sigmoid((dist / dist.max() - torch.sigmoid(self.radius)) * self.temperature)
            b, c, hh, ww = mag.shape
            mag_high = self.asf((mag * high).view(b, c, -1).permute(0, 2, 1)) \
                .permute(0, 2, 1).reshape(b, c, hh, ww)
            mag = self.mix_conv(torch.cat([mag, mag_high], dim=1)) + mag_residual
            penalty = F.mse_loss(mag, mag_residual) * self.penalty_weight
            phase = self.psc(torch.angle(x))
            x = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
            x = torch.fft.irfft2(torch.fft.ifftshift(x, dim=(-2, -1)), s=(h, w),
                                 dim=(-2, -1), norm="ortho")
            return x + residual, penalty


@registry.MODEL.register("PADFusionR50")
class PADFusionR50(_DualSegBase):
    def __init__(self, config):
        super().__init__(config)
        self.en_sar = self._encoder(self.config.sar_in_channels)
        self.en_rgb = self._encoder(self.config.rgb_in_channels)
        self.fuse = nn.ModuleList(PADModule(c, self.config.pad_penalty_weight)
                                  for c in self.fuse_channels)

    def segment(self, sar, rgb, modes):
        sar, rgb = _zero_missing(sar, rgb, modes)
        fs, fr = self.en_sar(sar), self.en_rgb(rgb)
        fused, penalties = [], []
        for module, a, b in zip(self.fuse, fr, fs):  # PAD convention: (optical, SAR)
            out, penalty = module(a, b)
            fused.append(out)
            penalties.append(penalty)
        return self.decoder(fused), {"pad_penalty": torch.stack(penalties).sum()}

    def set_default_config(self):
        super().set_default_config()
        self.config.update(dict(pad_penalty_weight=0.1))


@registry.MODEL.register("ShaSpec")
class ShaSpec(_DualSegBase):
    def __init__(self, config):
        super().__init__(config)
        self.shared = self._encoder(self.config.sar_in_channels)
        # The shared trunk encodes both sensors; each sensor gets its own stem,
        # the optical one initialised from the ImageNet RGB stem.
        self.rgb_stem = copy.deepcopy(self._encoder(self.config.rgb_in_channels).resnet.conv1)
        self.spec_sar = self._encoder(self.config.sar_in_channels)
        self.spec_rgb = self._encoder(self.config.rgb_in_channels)
        self.proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(2 * c, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(True),
                          nn.Conv2d(c, c, 1))
            for c in self.fuse_channels)
        self.fuse = nn.ModuleList(
            nn.Sequential(nn.Conv2d(2 * c, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(True))
            for c in self.fuse_channels)
        # Domain classifier on specific features (deepest level).
        self.dco = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2048, 2))

    @contextmanager
    def _rgb_stem(self):
        original = self.shared.resnet.conv1
        self.shared.resnet.conv1 = self.rgb_stem
        try:
            yield
        finally:
            self.shared.resnet.conv1 = original

    def segment(self, sar, rgb, modes):
        sar, rgb = _zero_missing(sar, rgb, modes)  # a missing stream never reaches a loss
        r_sar = self.shared(sar)
        with self._rgb_stem():
            r_rgb = self.shared(rgb)
        s_sar, s_rgb = self.spec_sar(sar), self.spec_rgb(rgb)
        has_sar = (modes != SAR_MISSING).view(-1, 1, 1, 1)
        has_rgb = (modes != RGB_MISSING).view(-1, 1, 1, 1)
        fused = []
        for level, proj in enumerate(self.proj):
            f_sar = proj(torch.cat([r_sar[level], s_sar[level]], 1)) + r_sar[level]
            f_rgb = proj(torch.cat([r_rgb[level], s_rgb[level]], 1)) + r_rgb[level]
            # Missing-modality feature generation: mean shared feature of the
            # available modalities (N = 2, so the other modality's r).
            f_sar = torch.where(has_sar, f_sar, r_rgb[level])
            f_rgb = torch.where(has_rgb, f_rgb, r_sar[level])
            fused.append(self.fuse[level](torch.cat([f_sar, f_rgb], 1)))
        aux = {}
        if self.training:
            both = (modes == COMPLETE)
            # Distribution alignment (L1 between shared features), available pairs only.
            if bool(both.any()):
                aux["dao_loss"] = self.config.alpha * sum(
                    F.l1_loss(a[both], b[both]) for a, b in zip(r_sar, r_rgb)) / len(r_sar)
            # Domain classification on specific features of available modalities.
            logits = torch.cat([self.dco(s_sar[-1][has_sar.flatten()]),
                                self.dco(s_rgb[-1][has_rgb.flatten()])])
            target = torch.cat([torch.zeros(int(has_sar.sum()), dtype=torch.long),
                                torch.ones(int(has_rgb.sum()), dtype=torch.long)]).to(logits.device)
            aux["dco_loss"] = self.config.beta * F.cross_entropy(logits, target)
        return self.decoder(fused), aux

    def set_default_config(self):
        super().set_default_config()
        self.config.update(dict(alpha=0.1, beta=0.02))
