"""STARS (Wang et al., JAG 151:105378, 2026; arXiv 2601.17342) under the revision protocol.

Architecture: the authors' release https://github.com/CVEO/STARS (src/starsv2.py, the
model train.py imports; single commit 'init', 2026-02-09, adapted here under the matched protocol so all architectural and objective choices can be inspected).  Kept unchanged:
per-modality 7x7 stems, a shared ResNet-50 trunk and two specific ResNet-50 trunks
(ImageNet initialised, conv1/maxpool removed as in the release), translation modules
A2B/B2A at every level, SCSE residual fusion of specific + shared features, three FPN
decoders (SAR, optical, joint on the concatenated shared features).

Objective: the paper's equations, because the release does not run as published
(src/__init__.py imports missing modules; train.py reads result['sar_logits'] but the
model returns 'sar_loigts'; the contrastive loss indexes feature maps with
full-resolution label indices).  Paper vs release:
  L_seg  = mean of the three branch losses (paper Eq. 6; release sums them)
  L_ncs  = -1/2 [cos(A2B(f_sar), sg(f_rgb)) + cos(B2A(f_rgb), sg(f_sar))], stages 3 and 4
           (paper Eq. 4; the release uses MSE at all four stages)
  L_psc  = class-balanced pixel InfoNCE on shared features, stages 3 and 4, 32 pixels
           per class, tau = 0.07 (paper Eq. 5; tau and N from the release); labels are
           nearest-downsampled to the feature grid (fixes the release's indexing)
  L      = L_seg + 0.5 L_psc + 0.2 L_ncs   (alpha, beta from the paper's ablations)
Revision-protocol changes (as for every baseline): the shared segmentation loss
(HarmonizedSegLoss) replaces label-smoothed cross-entropy in each branch; optimiser,
schedule, data and checkpoint selection are the shared protocol's.

Inference (the release defines single-modality outputs only in eval mode):
  infer_mode 'fuse' -> joint decoder on [shared_sar, shared_rgb] (trained on complete
                       pairs, the only both-modality output the model has)
  infer_mode 'sar'  -> SAR decoder (optical missing);  'rgb' -> optical decoder.
STARS trains every branch on every (complete) sample; it uses no modality dropout.
"""
from __future__ import annotations

import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from ever import registry
from ever.interface import ERModule
from torchvision.models import resnet50

from module.baseline.revision_baselines import HarmonizedSegLoss, _DualSegBase

# Same ImageNet weights (and local hub cache) as every other ResNet-50 in this repo (base_resnet/_resnets.py).
IMAGENET_R50 = "https://download.pytorch.org/models/resnet50-19c8e357.pth"


# ----------------------------------------------------------------- release modules
class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                   nn.GroupNorm(32, out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=True)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x, skip):
        return F.interpolate(x, scale_factor=2.0, mode="nearest") + self.skip_conv(skip)


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()
        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]
        for _ in range(1, n_upsamples):
            blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class FPNDecoder(nn.Module):
    def __init__(self, encoder_channels: List[int], num_class: int, pyramid_channels=256,
                 segmentation_channels=128, dropout=0.2):
        super().__init__()
        c = encoder_channels[::-1]
        self.p5 = nn.Conv2d(c[0], pyramid_channels, kernel_size=1)
        self.p4 = FPNBlock(pyramid_channels, c[1])
        self.p3 = FPNBlock(pyramid_channels, c[2])
        self.p2 = FPNBlock(pyramid_channels, c[3])
        self.seg_blocks = nn.ModuleList(SegmentationBlock(pyramid_channels, segmentation_channels, n)
                                        for n in (3, 2, 1, 0))
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)  # defined but unused in the release
        self.classifier = nn.Conv2d(segmentation_channels, num_class, kernel_size=1)

    def forward(self, features, H, W):
        c2, c3, c4, c5 = features
        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)
        x = sum(b(p) for b, p in zip(self.seg_blocks, (p5, p4, p3, p2)))
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return self.classifier(x)


class TranslationModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.InstanceNorm2d(channels),
                                 nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        return self.net(x)


class ResnetTrunk(nn.Module):
    """ResNet-50 layer1-4 without conv1/bn1/maxpool (release: ResnetWarrpper, del_first_conv)."""

    def __init__(self, resnet):
        super().__init__()
        resnet = copy.deepcopy(resnet)
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

    def forward(self, x):
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        return x2, x3, x4, self.layer4(x4)


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, ks, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        nn.init.kaiming_normal_(self.conv.weight, a=1)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, in_channels // reduction, 1),
                                 nn.ReLU(inplace=True), nn.Conv2d(in_channels // reduction, in_channels, 1),
                                 nn.Sigmoid())
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class FeatureFusionModuleSCSE_res(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.scse_1 = SCSEModule(in_chan)
        self.scse_2 = SCSEModule(in_chan)
        self.convblk = ConvBNReLU(in_chan * 2, out_chan, ks=1, stride=1, padding=0)
        self.scse = SCSEModule(out_chan)

    def forward(self, fsp, fcp):
        fsp = self.scse_1(fsp)
        feat = self.scse(self.convblk(torch.cat([fsp, self.scse_2(fcp)], dim=1))) + fsp
        return torch.cat([feat, fcp], dim=1)


# ----------------------------------------------------------------- losses (paper)
@torch.no_grad()
def _sample_pixels(labels, per_class=32, ignore_index=-1):
    flat = labels.reshape(-1)
    idx, lab = [], []
    for cls in torch.unique(flat):
        if cls == ignore_index:
            continue
        where = (flat == cls).nonzero(as_tuple=True)[0]
        pick = where[torch.randperm(where.numel(), device=labels.device)[:per_class]]
        idx.append(pick)
        lab.append(flat[pick])
    if not idx:
        return None, None
    return torch.cat(idx), torch.cat(lab)


def psc_loss(f1, f2, labels, tau=0.07, per_class=32):
    """Pixel-level sampled cross-modal InfoNCE (paper Eq. 5), labels at the feature grid."""
    B, C, h, w = f1.shape
    lab = F.interpolate(labels[:, None].float(), size=(h, w), mode="nearest")[:, 0].long()
    indices, sampled = _sample_pixels(lab, per_class)
    if indices is None:
        return f1.sum() * 0
    a = F.normalize(f1.float().permute(0, 2, 3, 1).reshape(-1, C)[indices], dim=1)
    b = F.normalize(f2.float().permute(0, 2, 3, 1).reshape(-1, C)[indices], dim=1)
    logits = a @ b.T / tau
    pos = (sampled[None] == sampled[:, None]).float()
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return (-(pos * log_prob).sum(1) / pos.sum(1)).mean()


def ncs_loss(p, f):
    """Negative cosine similarity with stop-gradient on the target (paper Eq. 4, one direction)."""
    return -F.cosine_similarity(p.float(), f.detach().float(), dim=1).mean()


# ----------------------------------------------------------------- model
@registry.MODEL.register("STARS")
class STARS(ERModule):
    CHANNELS = (256, 512, 1024, 2048)

    def __init__(self, config):
        super().__init__(config)
        self.infer_mode = self.config.infer_mode
        res = resnet50()
        if self.config.pretrained:
            res.load_state_dict(torch.hub.load_state_dict_from_url(IMAGENET_R50, map_location="cpu", progress=False))
        self.shared_encoder, self.spec_sar, self.spec_rgb = ResnetTrunk(res), ResnetTrunk(res), ResnetTrunk(res)
        self.sar_stem = nn.Sequential(nn.Conv2d(self.config.sar_in_channels, 64, 7, 2, 3, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.rgb_stem = nn.Sequential(nn.Conv2d(self.config.rgb_in_channels, 64, 7, 2, 3, bias=False),
                                      nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        ch = self.CHANNELS
        self.A2B = nn.ModuleList(TranslationModule(c) for c in ch)
        self.B2A = nn.ModuleList(TranslationModule(c) for c in ch)
        self.fuse_sar = nn.ModuleList(FeatureFusionModuleSCSE_res(c, c) for c in ch)
        self.fuse_rgb = nn.ModuleList(FeatureFusionModuleSCSE_res(c, c) for c in ch)
        double = [2 * c for c in ch]
        k = self.config.num_classes
        self.decoder_sar = FPNDecoder(double, k)
        self.decoder_rgb = FPNDecoder(double, k)
        self.decoder_joint = FPNDecoder(double, k)
        self.seg_loss = HarmonizedSegLoss(self.config.loss.harmonized, k)
        # Training degradations (configs/revision/baselines.py RECIPES); off unless config.degrade.
        _init_degrade(self)

    def _sar(self, x):
        h = self.sar_stem(x)
        return self.shared_encoder(h), self.spec_sar(h)

    def _rgb(self, x):
        h = self.rgb_stem(x)
        return self.shared_encoder(h), self.spec_rgb(h)

    def forward(self, x, y=None):
        s = self.config.sar_in_channels
        sar, rgb = x[:, :s], x[:, s:s + self.config.rgb_in_channels]
        H, W = x.shape[-2:]
        if not self.training:
            if self.infer_mode == "sar":
                sh, sp = self._sar(sar)
                logits = self.decoder_sar([m(a, b) for m, a, b in zip(self.fuse_sar, sp, sh)], H, W)
            elif self.infer_mode == "rgb":
                sh, sp = self._rgb(rgb)
                logits = self.decoder_rgb([m(a, b) for m, a, b in zip(self.fuse_rgb, sp, sh)], H, W)
            else:
                sh_s, _ = self._sar(sar)
                sh_r, _ = self._rgb(rgb)
                logits = self.decoder_joint([torch.cat([a, b], 1) for a, b in zip(sh_s, sh_r)], H, W)
            return F.softmax(logits.float(), dim=1)

        sar, rgb = _degrade(self, sar, rgb)
        target = y["cls"].long()
        sh_s, sp_s = self._sar(sar)
        sh_r, sp_r = self._rgb(rgb)
        logit_s = self.decoder_sar([m(a, b) for m, a, b in zip(self.fuse_sar, sp_s, sh_s)], H, W)
        logit_r = self.decoder_rgb([m(a, b) for m, a, b in zip(self.fuse_rgb, sp_r, sh_r)], H, W)
        logit_j = self.decoder_joint([torch.cat([a, b], 1) for a, b in zip(sh_s, sh_r)], H, W)
        losses = {}
        for name, logit in (("sar", logit_s), ("rgb", logit_r), ("fuse", logit_j)):
            for key, value in self.seg_loss(logit, target, name).items():
                losses[key] = value / 3.0  # paper Eq. 6: mean over the three branches
        deep = (2, 3)  # stages 3 and 4
        losses["ncs_loss"] = self.config.beta * sum(
            0.5 * (ncs_loss(self.A2B[i](sh_s[i]), sh_r[i]) + ncs_loss(self.B2A[i](sh_r[i]), sh_s[i]))
            for i in deep) / len(deep)
        losses["psc_loss"] = self.config.alpha * sum(psc_loss(sh_s[i], sh_r[i], target) for i in deep) / len(deep)
        losses["mem"] = torch.tensor(float(torch.cuda.max_memory_allocated() // 2 ** 20), device=x.device)
        return losses

    def custom_param_groups(self):
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            if p.requires_grad:
                (no_decay if p.ndim <= 1 else decay).append(p)
        return [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]

    def set_default_config(self):
        self.config.update(dict(
            num_classes=7, sar_in_channels=1, rgb_in_channels=3, infer_mode="fuse", pretrained=True,
            alpha=0.5, beta=0.2, loss=dict(harmonized=dict()),
        ))


# ----------------------------------------------------------------- degradation hook
def _init_degrade(module):
    """Same fields as revision_baselines._DualSegBase (training degradations)."""
    from module.reliability import phys_degrade as phys
    from module.reliability import train_degrade as tdeg
    deg = dict(module.config.get("degrade", None) or {})
    module.deg_prob = float(deg.get("prob", 0.0))
    module.deg_sar_prob = float(deg.get("sar_prob", 0.0))
    module.deg_types = tuple(deg.get("types", tdeg.OPTICAL_TYPES))
    module.deg_optical = str(deg.get("optical", "generic"))
    module.deg_sar = str(deg.get("sar", "generic"))
    module.deg_coreg_prob = float(deg.get("coreg_prob", 0.0))
    module.deg_coreg_max = float(deg.get("coreg_max", 12.0))
    module.deg_unit = float(deg.get("unit", 1.0))
    module.deg_max_dn = float(deg.get("max_dn", 255.0))
    module.deg_sar_domain = str(deg.get("sar_domain", "amp"))
    module.deg_gsd = float(deg.get("gsd", 5.0))
    module.deg_ablate = tuple(deg.get("ablate", ()))
    module.deg_optical_types = tuple(deg.get("optical_types", phys.OPTICAL_TYPES))
    module.deg_coverage = deg.get("coverage", "uniform")
    module.want_rel = False
    module._rel_target = module._rel_valid = None
    for name in ("rgb_mean", "rgb_std", "sar_mean", "sar_std"):
        module.register_buffer(name, torch.tensor(list(deg.get(name, [0.0])), dtype=torch.float32),
                               persistent=False)


def _degrade(module, sar, rgb):
    from module.reliability import phys_degrade as phys
    from module.reliability import train_degrade as tdeg
    if module.deg_prob > 0:
        rgb = _DualSegBase._degrade_rgb(module, rgb)
    if module.deg_sar_prob > 0:
        if module.deg_sar == "phys":
            sar = phys.random_sar(sar, module.sar_mean, module.sar_std, module.deg_sar_prob, module.deg_sar_domain)
        else:
            sar = tdeg.degrade_sar(sar, module.sar_mean, module.sar_std, module.deg_sar_prob,
                                   domain=module.deg_sar_domain)
    if module.deg_coreg_prob > 0:
        sar = phys.random_coreg(sar, module.deg_coreg_prob, module.deg_coreg_max)
    return sar, rgb
