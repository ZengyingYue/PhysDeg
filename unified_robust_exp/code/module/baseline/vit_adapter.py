"""ViT-Adapter (Chen et al., ICLR 2023) for the TransOSS shared meta encoder.

The plain TransOSS ViT-B/16 is single-resolution (stride 16) and was
contrastively pretrained for *global* ship re-identification, so its patch
tokens carry weak spatial-semantic structure — exactly the wrong starting
point for dense land-cover segmentation (root causes R4 + R5).  ViT-Adapter
fixes this without touching the pretrained ViT weights: a parallel
convolutional branch supplies genuine multi-scale spatial priors and two
cross-attention bridges (Injector / Extractor) exchange information with the
frozen-ish transformer trunk, yielding a real {1/4, 1/8, 1/16, 1/32} feature
pyramid.

This is a pure-PyTorch implementation using standard multi-head cross-attention
(the official paper uses multi-scale deformable attention, MSDeformAttn, which
needs a compiled CUDA op — a drop-in efficiency upgrade we deliberately defer).

Dual-modality note: the Spatial Prior Module uses a modality-specific input
stem (`stem_sar` 1-ch / `stem_rgb` 3-ch) feeding a shared body, mirroring
TransOSS's dual-tokenizer / shared-trunk design.

Key knobs:
  embed_dim     768 (ViT-B)
  spm_dim       64  (SPM stem width)
  n_heads       8   (cross-attention heads)
  out_channels  (128,256,512,1024) — match ConvNeXt-base / CoDER fuse widths
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spatial Prior Module — parallel conv branch giving 1/4,1/8,1/16,1/32 maps
# ---------------------------------------------------------------------------
class SpatialPriorModule(nn.Module):
    def __init__(self, sar_in: int, rgb_in: int, spm_dim: int, embed_dim: int):
        super().__init__()

        def stem(in_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, spm_dim, 3, 2, 1, bias=False),
                nn.BatchNorm2d(spm_dim), nn.ReLU(inplace=True),
                nn.Conv2d(spm_dim, spm_dim, 3, 1, 1, bias=False),
                nn.BatchNorm2d(spm_dim), nn.ReLU(inplace=True),
                nn.Conv2d(spm_dim, spm_dim, 3, 1, 1, bias=False),
                nn.BatchNorm2d(spm_dim), nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1),                       # -> 1/4
            )

        self.stem_sar = stem(sar_in)
        self.stem_rgb = stem(rgb_in)
        # shared body 1/4 -> 1/8 -> 1/16 -> 1/32
        self.conv2 = nn.Sequential(
            nn.Conv2d(spm_dim, 2 * spm_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(2 * spm_dim), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * spm_dim, 4 * spm_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(4 * spm_dim), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * spm_dim, 4 * spm_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(4 * spm_dim), nn.ReLU(inplace=True))
        # project every level to the transformer width D
        self.fc1 = nn.Conv2d(spm_dim, embed_dim, 1)          # 1/4
        self.fc2 = nn.Conv2d(2 * spm_dim, embed_dim, 1)      # 1/8
        self.fc3 = nn.Conv2d(4 * spm_dim, embed_dim, 1)      # 1/16
        self.fc4 = nn.Conv2d(4 * spm_dim, embed_dim, 1)      # 1/32

    def forward(self, x: torch.Tensor, modality: str):
        s = self.stem_sar(x) if modality == "sar" else self.stem_rgb(x)  # 1/4
        c2 = self.conv2(s)    # 1/8
        c3 = self.conv3(c2)   # 1/16
        c4 = self.conv4(c3)   # 1/32
        c1 = self.fc1(s)      # 1/4,  D
        c2 = self.fc2(c2)     # 1/8,  D
        c3 = self.fc3(c3)     # 1/16, D
        c4 = self.fc4(c4)     # 1/32, D
        return c1, c2, c3, c4


# ---------------------------------------------------------------------------
# Standard multi-head cross-attention
# ---------------------------------------------------------------------------
class CrossAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, q_tok: torch.Tensor, kv_tok: torch.Tensor) -> torch.Tensor:
        B, Nq, C = q_tok.shape
        Nk = kv_tok.shape[1]
        q = self.q(q_tok).reshape(B, Nq, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(kv_tok).reshape(B, Nk, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        out = F.scaled_dot_product_attention(q, k, v)        # [B, h, Nq, hd]
        out = out.transpose(1, 2).reshape(B, Nq, C)
        return self.proj(out)


# ---------------------------------------------------------------------------
# ConvFFN — depthwise-conv FFN over the (split) spatial token maps
# ---------------------------------------------------------------------------
class ConvFFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.dw = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, c: torch.Tensor, shapes: Sequence[Tuple[int, int]]) -> torch.Tensor:
        # c: [B, N_c, D] = concat of three level token blocks; shapes lists (H,W) per level.
        B, _, _ = c.shape
        x = self.fc1(c)                                       # [B, N_c, hidden]
        # depthwise conv per level then re-concat
        outs = []
        off = 0
        for (h, w) in shapes:
            n = h * w
            seg = x[:, off:off + n]                           # [B, n, hidden]
            seg = seg.transpose(1, 2).reshape(B, -1, h, w)    # [B, hidden, h, w]
            seg = self.dw(seg)
            seg = seg.flatten(2).transpose(1, 2)              # [B, n, hidden]
            outs.append(seg)
            off += n
        x = torch.cat(outs, dim=1)
        x = self.act(x)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Injector / Extractor
# ---------------------------------------------------------------------------
class Injector(nn.Module):
    """ViT patch tokens (query) absorb spatial priors from `c` (key/value)."""

    def __init__(self, dim: int, n_heads: int = 8, gamma_init: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6)
        self.attn = CrossAttention(dim, n_heads)
        # per-channel residual gate, init 0 → identity at start (protects pretrained ViT)
        self.gamma = nn.Parameter(gamma_init * torch.ones(dim))

    def forward(self, x_patch: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return x_patch + self.gamma * self.attn(self.norm_q(x_patch), self.norm_kv(c))


class Extractor(nn.Module):
    """Spatial tokens `c` (query) pull refined info from ViT patch tokens, then
    a depthwise ConvFFN sharpens local structure."""

    def __init__(self, dim: int, n_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6)
        self.attn = CrossAttention(dim, n_heads)
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = ConvFFN(dim, mlp_ratio)

    def forward(self, c, x_patch, shapes):
        c = c + self.attn(self.norm_q(c), self.norm_kv(x_patch))
        c = c + self.ffn(self.norm_ffn(c), shapes)
        return c


# ---------------------------------------------------------------------------
# Full adapter — orchestrates SPM + 4 interaction stages + pyramid build
# ---------------------------------------------------------------------------
class ViTAdapter(nn.Module):
    """Wraps a list of ViT blocks with the adapter interaction machinery and
    emits a {stride 4,8,16,32} pyramid with the requested channel widths.

    The transformer blocks themselves (and patch_embed/pos/cls/mie) live on the
    parent TransOSSEncoder; this module only owns the *new* adapter params and
    the forward orchestration, which the encoder calls.
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        sar_in: int,
        rgb_in: int,
        out_channels: Sequence[int],
        spm_dim: int = 64,
        n_heads: int = 8,
        n_stages: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        assert depth % n_stages == 0, "ViT depth must split evenly into stages"
        self.depth = depth
        self.n_stages = n_stages
        self.blocks_per_stage = depth // n_stages

        self.spm = SpatialPriorModule(sar_in, rgb_in, spm_dim, embed_dim)
        self.injectors = nn.ModuleList(
            [Injector(embed_dim, n_heads) for _ in range(n_stages)]
        )
        self.extractors = nn.ModuleList(
            [Extractor(embed_dim, n_heads, mlp_ratio) for _ in range(n_stages)]
        )
        # output projections to FPN channel widths, strides (4,8,16,32)
        self.out_norm = nn.ModuleList([nn.LayerNorm(embed_dim, eps=1e-6) for _ in range(4)])
        self.out_proj = nn.ModuleList(
            [nn.Conv2d(embed_dim, oc, 1) for oc in out_channels]
        )
        self.embed_dim = embed_dim

    @staticmethod
    def _flatten(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, D, H, W = x.shape
        return x.flatten(2).transpose(1, 2), (H, W)          # [B, HW, D], (H,W)

    @staticmethod
    def _unflatten(tok: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        B, N, D = tok.shape
        H, W = hw
        return tok.transpose(1, 2).reshape(B, D, H, W)

    def run(
        self,
        x_img: torch.Tensor,
        modality: str,
        x_tokens: torch.Tensor,          # [B, 1+Np, D]  (cls + patch), post pos/mie
        patch_hw: Tuple[int, int],
        blocks: nn.ModuleList,
    ) -> List[torch.Tensor]:
        B = x_img.shape[0]
        # --- spatial priors ---
        c1, c2, c3, c4 = self.spm(x_img, modality)           # maps at 1/4,1/8,1/16,1/32
        t2, s2 = self._flatten(c2)
        t3, s3 = self._flatten(c3)
        t4, s4 = self._flatten(c4)
        c = torch.cat([t2, t3, t4], dim=1)                   # spatial tokens
        shapes = [s2, s3, s4]

        cls = x_tokens[:, :1]
        x_patch = x_tokens[:, 1:]                            # [B, Np, D]

        # --- 4 interaction stages ---
        for st in range(self.n_stages):
            x_patch = self.injectors[st](x_patch, c)
            x = torch.cat([cls, x_patch], dim=1)
            for b in range(self.blocks_per_stage):
                x = blocks[st * self.blocks_per_stage + b](x)
            cls, x_patch = x[:, :1], x[:, 1:]
            c = self.extractors[st](c, x_patch, shapes)

        # --- split spatial tokens back to maps ---
        n2, n3 = s2[0] * s2[1], s3[0] * s3[1]
        f2 = self._unflatten(c[:, :n2], s2)                  # 1/8
        f3 = self._unflatten(c[:, n2:n2 + n3], s3)           # 1/16
        f4 = self._unflatten(c[:, n2 + n3:], s4)             # 1/32
        # fuse the ViT trunk output (1/16) into the 1/16 level
        x_map = self._unflatten(x_patch, patch_hw)           # 1/16 (stride 16)
        if x_map.shape[-2:] != f3.shape[-2:]:
            x_map = F.interpolate(x_map, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        f3 = f3 + x_map
        # 1/4 level: SPM c1 + upsampled 1/8
        f1 = c1 + F.interpolate(f2, size=c1.shape[-2:], mode="bilinear", align_corners=False)

        feats = [f1, f2, f3, f4]
        # channel-norm (LayerNorm over D) then project to FPN widths
        out = []
        for j, fmap in enumerate(feats):
            t, hw = self._flatten(fmap)
            t = self.out_norm[j](t)
            fmap = self._unflatten(t, hw)
            out.append(self.out_proj[j](fmap))
        return out
