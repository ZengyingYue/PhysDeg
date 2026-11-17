"""TransOSS ViT-B/16 encoder for CoDER's shared meta path (`en_content`).

TransOSS (Wang et al. ICCV 2025, *Cross-modal Ship Re-Identification via
Optical and SAR Imagery*) is a single ViT-B/16 backbone with **two patch
embeddings** — `patch_embed` for RGB and `patch_embed_SAR` for SAR — plus a
2-entry modality identity embedding `mie_embed[modality]` that is added to
every patch token.  This is exactly the structure CoDER's shared meta
encoder needs: one set of transformer blocks consumed by both modalities,
where modality identity is the only forwarded difference.

Multi-scale outputs (ViTDet recipe, Li et al. ECCV 2022):
  • All 12 blocks live at the patch grid resolution (stride 16).
  • We tap features at blocks {3, 6, 9, 12} and reshape to (B, D, H/16, W/16).
  • Per tap a lightweight projection sends them to FPN-compatible
    {stride, channels} matching ConvNeXt-base outputs: (4,128) / (8,256) /
    (16,512) / (32,1024).  Strides 4 / 8 are produced by stride-2 ConvTranspose,
    stride 32 by stride-2 Conv.

Pretrained weights (`vit_b512_pre.pth`):
  • The local checkpoint was trained on ship images at 128 patch tokens
    (likely a 128×256 input → 8×16 patch grid).  We always 2D-interpolate the
    pos_embed to whatever grid `img_size / patch_size` gives at runtime.
  • `patch_embed_SAR.proj.weight` is (768, 3, 16, 16) — TransOSS encodes SAR
    as 3-channel (VV / VH / log-ratio in the paper).  EarthMiss SAR is single
    channel; we adapt the pretrained stem by averaging across the input-channel
    dim, matching what `timm.create_model(in_chans=1, pretrained=True)` does.
"""
from __future__ import annotations

import logging
import math
import os
import re
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_PRETRAINED_PATH = "/mnt/e/experiment/models/TransOSS/vit_b512_pre.pth"
_log = logging.getLogger(__name__)


# ----- ViT building blocks -------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)  # [B, D, H/p, W/p]


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each [B, h, N, hd]
        # Prefer the fused SDPA kernel — gives us flash-attention on Blackwell.
        out = F.scaled_dot_product_attention(q, k, v)  # [B, h, N, hd]
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ----- Multi-scale tap projector -------------------------------------------
class _Tap(nn.Module):
    """Project a stride-16 ViT feature to a target stride / channels via
    a single up- or down-sampling step plus a 1×1 projection.

    Matches the ViTDet/SimpleFPN scheme: one transposed-conv2d per 2× upsample
    (so two for 4×), one stride-2 conv2d per 2× downsample.
    """

    def __init__(self, embed_dim: int, out_channels: int, src_stride: int, dst_stride: int):
        super().__init__()
        ratio = src_stride // dst_stride if dst_stride <= src_stride else 0
        down_ratio = dst_stride // src_stride if dst_stride > src_stride else 0

        layers: List[nn.Module] = []
        cur = embed_dim
        if ratio == 4:
            layers += [
                nn.ConvTranspose2d(cur, embed_dim // 2, 2, stride=2),
                nn.GroupNorm(32, embed_dim // 2),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, 2, stride=2),
            ]
            cur = embed_dim // 4
        elif ratio == 2:
            layers += [nn.ConvTranspose2d(cur, embed_dim // 2, 2, stride=2)]
            cur = embed_dim // 2
        elif ratio == 1:
            pass  # same stride
        elif down_ratio == 2:
            layers += [nn.Conv2d(cur, embed_dim, kernel_size=3, stride=2, padding=1)]
        else:
            raise ValueError(
                f"_Tap unsupported stride change src={src_stride} → dst={dst_stride}"
            )
        # Final 1×1 projection to FPN channels.
        layers += [nn.Conv2d(cur, out_channels, kernel_size=1)]
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ----- TransOSS encoder ----------------------------------------------------
class TransOSSEncoder(nn.Module):
    """ViT-B/16 with dual patch embeds (SAR/RGB) + modality identity embed.

    Interface mirrors `ConvNeXtEncoder`:
        forward(x) -> (features_list, whitening_list)
    where `features_list` is a 4-level FPN-compatible pyramid at strides
    (4, 8, 16, 32) with channels matching ConvNeXt-base (128, 256, 512, 1024)
    by default.  Whitening list stays empty (no IBN in ViT-B/16).

    Modality routing: by convention CoDER calls the same `en_content` for
    both SAR and RGB inputs.  We pick the patch embed based on the input
    channel count, falling back to an attribute toggle for unambiguous control:

        encoder.set_modality('sar' | 'rgb')   # sticky for the next call
        encoder(x_1ch)                         # auto-detects sar (1 channel)
        encoder(x_3ch)                         # auto-detects rgb (3 channels)
    """

    EMBED_DIM = 768
    NUM_HEADS = 12
    DEPTH = 12

    def __init__(
        self,
        sar_in_channels: int = 1,
        rgb_in_channels: int = 3,
        patch_size: int = 16,
        img_size: int = 512,
        tap_blocks: Sequence[int] = (3, 6, 9, 12),
        fpn_channels: Sequence[int] = (128, 256, 512, 1024),
        fpn_strides: Sequence[int] = (4, 8, 16, 32),
        pretrained: bool = True,
        pretrained_path: str = DEFAULT_PRETRAINED_PATH,
        drop_pretrained_cls: bool = True,
        tap_norm: bool = True,
        use_adapter: bool = False,
        adapter_spm_dim: int = 64,
        adapter_heads: int = 8,
    ):
        super().__init__()
        self.sar_in_channels = sar_in_channels
        self.rgb_in_channels = rgb_in_channels
        self.patch_size = patch_size
        self.default_img_size = img_size
        self.tap_blocks = tuple(int(b) for b in tap_blocks)
        assert len(self.tap_blocks) == len(fpn_channels) == len(fpn_strides) == 4
        # block index → tap position (FPN level) lookup.
        self._tap_index = {b: j for j, b in enumerate(self.tap_blocks)}

        # Patch embeds: SAR and RGB.
        self.patch_embed_SAR = PatchEmbed(sar_in_channels, self.EMBED_DIM, patch_size)
        self.patch_embed = PatchEmbed(rgb_in_channels, self.EMBED_DIM, patch_size)
        # CLS token (kept for state-dict compat with TransOSS; not used by FPN).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.EMBED_DIM))
        # 2D pos_embed for patch tokens at the *current* grid; we re-interpolate
        # at runtime so the buffer is built lazily.  Stored as a parameter so it
        # gets included in checkpoints and can be tuned.
        n_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, self.EMBED_DIM))
        # Modality identity embed: (2, 1, D) — index 0 = RGB, 1 = SAR (TransOSS).
        self.mie_embed = nn.Parameter(torch.zeros(2, 1, self.EMBED_DIM))

        # Transformer blocks.
        self.blocks = nn.ModuleList(
            [Block(self.EMBED_DIM, self.NUM_HEADS, mlp_ratio=4.0) for _ in range(self.DEPTH)]
        )
        self.norm = nn.LayerNorm(self.EMBED_DIM, eps=1e-6)

        # Multi-scale extraction mode.  Two options:
        #   use_adapter=False → ViTDet SimpleFPN ("fake" pyramid): tap 4 blocks
        #     and up/down-sample a single stride-16 representation.
        #   use_adapter=True  → ViT-Adapter (Chen et al. ICLR 2023): a parallel
        #     conv branch + Injector/Extractor cross-attention give a *genuine*
        #     {4,8,16,32} pyramid (fixes R4 single-resolution + R5 weak-spatial).
        self.use_adapter = bool(use_adapter)
        self.fpn_channels = tuple(int(c) for c in fpn_channels)
        self.fpn_strides = tuple(int(s) for s in fpn_strides)

        if self.use_adapter:
            from module.baseline.vit_adapter import ViTAdapter
            self.adapter = ViTAdapter(
                embed_dim=self.EMBED_DIM,
                depth=self.DEPTH,
                sar_in=sar_in_channels,
                rgb_in=rgb_in_channels,
                out_channels=self.fpn_channels,
                spm_dim=adapter_spm_dim,
                n_heads=adapter_heads,
                n_stages=4,
            )
            self.tap_norm = False
        else:
            # Per-tap LayerNorm (ViTDet recipe): the residual stream grows ~7× in
            # magnitude from block 3 → 12 (std 0.8 → 1.9, absmax 6 → 42), so each
            # tapped feature must be normalised over the channel dim before its
            # up/down-sampling projector — otherwise the four FPN levels arrive at
            # wildly different scales and the deepest level dominates the pyramid.
            self.tap_norm = bool(tap_norm)
            if self.tap_norm:
                self.tap_norms = nn.ModuleList(
                    [nn.LayerNorm(self.EMBED_DIM, eps=1e-6) for _ in self.tap_blocks]
                )
            # Multi-scale tap projectors (ViTDet SimpleFPN).
            self.taps = nn.ModuleList(
                [
                    _Tap(self.EMBED_DIM, ch, src_stride=patch_size, dst_stride=s)
                    for ch, s in zip(fpn_channels, fpn_strides)
                ]
            )

        # Sticky modality override; auto-detect by channel count when None.
        self._sticky_modality: str | None = None

        if pretrained:
            self._load_pretrained(pretrained_path, drop_cls=drop_pretrained_cls)

    # ----- pretrained loading ---------------------------------------------
    @staticmethod
    def _strip_prefix(sd: dict, prefix: str) -> dict:
        out = {}
        for k, v in sd.items():
            out[k[len(prefix):] if k.startswith(prefix) else k] = v
        return out

    def _load_pretrained(self, path: str, drop_cls: bool = True) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"TransOSS pretrained weights not found at {path}.  Place "
                "`vit_b512_pre.pth` from the TransOSS release there or pass "
                "pretrained_path=... explicitly."
            )
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        # Strip a possible DDP "module." prefix.
        sd = self._strip_prefix(sd, "module.")
        # The HOSS_TransOSS.pth variant uses "base.<X>" nesting.
        sd = self._strip_prefix(sd, "base.")

        # Drop classification head (`fc.*`) and the auxiliary norm we don't use
        # specially (already in our state).  Drop the bottleneck if present.
        sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
        sd = {k: v for k, v in sd.items() if not k.startswith("bottleneck.")}

        # Adapt patch_embed_SAR from 3-channel pretrained → sar_in_channels.
        sd = self._adapt_patch_embed(sd, "patch_embed_SAR", self.sar_in_channels)
        sd = self._adapt_patch_embed(sd, "patch_embed", self.rgb_in_channels)

        # Interpolate pos_embed from pretrained grid to ours.
        if "pos_embed" in sd:
            sd["pos_embed"] = self._interpolate_pos_embed(sd["pos_embed"], self.pos_embed.shape)

        missing, unexpected = self.load_state_dict(sd, strict=False)
        # taps.* / tap_norms.* / adapter.* are randomly init — fine to be missing.
        _new_prefixes = ("taps.", "tap_norms.", "adapter.")
        missing_real = [
            k for k in missing if not k.startswith(_new_prefixes)
        ]
        if missing_real:
            _log.warning("TransOSS load: missing %d non-tap params (e.g. %s)",
                         len(missing_real), missing_real[:5])
        if unexpected:
            _log.warning("TransOSS load: unexpected %d params (e.g. %s)",
                         len(unexpected), unexpected[:5])

    @staticmethod
    def _adapt_patch_embed(sd: dict, prefix: str, target_in_chans: int) -> dict:
        key = f"{prefix}.proj.weight"
        if key not in sd:
            return sd
        w = sd[key]
        cur_in = w.shape[1]
        if cur_in == target_in_chans:
            return sd
        # Average across input channels, then broadcast (timm convention).
        w_avg = w.mean(dim=1, keepdim=True)
        if target_in_chans == 1:
            new_w = w_avg
        else:
            new_w = w_avg.expand(-1, target_in_chans, -1, -1).contiguous()
        sd = dict(sd)
        sd[key] = new_w
        return sd

    @staticmethod
    def _interpolate_pos_embed(src: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Resample 1D-tokenized pos_embed from a non-square source grid to a
        square target grid via 2D bilinear interpolation.

        src: [1, N_src+1, D]   N_src likely 128 (TransOSS) or whatever was
                               saved.  Includes 1 cls token at index 0.
        target_shape: [1, N_dst+1, D]   N_dst is square at runtime.
        """
        _, N_dst_plus1, D = target_shape
        N_dst = N_dst_plus1 - 1
        N_src = src.shape[1] - 1
        if N_src == N_dst:
            return src
        cls_tok = src[:, :1]            # [1, 1, D]
        patch = src[:, 1:]              # [1, N_src, D]

        # Try to factor the source token count into a (h_src, w_src) grid.
        # 128 → 8×16 (TransOSS ship aspect); fall back to closest square root.
        h_src, w_src = TransOSSEncoder._infer_grid(N_src)
        # Target grid is the square at our runtime resolution.
        h_dst = int(round(math.sqrt(N_dst)))
        # If we changed input size aspect ratio later this needs revising —
        # for EarthMiss 512×512 the dst grid is always square.
        w_dst = N_dst // h_dst
        assert h_dst * w_dst == N_dst, f"target N_dst={N_dst} not square"

        patch_2d = patch.transpose(1, 2).reshape(1, D, h_src, w_src)
        patch_resized = F.interpolate(
            patch_2d, size=(h_dst, w_dst), mode="bilinear", align_corners=False
        )
        patch_out = patch_resized.reshape(1, D, h_dst * w_dst).transpose(1, 2)
        return torch.cat([cls_tok, patch_out], dim=1)

    @staticmethod
    def _infer_grid(n: int) -> tuple[int, int]:
        """Closest (h, w) factorisation; if not factorisable use ceil-sqrt."""
        h = int(round(math.sqrt(n)))
        while h > 1 and n % h != 0:
            h -= 1
        if h == 0 or h == 1:
            # Truly irreducible — fall back to assuming a 1×N row (won't get
            # us perfect 2D structure, but the bilinear interp still works).
            return 1, n
        return h, n // h

    # ----- forward --------------------------------------------------------
    def set_modality(self, mode: str | None) -> None:
        if mode is not None and mode not in {"sar", "rgb"}:
            raise ValueError(f"set_modality expects 'sar'/'rgb'/None, got {mode}")
        self._sticky_modality = mode

    def _resolve_modality(self, x: torch.Tensor) -> str:
        if self._sticky_modality is not None:
            return self._sticky_modality
        C = x.shape[1]
        if C == self.sar_in_channels and self.sar_in_channels != self.rgb_in_channels:
            return "sar"
        if C == self.rgb_in_channels:
            return "rgb"
        raise ValueError(
            f"Cannot auto-detect modality from input channels {C}; "
            f"sar_in_channels={self.sar_in_channels}, "
            f"rgb_in_channels={self.rgb_in_channels}.  Call "
            "`encoder.set_modality('sar'|'rgb')` to disambiguate."
        )

    def _get_pos_embed(self, h_grid: int, w_grid: int) -> torch.Tensor:
        """Return pos_embed sized to the current grid, interpolating if needed.

        Interpolation goes through autograd (no `.data` strip) so the saved
        pos_embed remains trainable even when the runtime grid differs from
        the saved one (e.g., dryrun 64×64 vs default 512×512).
        """
        N_dst = h_grid * w_grid
        if self.pos_embed.shape[1] == N_dst + 1:
            return self.pos_embed
        return self._interpolate_pos_embed(
            self.pos_embed,
            target_shape=(1, N_dst + 1, self.EMBED_DIM),
        )

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        modality = self._resolve_modality(x)
        if modality == "sar":
            z = self.patch_embed_SAR(x)
            mie_idx = 1
        else:
            z = self.patch_embed(x)
            mie_idx = 0
        B, D, Hg, Wg = z.shape
        z = z.flatten(2).transpose(1, 2)                # [B, N, D]
        # Add modality identity to every patch (CLS gets none, like TransOSS).
        z = z + self.mie_embed[mie_idx].unsqueeze(0)    # [1, 1, D] → broadcast
        # Pos embed: split cls/patch and add patch tokens, cat cls.
        pos = self._get_pos_embed(Hg, Wg)
        cls = self.cls_token.expand(B, -1, -1) + pos[:, :1]
        z = z + pos[:, 1:]
        z = torch.cat([cls, z], dim=1)                  # [B, N+1, D]

        # ViT-Adapter path: genuine multi-scale via SPM + Injector/Extractor.
        if self.use_adapter:
            feats = self.adapter.run(
                x_img=x, modality=modality, x_tokens=z,
                patch_hw=(Hg, Wg), blocks=self.blocks,
            )
            return feats, []

        taps_needed = set(self.tap_blocks)
        tapped: dict[int, torch.Tensor] = {}
        for i, block in enumerate(self.blocks, start=1):
            z = block(z)
            if i in taps_needed:
                # Drop CLS; normalise the residual-stream tokens over the
                # channel dim (ViTDet) before reshaping to (B, D, Hg, Wg).
                f = z[:, 1:]
                if self.tap_norm:
                    f = self.tap_norms[self._tap_index[i]](f)
                f = f.transpose(1, 2).reshape(B, D, Hg, Wg)
                tapped[i] = f

        # Project in the order specified by tap_blocks (which is the order
        # of the FPN levels stride 4 → 32).
        out_feats: List[torch.Tensor] = []
        for block_idx, tap in zip(self.tap_blocks, self.taps):
            out_feats.append(tap(tapped[block_idx]))
        return out_feats, []


def _drop_pretrained_pos_embed(*args, **kwargs):
    """Public helper for tests — re-export of the interpolation routine."""
    return TransOSSEncoder._interpolate_pos_embed(*args, **kwargs)
