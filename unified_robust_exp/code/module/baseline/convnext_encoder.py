"""ConvNeXt-base encoder for v3 Option-1 / Option-3.

Wraps `timm.create_model('convnext_base', features_only=True)` and loads a
LOCAL pretrained checkpoint (HuggingFace Hub is unreachable from this
WSL environment).  The state-dict key naming differs between the saved
checkpoint (`stem.0.weight`) and the features_only model (`stem_0.weight`),
so a small regex remap is applied.

For SAR (1-channel) inputs we initialise the stem conv to the mean across
the 3-channel pretrained stem (the same transform `timm` applies internally
when `in_chans=1` + `pretrained=True`).

Output: a 4-level feature pyramid at strides [4, 8, 16, 32] with channel
widths [128, 256, 512, 1024].  The return signature mirrors the existing
`ResNetEncoder`: `(features_list, whitening_positions_list)`.

Plan B / Option-3: optional IBN-Net-style InstanceNorm insertion.

  When `wt_layer` is provided as a 7-int list mirroring the ResNet convention
  used elsewhere in this repo (positions 3..6 correspond to feature-pyramid
  outputs 0..3), every position set to `2` triggers an
  `InstanceNorm2d(affine=False)` applied to the matching pyramid feature.
  The pre-IN feature is also captured into `w_arr` so the existing MMR cov
  machinery (`CovMatrix_mmr`, `instance_whitening_loss`) can consume it
  unchanged.

  Convention:
      wt_layer[3] == 2 → IN on feat[0] (stride 4, 128 ch) — analog to MetaRS Layer1-out
      wt_layer[4] == 2 → IN on feat[1] (stride 8, 256 ch) — analog to MetaRS Layer2-out
      wt_layer[5] == 2 → IN on feat[2] (stride 16, 512 ch)
      wt_layer[6] == 2 → IN on feat[3] (stride 32, 1024 ch)
      positions 0,1,2  → reserved for stem / pre-stem (not exposed by timm features_only),
                          ignored

  With `wt_layer=None` or all zeros, behaviour is identical to Option-1.
"""
from __future__ import annotations
import os
import re
from typing import List, Tuple

import torch
import torch.nn as nn

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ConvNeXtEncoder requires `timm`.  Install via `pip install timm>=1.0`."
    ) from exc


DEFAULT_PRETRAINED_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "pretrained",
    "pytorch_model.bin",
)


def _remap_timm_keys(sd: dict) -> dict:
    """Map saved-checkpoint keys (`stem.0`, `stages.0.blocks…`) to the
    features_only model's keys (`stem_0`, `stages_0.blocks…`)."""
    out = {}
    for k, v in sd.items():
        k2 = re.sub(r"^stem\.(\d+)", r"stem_\1", k)
        k2 = re.sub(r"^stages\.(\d+)", r"stages_\1", k2)
        out[k2] = v
    return out


def _adapt_stem_to_channels(sd: dict, target_in_chans: int) -> dict:
    """Convert pretrained stem conv weights from 3-channel to `target_in_chans`.

    For target_in_chans==3 the stem is unchanged.  For other values the stem
    weight is averaged across the input channel dim and broadcast to the new
    in_chans (matching what `timm` does when pretrained=True + in_chans!=3).
    """
    if target_in_chans == 3:
        return sd
    key = "stem_0.weight"
    if key not in sd:
        return sd
    w = sd[key]
    # w shape: [out_chans, 3, kH, kW] → average and replicate
    w_avg = w.mean(dim=1, keepdim=True)  # [out_chans, 1, kH, kW]
    if target_in_chans == 1:
        new_w = w_avg
    else:
        new_w = w_avg.expand(-1, target_in_chans, -1, -1).contiguous()
    sd = dict(sd)
    sd[key] = new_w
    return sd


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt-base feature extractor returning a 4-level pyramid.

    With `wt_layer` specifying IN positions (Plan B), the matching pyramid
    outputs go through InstanceNorm2d(affine=False) and the pre-IN tensor is
    captured in w_arr for the CoDER MMR machinery.
    """

    def __init__(
        self,
        in_channels: int = 3,
        pretrained: bool = True,
        pretrained_path: str | None = None,
        model_name: str = "convnext_base",
        wt_layer: list | None = None,
    ):
        # auto-select weights path by model_name
        if pretrained_path is None:
            if "large" in model_name:
                pretrained_path = os.path.join(os.path.dirname(DEFAULT_PRETRAINED_PATH), "convnext_large_pytorch_model.bin")
            else:
                pretrained_path = DEFAULT_PRETRAINED_PATH
        super().__init__()
        # Build without timm's pretrained=True (which calls HF Hub).
        self.model = timm.create_model(
            model_name,
            features_only=True,
            pretrained=False,
            in_chans=in_channels,
            out_indices=(0, 1, 2, 3),
        )
        self.in_channels = in_channels
        self.feature_channels: List[int] = list(self.model.feature_info.channels())
        self.feature_strides: List[int] = list(self.model.feature_info.reduction())

        # Plan B — optional IBN-style IN at feature-pyramid outputs.
        # wt_layer positions 3..6 map to pyramid outputs 0..3.
        self.wt_layer = list(wt_layer) if wt_layer is not None else [0] * 7
        if len(self.wt_layer) < 7:
            self.wt_layer = (self.wt_layer + [0] * 7)[:7]
        self._ins = nn.ModuleDict()
        for wt_idx in (3, 4, 5, 6):
            if self.wt_layer[wt_idx] == 2:
                feat_idx = wt_idx - 3
                ch = self.feature_channels[feat_idx]
                self._ins[str(feat_idx)] = nn.InstanceNorm2d(ch, affine=False)

        if pretrained:
            if not os.path.isfile(pretrained_path):
                raise FileNotFoundError(
                    f"ConvNeXt pretrained checkpoint not found at {pretrained_path}. "
                    "Download from "
                    "https://hf-mirror.com/timm/convnext_base.fb_in22k_ft_in1k/"
                    "resolve/main/pytorch_model.bin "
                    "and place it at the expected path, or pass pretrained_path=…"
                )
            sd = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            sd = _remap_timm_keys(sd)
            sd = _adapt_stem_to_channels(sd, in_channels)
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            # Drop the classifier head keys from unexpected — features_only
            # mode doesn't have them, that's intentional.
            unexpected = [k for k in unexpected if not k.startswith("head.")]
            if missing:
                # Allow stem conv to be missing when in_channels mismatched in a way
                # adapt_stem_to_channels didn't handle — we already initialised the
                # stem conv via timm.create_model; leave it as-is.
                pass
            if unexpected:
                # Surface anything unexpected we don't know about.
                import logging
                logging.getLogger(__name__).warning(
                    "ConvNeXt load_state_dict had unexpected keys: %s", unexpected[:5]
                )

    @property
    def out_channels(self) -> List[int]:
        return self.feature_channels

    @property
    def stem_conv(self) -> nn.Conv2d:
        return self.model.stem_0

    @stem_conv.setter
    def stem_conv(self, conv: nn.Conv2d) -> None:
        self.model.stem_0 = conv

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        feats = self.model(x)
        if not self._ins:
            return feats, []
        # Plan B: apply IN at marked positions, capturing the post-IN tensor
        # in w_arr (matches ResNet IBN convention — see _resnets.Bottleneck
        # forward: `_w = out` after `out = self.instance_norm_layer(out)`).
        out_feats: List[torch.Tensor] = []
        w_arr: List[torch.Tensor] = []
        for idx, f in enumerate(feats):
            key = str(idx)
            if key in self._ins:
                f = self._ins[key](f)
                w_arr.append(f)
            out_feats.append(f)
        return out_feats, w_arr
