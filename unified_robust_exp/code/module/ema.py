"""Exponential Moving Average for model weights (CoDER v3).

Maintains a shadow copy of model parameters that is updated after every
optimizer step with `ema = decay * ema + (1 - decay) * current`.  Evaluating
from the EMA weights typically gives +0.5-1 mIoU on segmentation tasks since
it averages out late-stage LR oscillations.

Usage (train-time):
    ema = EMAModel(model, decay=0.999)
    # ... after each optimizer.step():
    ema.update(model)
    # ... at eval-time (saves + restores live weights):
    original = ema.apply_to(model)
    metrics = evaluate(model)
    ema.restore(model, *original)

The shadow weights also include buffers (BatchNorm running stats), which is
necessary for correct BN behaviour at eval.  Buffers are simply copied from
the live model (no EMA on them) per common convention.
"""
from __future__ import annotations
from typing import Dict, Tuple

import torch
import torch.nn as nn


class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.detach().clone()
        self.buffer_shadow: Dict[str, torch.Tensor] = {}
        for name, buf in model.named_buffers():
            self.buffer_shadow[name] = buf.data.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for name, param in model.named_parameters():
            if name in self.shadow and param.requires_grad:
                self.shadow[name].mul_(d).add_(param.data, alpha=1.0 - d)
        # Buffers (BN running stats etc.) tracked directly — no smoothing.
        for name, buf in model.named_buffers():
            if name in self.buffer_shadow:
                self.buffer_shadow[name].copy_(buf.data)

    @torch.no_grad()
    def apply_to(
        self, model: nn.Module
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Swap EMA weights into model.  Returns (params_backup, buffers_backup)."""
        params_backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                params_backup[name] = param.data.detach().clone()
                param.data.copy_(self.shadow[name])
        buffers_backup: Dict[str, torch.Tensor] = {}
        for name, buf in model.named_buffers():
            if name in self.buffer_shadow:
                buffers_backup[name] = buf.data.detach().clone()
                buf.data.copy_(self.buffer_shadow[name])
        return params_backup, buffers_backup

    @torch.no_grad()
    def restore(
        self,
        model: nn.Module,
        params_backup: Dict[str, torch.Tensor],
        buffers_backup: Dict[str, torch.Tensor],
    ) -> None:
        for name, param in model.named_parameters():
            if name in params_backup:
                param.data.copy_(params_backup[name])
        for name, buf in model.named_buffers():
            if name in buffers_backup:
                buf.data.copy_(buffers_backup[name])

    def save(self, path: str) -> None:
        torch.save(
            {
                "decay": self.decay,
                "shadow": self.shadow,
                "buffer_shadow": self.buffer_shadow,
            },
            path,
        )

    def load(self, path: str, map_location="cpu") -> None:
        obj = torch.load(path, map_location=map_location, weights_only=False)
        self.decay = float(obj["decay"])
        self.shadow = obj["shadow"]
        self.buffer_shadow = obj["buffer_shadow"]
