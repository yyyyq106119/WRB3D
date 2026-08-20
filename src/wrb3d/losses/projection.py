"""Explicit output projections; none of them is the formal main-loss path."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def soft_clip_01(x: Tensor, beta: float = 10.0) -> Tensor:
    return F.softplus(x, beta=float(beta)) - F.softplus(x - 1.0, beta=float(beta))


class Projection(nn.Module):
    MODES = {"none", "hard_clamp", "soft_clip", "sigmoid"}

    def __init__(self, mode: str = "none", beta: float = 10.0) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unsupported projection mode={mode!r}")
        self.mode = mode
        self.beta = float(beta)

    def forward(self, raw: Tensor) -> Tensor:
        if self.mode == "none":
            return raw
        if self.mode == "hard_clamp":
            return raw.clamp(0.0, 1.0)
        if self.mode == "soft_clip":
            return soft_clip_01(raw, self.beta)
        return torch.sigmoid(raw)

    def all_views(self, raw: Tensor) -> dict[str, Tensor]:
        return {
            "B_raw": raw,
            "B_clamp": raw.clamp(0.0, 1.0),
            "B_soft": soft_clip_01(raw, self.beta),
            "B_sigmoid": torch.sigmoid(raw),
            "B_projected": self(raw),
        }

