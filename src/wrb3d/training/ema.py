"""Single authoritative exponential-moving-average implementation."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError("EMA decay must be within [0,1)")
        self.decay = float(decay)
        self.num_updates = 0
        self.module = deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = model.state_dict()
        target = self.module.state_dict()
        for name, value in target.items():
            incoming = source[name].detach().to(device=value.device)
            if value.is_floating_point():
                value.mul_(self.decay).add_(incoming, alpha=1.0 - self.decay)
            else:
                value.copy_(incoming)
        self.num_updates += 1

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "module": self.module.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state["decay"])
        self.num_updates = int(state.get("num_updates", 0))
        self.module.load_state_dict(state["module"], strict=True)

