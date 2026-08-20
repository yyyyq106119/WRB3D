"""The single authoritative fixed orthonormal one-level 3-D Haar transform.

The 2x2x2 block is flattened with W varying fastest, then H, then D.  The
Hadamard rows therefore have the public order

    LLL, HLL, LHL, HHL, LLH, HLH, LHH, HHH.

This corrects labels in WBBDM_v10/v11 without changing the legacy numerical
channel order.  L/H in a label means low/high response along D, H, W.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

LOW_BAND = "LLL"
HIGH_BAND_ORDER = ("HLL", "LHL", "HHL", "LLH", "HLH", "LHH", "HHH")
ALL_BAND_ORDER = (LOW_BAND, *HIGH_BAND_ORDER)


@dataclass(frozen=True)
class WaveletMeta:
    original_shape: tuple[int, int, int]
    padding: tuple[int, int, int]


def _basis() -> Tensor:
    return torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, -1, -1, -1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1],
            [1, 1, -1, -1, -1, -1, 1, 1],
            [1, -1, 1, -1, 1, -1, 1, -1],
            [1, -1, 1, -1, -1, 1, -1, 1],
            [1, -1, -1, 1, 1, -1, -1, 1],
            [1, -1, -1, 1, -1, 1, 1, -1],
        ],
        dtype=torch.float32,
    ) / math.sqrt(8.0)


class DWT3D(nn.Module):
    """Fixed Haar analysis transform for tensors shaped ``[B,C,D,H,W]``."""

    def __init__(self, pad_mode: str = "replicate") -> None:
        super().__init__()
        if pad_mode not in {"replicate", "reflect", "constant"}:
            raise ValueError(f"unsupported pad_mode={pad_mode!r}")
        self.pad_mode = pad_mode
        self.register_buffer("basis", _basis(), persistent=True)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, WaveletMeta]:
        if x.ndim != 5:
            raise ValueError(f"expected [B,C,D,H,W], got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError("DWT3D requires a floating-point tensor")
        original = tuple(int(v) for v in x.shape[-3:])
        pd, ph, pw = (v % 2 for v in original)
        out_dtype = x.dtype
        # Older CPU PyTorch builds do not implement replicate/reflect pad for
        # Half.  Promote before padding and keep the complete Haar calculation
        # in FP32; coefficients are cast back to the public input dtype below.
        work = x.float() if x.dtype in {torch.float16, torch.bfloat16} else x
        if pd or ph or pw:
            work = F.pad(work, (0, pw, 0, ph, 0, pd), mode=self.pad_mode)
        b, c, d, h, w = work.shape
        blocks = work.reshape(b, c, d // 2, 2, h // 2, 2, w // 2, 2)
        blocks = blocks.permute(0, 1, 2, 4, 6, 3, 5, 7).reshape(
            b, c, d // 2, h // 2, w // 2, 8
        )
        # Sum with exact +/-1 signs before applying the common normalization.
        # Multiplying every voxel by 1/sqrt(8) before cancellation can leave
        # spurious high-frequency coefficients for a constant volume.
        signs = torch.sign(self.basis).to(device=work.device, dtype=work.dtype)
        coeff = torch.einsum("bcdhwk,lk->bcdhwl", blocks, signs) / math.sqrt(8.0)
        # Public coefficient channels are band-major: all modality channels of
        # LLL, followed by all channels of HLL, ..., HHH.  Keeping the explicit
        # band axis prevents a C-major flatten from silently mixing low/high
        # semantics when C > 1.
        coeff = coeff.permute(0, 5, 1, 2, 3, 4).to(out_dtype)
        low = coeff[:, 0]
        high = coeff[:, 1:].reshape(b, 7 * c, d // 2, h // 2, w // 2)
        return low, high, WaveletMeta(original, (pd, ph, pw))

    decompose = forward


class IDWT3D(nn.Module):
    """Fixed Haar synthesis transform matching :class:`DWT3D`."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("basis", _basis(), persistent=True)

    def forward(
        self,
        low: Tensor,
        high: Tensor,
        meta: WaveletMeta | Sequence[int] | None = None,
    ) -> Tensor:
        if low.ndim != 5 or high.ndim != 5:
            raise ValueError("low and high must be [B,C,D,H,W]")
        if high.shape[0] != low.shape[0] or high.shape[2:] != low.shape[2:]:
            raise ValueError("low/high batch and spatial shapes must match")
        if high.shape[1] != 7 * low.shape[1]:
            raise ValueError("high channels must equal seven times low channels")
        if low.dtype != high.dtype or low.device != high.device:
            raise ValueError("low/high dtype and device must match")
        b, c, d, h, w = low.shape
        out_dtype = low.dtype
        high_bands = high.reshape(b, 7, c, d, h, w)
        coeff = torch.cat((low.unsqueeze(1), high_bands), dim=1)
        work = coeff.float() if coeff.dtype in {torch.float16, torch.bfloat16} else coeff
        coeff_blocks = work.permute(0, 2, 3, 4, 5, 1)
        signs = torch.sign(self.basis).to(device=low.device, dtype=work.dtype)
        blocks = torch.einsum(
            "bcdhwl,lk->bcdhwk",
            coeff_blocks,
            signs.t(),
        ) / math.sqrt(8.0)
        x = blocks.reshape(b, c, d, h, w, 2, 2, 2).permute(0, 1, 2, 5, 3, 6, 4, 7)
        x = x.reshape(b, c, 2 * d, 2 * h, 2 * w).to(out_dtype)
        if meta is not None:
            shape = meta.original_shape if isinstance(meta, WaveletMeta) else tuple(int(v) for v in meta)
            if len(shape) != 3:
                raise ValueError("output shape must contain D,H,W")
            x = x[..., : shape[0], : shape[1], : shape[2]]
        return x

    reconstruct = forward
