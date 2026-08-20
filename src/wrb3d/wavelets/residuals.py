"""Canonical MRI-to-PET residual definitions in Haar coefficient space."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class WaveletResiduals:
    low: Tensor
    high: Tensor


def compute_residuals(
    mri_low: Tensor,
    mri_high: Tensor,
    pet_low: Tensor,
    pet_high: Tensor,
) -> WaveletResiduals:
    if mri_low.shape != pet_low.shape or mri_high.shape != pet_high.shape:
        raise ValueError("MRI/PET wavelet coefficient shapes must match")
    return WaveletResiduals(low=pet_low - mri_low, high=pet_high - mri_high)


def add_residuals(
    mri_low: Tensor,
    mri_high: Tensor,
    residual_low: Tensor,
    residual_high: Tensor,
) -> tuple[Tensor, Tensor]:
    if mri_low.shape != residual_low.shape or mri_high.shape != residual_high.shape:
        raise ValueError("MRI coefficient and predicted residual shapes must match")
    return mri_low + residual_low, mri_high + residual_high

