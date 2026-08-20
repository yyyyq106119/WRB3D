import torch

from wrb3d.wavelets import DWT3D, IDWT3D


def test_raw_reconstruction_gradient_reaches_low_and_high(paired_tensors):
    mri, _ = paired_tensors
    low, high, meta = DWT3D()(mri)
    residual_low = torch.randn_like(low, requires_grad=True)
    residual_high = torch.randn_like(high, requires_grad=True)
    raw = IDWT3D()(low + residual_low, high + residual_high, meta)
    raw.square().mean().backward()
    assert residual_low.grad is not None and residual_low.grad.abs().sum() > 0
    assert residual_high.grad is not None and residual_high.grad.abs().sum() > 0

