import torch

from wrb3d.wavelets import DWT3D, IDWT3D


def test_zero_residual_reconstructs_mri(paired_tensors):
    mri, _ = paired_tensors
    low, high, meta = DWT3D()(mri)
    reconstructed = IDWT3D()(low + torch.zeros_like(low), high + torch.zeros_like(high), meta)
    torch.testing.assert_close(reconstructed, mri, atol=1e-6, rtol=1e-6)

