import torch

from wrb3d.wavelets import DWT3D, IDWT3D, add_residuals, compute_residuals


def test_wavelet_residual_equivalence(paired_tensors):
    mri, pet = paired_tensors
    transform, inverse = DWT3D(), IDWT3D()
    a_low, a_high, meta = transform(mri)
    b_low, b_high, _ = transform(pet)
    residual = compute_residuals(a_low, a_high, b_low, b_high)
    reconstructed = inverse(*add_residuals(a_low, a_high, residual.low, residual.high), meta)
    torch.testing.assert_close(reconstructed, pet, atol=1e-6, rtol=1e-6)

