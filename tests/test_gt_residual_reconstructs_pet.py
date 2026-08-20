import torch

from wrb3d.wavelets import DWT3D, IDWT3D


def test_gt_residual_reconstructs_pet(paired_tensors):
    mri, pet = paired_tensors
    transform = DWT3D()
    a_low, a_high, meta = transform(mri)
    b_low, b_high, _ = transform(pet)
    reconstructed = IDWT3D()(a_low + (b_low - a_low), a_high + (b_high - a_high), meta)
    torch.testing.assert_close(reconstructed, pet, atol=1e-6, rtol=1e-6)

