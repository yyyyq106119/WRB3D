import torch

from wrb3d.wavelets import compute_residuals


def test_high_residual_is_pet_minus_mri_per_band():
    a_low = torch.randn(1, 1, 2, 2, 2)
    b_low = torch.randn_like(a_low)
    a_high = torch.randn(1, 7, 2, 2, 2)
    b_high = torch.randn_like(a_high)
    assert torch.equal(compute_residuals(a_low, a_high, b_low, b_high).high, b_high - a_high)

