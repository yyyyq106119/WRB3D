import torch

from wrb3d.wavelets import DWT3D


def test_wavelet_energy_preservation_even_shape():
    x = torch.randn(2, 3, 8, 10, 12)
    low, high, _ = DWT3D()(x)
    torch.testing.assert_close(
        x.square().sum(), low.square().sum() + high.square().sum(), atol=2e-4, rtol=2e-6
    )

