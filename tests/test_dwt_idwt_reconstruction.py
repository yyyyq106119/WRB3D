import pytest
import torch

from wrb3d.wavelets import DWT3D, IDWT3D


@pytest.mark.parametrize("shape", [(1, 1, 8, 10, 12), (1, 2, 7, 9, 11)])
def test_dwt_idwt_reconstruction(shape):
    x = torch.randn(shape)
    low, high, meta = DWT3D()(x)
    reconstructed = IDWT3D()(low, high, meta)
    torch.testing.assert_close(reconstructed, x, atol=1e-6, rtol=1e-6)


def test_cpu_float16_odd_shape_padding_is_supported():
    torch.manual_seed(804)
    x = torch.randn(1, 1, 9, 10, 11, dtype=torch.float16)
    low, high, meta = DWT3D()(x)
    reconstructed = IDWT3D()(low, high, meta)
    assert low.dtype == high.dtype == reconstructed.dtype == torch.float16
    torch.testing.assert_close(reconstructed, x, atol=2e-3, rtol=2e-3)
