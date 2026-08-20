import math

import torch

from wrb3d.wavelets import DWT3D, HIGH_BAND_ORDER


def _pattern(label):
    axes = []
    for symbol in label:
        axes.append(torch.tensor([1.0, 1.0]) if symbol == "L" else torch.tensor([1.0, -1.0]))
    return torch.einsum("d,h,w->dhw", *axes).reshape(1, 1, 2, 2, 2)


def test_wavelet_subband_order_matches_tensor_axes():
    assert HIGH_BAND_ORDER == ("HLL", "LHL", "HHL", "LLH", "HLH", "LHH", "HHH")
    transform = DWT3D()
    for index, label in enumerate(HIGH_BAND_ORDER):
        low, high, _ = transform(_pattern(label))
        assert low.abs().max() < 1e-6
        expected = torch.zeros_like(high)
        expected[:, index] = math.sqrt(8.0)
        torch.testing.assert_close(high, expected, atol=1e-6, rtol=1e-6)

