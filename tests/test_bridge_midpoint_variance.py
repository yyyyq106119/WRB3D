import torch

from wrb3d.bridges import ResidualBrownianBridge


def test_bridge_midpoint_variance_formula():
    bridge = ResidualBrownianBridge(10)
    residual = torch.zeros(1, 1, 2, 2, 2)
    _, variance = bridge.marginal_parameters(residual, 5, 0.8)
    torch.testing.assert_close(variance, torch.full_like(residual, 0.4))

