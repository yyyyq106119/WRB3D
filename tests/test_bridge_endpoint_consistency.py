import torch

from wrb3d.bridges import ResidualBrownianBridge


def test_bridge_endpoint_consistency():
    bridge = ResidualBrownianBridge(10)
    residual = torch.randn(3, 7, 2, 2, 2)
    noise = torch.randn_like(residual)
    at_target, _ = bridge.q_sample(residual, torch.zeros(3, dtype=torch.long), 0.4, noise)
    at_source, _ = bridge.q_sample(residual, torch.full((3,), 10), 0.4, noise)
    torch.testing.assert_close(at_target, residual)
    torch.testing.assert_close(at_source, torch.zeros_like(residual), atol=1e-7, rtol=0)

