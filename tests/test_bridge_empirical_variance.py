import torch

from wrb3d.bridges import ResidualBrownianBridge


def test_bridge_empirical_variance():
    torch.manual_seed(5)
    bridge = ResidualBrownianBridge(10)
    clean = torch.zeros(20000, 1, 1, 1, 1)
    sample, _ = bridge.q_sample(clean, torch.full((20000,), 5), 0.8)
    assert abs(float(sample.var(unbiased=False)) - 0.4) < 0.015

