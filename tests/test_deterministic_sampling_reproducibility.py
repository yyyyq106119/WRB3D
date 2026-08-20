import torch

from wrb3d.bridges import ResidualBrownianBridge


def test_deterministic_sampling_reproducibility():
    bridge = ResidualBrownianBridge(20)
    initial = torch.zeros(1, 1, 2, 2, 2)
    predictor = lambda state, t: torch.ones_like(state) * 0.25 + 0.1 * state
    first = bridge.sample_loop(predictor, initial, 0.2, num_steps=5).residual
    second = bridge.sample_loop(predictor, initial, 0.2, num_steps=5).residual
    assert torch.equal(first, second)

