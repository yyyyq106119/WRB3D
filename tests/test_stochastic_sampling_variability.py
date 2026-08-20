import torch

from wrb3d.bridges import ResidualBrownianBridge


def test_stochastic_sampling_variability():
    bridge = ResidualBrownianBridge(20)
    initial = torch.zeros(1, 1, 2, 2, 2)
    predictor = lambda state, t: 0.5 * state + 0.1
    one = bridge.sample_loop(
        predictor,
        initial,
        0.5,
        num_steps=5,
        stochastic=True,
        generator=torch.Generator().manual_seed(1),
    ).residual
    two = bridge.sample_loop(
        predictor,
        initial,
        0.5,
        num_steps=5,
        stochastic=True,
        generator=torch.Generator().manual_seed(2),
    ).residual
    assert not torch.equal(one, two)

