from wrb3d.bridges import ResidualBrownianBridge


def test_sampling_supports_5_15_50_steps():
    bridge = ResidualBrownianBridge(1000)
    for count in (5, 15, 50):
        steps = bridge.sampling_timesteps(count, "cpu")
        assert len(steps) == count + 1
        assert steps[0] == 1000 and steps[-1] == 0

