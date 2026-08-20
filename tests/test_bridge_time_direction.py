from wrb3d.bridges import ResidualBrownianBridge


def test_bridge_time_direction_is_source_T_to_target_zero():
    bridge = ResidualBrownianBridge(50)
    assert bridge.sampling_timesteps(5, "cpu").tolist() == [50, 40, 30, 20, 10, 0]

