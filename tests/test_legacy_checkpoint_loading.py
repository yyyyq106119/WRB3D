from wrb3d.models import WaveletResidualBridgeModel
from wrb3d.training import load_checkpoint, save_checkpoint


def _legacy():
    return WaveletResidualBridgeModel(
        channels=(4, 8),
        condition_dim=16,
        num_timesteps=10,
        prediction_target="endpoint_x0_legacy",
    )


def test_legacy_checkpoint_loading_in_legacy_mode(tmp_path):
    source = _legacy()
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, source, config={})
    target = _legacy()
    result = load_checkpoint(path, target, config={})
    assert not result["semantic_mismatch"]
    assert not result["architecture_mismatch"]
