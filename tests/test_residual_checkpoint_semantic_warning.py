import pytest

from wrb3d.models import WaveletResidualBridgeModel
from wrb3d.training import load_checkpoint, save_checkpoint


def test_residual_checkpoint_semantic_warning(tmp_path):
    legacy = WaveletResidualBridgeModel(
        channels=(4, 8), condition_dim=16, num_timesteps=10, prediction_target="endpoint_x0_legacy"
    )
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, legacy)
    residual = WaveletResidualBridgeModel(channels=(4, 8), condition_dim=16, num_timesteps=10)
    with pytest.warns(RuntimeWarning, match="output semantics are incompatible"):
        result = load_checkpoint(path, residual, warm_start=True)
    assert result["semantic_mismatch"]
    assert result["warm_start"]

