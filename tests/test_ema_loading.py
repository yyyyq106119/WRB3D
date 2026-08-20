import torch

from wrb3d.models import WaveletResidualBridgeModel
from wrb3d.training import EMA, load_checkpoint, save_checkpoint


def _model():
    return WaveletResidualBridgeModel(channels=(4, 8), condition_dim=16, num_timesteps=10)


def test_ema_loading(tmp_path):
    model = _model()
    ema = EMA(model, 0.9)
    with torch.no_grad():
        next(model.parameters()).add_(1)
    ema.update(model)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, ema=ema, config={})
    restored = _model()
    restored_ema = EMA(restored, 0.5)
    result = load_checkpoint(path, restored, ema=restored_ema, config={})
    assert result["ema_loaded"]
    assert restored_ema.num_updates == 1
    for left, right in zip(ema.module.parameters(), restored_ema.module.parameters()):
        torch.testing.assert_close(left, right)
