import torch

from wrb3d.models import WaveletResidualBridgeModel


def test_soft_clip_not_main_loss(paired_tensors, covariances):
    model = WaveletResidualBridgeModel(
        channels=(4, 8), condition_dim=16, num_timesteps=10, projection_mode="soft_clip"
    )
    output = model.forward_train(*paired_tensors, *covariances, t=torch.tensor([5]))
    assert model.loss_image_domain == "raw"
    assert not bool(output["logs"]["image_supervision_is_projected"])
    assert output["B_raw"].data_ptr() != output["B_views"]["B_soft"].data_ptr()
