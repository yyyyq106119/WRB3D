import torch

from wrb3d.losses import ResidualBridgeLoss


def test_raw_loss_before_projection():
    loss = ResidualBridgeLoss(lambda_low=0, lambda_high=0, lambda_image=1, lambda_range=0)
    low = torch.zeros(1, 1, 1, 1, 1)
    high = torch.zeros(1, 7, 1, 1, 1)
    raw = torch.tensor([[[[[-0.5, 1.5]]]]])
    target = torch.zeros_like(raw)
    total, _ = loss(low, low, high, high, raw, target)
    projected_total, _ = loss(
        low, low, high, high, raw, target, image_for_loss=raw.clamp(0, 1)
    )
    assert total != projected_total

