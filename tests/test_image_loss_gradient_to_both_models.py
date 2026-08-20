import torch


def test_image_loss_gradient_to_both_models(tiny_model, paired_tensors, covariances):
    _, pet = paired_tensors
    output = tiny_model.forward_train(*paired_tensors, *covariances, t=torch.tensor([5]))
    (output["B_raw"] - pet).abs().mean().backward()
    low = sum(float(p.grad.abs().sum()) for p in tiny_model.low_model.parameters() if p.grad is not None)
    high = sum(float(p.grad.abs().sum()) for p in tiny_model.high_model.parameters() if p.grad is not None)
    assert low > 0 and high > 0

