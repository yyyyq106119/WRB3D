import torch


def test_high_loss_gradient_to_high_model(tiny_model, paired_tensors, covariances):
    output = tiny_model.forward_train(*paired_tensors, *covariances, t=torch.tensor([5]))
    loss = (output["predicted_high_residual"] - output["target_high_residual"]).abs().mean()
    loss.backward()
    gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in tiny_model.high_model.parameters()
        if parameter.grad is not None
    )
    assert gradient > 0

