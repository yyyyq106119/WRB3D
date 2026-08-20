import torch


def test_low_loss_gradient_to_low_model(tiny_model, paired_tensors, covariances):
    output = tiny_model.forward_train(*paired_tensors, *covariances, t=torch.tensor([5]))
    loss = (output["predicted_low_residual"] - output["target_low_residual"]).abs().mean()
    loss.backward()
    gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in tiny_model.low_model.parameters()
        if parameter.grad is not None
    )
    assert gradient > 0

