import torch


def test_high_model_outputs_seven_residual_bands(tiny_model, paired_tensors, covariances):
    output = tiny_model.forward_train(*paired_tensors, *covariances, t=torch.tensor([5]))
    prediction = output["predicted_high_residual"]
    assert prediction.shape == output["target_high_residual"].shape
    assert prediction.shape[1] == 7

