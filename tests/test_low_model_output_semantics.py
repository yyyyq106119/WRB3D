import torch


def test_low_model_output_semantics_are_residual_x0(tiny_model, paired_tensors, covariances):
    mri, pet = paired_tensors
    output = tiny_model.forward_train(mri, pet, *covariances, t=torch.tensor([5]))
    assert tiny_model.prediction_target == "residual_x0"
    assert output["predicted_low_residual"].shape == output["target_low_residual"].shape
    assert not torch.equal(output["target_low_residual"], output["target_low_residual"] + 1)

