import torch


def test_low_high_same_timestep(tiny_model, paired_tensors, covariances):
    mri, pet = paired_tensors
    low_cov, high_cov = covariances
    output = tiny_model.forward_train(mri, pet, low_cov, high_cov, t=torch.tensor([4]))
    assert output["t_low"] is output["t_high"]
    assert output["t_low"].tolist() == [4]

