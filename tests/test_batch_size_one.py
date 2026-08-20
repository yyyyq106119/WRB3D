import torch


def test_batch_size_one(tiny_model, paired_tensors, covariances):
    output = tiny_model.forward_train(*paired_tensors, *covariances, t=torch.tensor([5]))
    assert output["B_raw"].shape[0] == 1
    assert torch.isfinite(output["loss"])

