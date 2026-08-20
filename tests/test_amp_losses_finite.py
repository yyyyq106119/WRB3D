import torch


def test_amp_losses_finite_cpu_bfloat16(tiny_model, paired_tensors, covariances):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = tiny_model.forward_train(*paired_tensors, *covariances, t=torch.tensor([5]))
    assert torch.isfinite(output["loss"])

