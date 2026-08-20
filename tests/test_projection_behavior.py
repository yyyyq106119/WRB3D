import torch

from wrb3d.losses import Projection, soft_clip_01


def test_projection_modes_and_soft_clip_samples():
    x = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], requires_grad=True)
    soft = soft_clip_01(x, 10.0)
    soft.sum().backward()
    assert torch.all(x.grad > 0)
    assert torch.equal(Projection("none")(x.detach()), x.detach())
    assert torch.equal(Projection("hard_clamp")(torch.tensor([-1.0, 2.0])), torch.tensor([0.0, 1.0]))
    assert torch.all((soft > 0) & (soft < 1))

