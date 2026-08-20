import pytest
import torch

from wrb3d.models import WaveletResidualBridgeModel


@pytest.fixture
def paired_tensors():
    generator = torch.Generator().manual_seed(7)
    mri = torch.rand((1, 1, 8, 8, 8), generator=generator)
    pet = (0.7 * mri + 0.3 * torch.rand(mri.shape, generator=generator)).clamp(0, 1)
    return mri, pet


@pytest.fixture
def tiny_model():
    torch.manual_seed(11)
    return WaveletResidualBridgeModel(
        channels=(4, 8), condition_dim=16, num_timesteps=10, projection_mode="none"
    )


@pytest.fixture
def covariances():
    return torch.tensor(0.05), torch.full((7,), 0.05)

