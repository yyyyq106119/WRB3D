import torch

from wrb3d.metrics import image_metrics, structural_similarity_3d


def test_identical_3d_images_have_unit_ssim_and_high_psnr():
    image = torch.linspace(0, 1, 8 * 8 * 8).reshape(1, 1, 8, 8, 8)
    metrics = image_metrics(image, image, data_range=1.0)
    assert torch.allclose(metrics["ssim3d"], torch.tensor(1.0), atol=1e-5)
    assert metrics["mae"] == 0
    assert metrics["psnr"] > 100


def test_ssim_requires_matching_3d_shapes():
    left = torch.zeros(1, 1, 4, 4, 4)
    right = torch.zeros(1, 1, 4, 4, 3)
    try:
        structural_similarity_3d(left, right)
    except ValueError:
        return
    raise AssertionError("mismatched images were accepted")
