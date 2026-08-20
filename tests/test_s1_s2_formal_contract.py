"""Twenty explicit engineering gates required by the formal S1/S2 protocol."""

from __future__ import annotations

import inspect
import socket

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from wrb3d.losses import (
    AuxiliaryWeights,
    aligned_high_losses,
    build_hotspot_mask,
    hotspot_losses,
)
from wrb3d.metrics import formal_case_metrics
from wrb3d.models import CaseAdaptiveWaveletCorrector, WaveletResidualBridgeModel
from wrb3d.training.s1s2 import WarmupCosineStepScheduler


def _model(corrector: bool = False) -> WaveletResidualBridgeModel:
    return WaveletResidualBridgeModel(
        channels=(4, 8),
        condition_dim=8,
        num_timesteps=10,
        prediction_target="residual_x0",
        low_to_high_condition="feature_gating",
        loss_kwargs={"high_band_stds": torch.ones(7), "image_domain": "raw"},
        corrector_kwargs={
            "enabled": corrector,
            "hidden_dim": 8,
            "gamma": 0.15,
            "identity_init": True,
        },
        auxiliary_loss_kwargs={
            "hotspot": {"enabled": True},
            "aligned_amplitude": {
                "enabled": True,
                "band_epsilon": [1e-10] * 7,
                "smooth_l1_beta": 0.1,
            },
            "gain_regularization": {"weight": 1e-3 if corrector else 0.0},
        },
    )


def _pair() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(41)
    mri = torch.rand(1, 1, 8, 8, 8)
    pet = torch.rand(1, 1, 8, 8, 8)
    return mri, pet


def test_01_disabled_features_preserve_canonical_numerics() -> None:
    model = _model(False)
    mri, pet = _pair()
    t = torch.tensor([5])
    low_noise = torch.zeros(1, 1, 4, 4, 4)
    high_noise = torch.zeros(1, 7, 4, 4, 4)
    first = model.forward_train(
        mri, pet, 0.1, torch.ones(7) * 0.1, t=t, noise_low=low_noise, noise_high=high_noise
    )
    second = model.forward_train(
        mri,
        pet,
        0.1,
        torch.ones(7) * 0.1,
        t=t,
        noise_low=low_noise,
        noise_high=high_noise,
        auxiliary_weights=AuxiliaryWeights(),
    )
    assert torch.equal(first["B_raw"], second["B_raw"])
    assert torch.equal(first["loss"], second["loss"])


def test_02_corrector_identity_initialization_is_exact() -> None:
    module = CaseAdaptiveWaveletCorrector(8, 8, hidden_dim=8)
    low = torch.randn(2, 8, 2, 2, 2)
    high = torch.randn(2, 8, 2, 2, 2)
    residual = torch.randn(2, 7, 4, 4, 4)
    corrected, gains = module(low, high, residual, torch.randn_like(residual))
    assert torch.equal(gains, torch.ones_like(gains))
    assert torch.equal(corrected, residual)


def test_03_shared_backbone_state_is_elementwise_identical() -> None:
    torch.manual_seed(1234)
    s2 = _model(False)
    shared = {
        key: value.clone()
        for key, value in s2.state_dict().items()
        if key.startswith(("low_model.", "high_model."))
    }
    torch.manual_seed(9)
    s1 = _model(True)
    s1.load_state_dict(shared, strict=False)
    for key, value in shared.items():
        assert torch.equal(s1.state_dict()[key], value)


def test_04_s2_has_no_corrector_parameter() -> None:
    assert not any("corrector" in name for name, _ in _model(False).named_parameters())


def test_05_corrector_signature_cannot_receive_gt_pet_or_roi() -> None:
    names = set(inspect.signature(CaseAdaptiveWaveletCorrector.forward).parameters)
    assert names == {
        "self",
        "low_bottleneck",
        "high_bottleneck",
        "raw_high_residual",
        "mri_high",
    }


def test_06_hotspot_mask_is_detached_supervision_only() -> None:
    pet = torch.rand(1, 1, 8, 8, 8, requires_grad=True)
    mask, _, _ = build_hotspot_mask(pet)
    assert not mask.requires_grad


def test_07_empty_foreground_safely_skips_case() -> None:
    pet = torch.zeros(1, 1, 8, 8, 8)
    mask, valid, _ = build_hotspot_mask(pet)
    losses = hotspot_losses(pet, pet, mask, valid)
    assert not bool(valid.item())
    assert losses["hotspot"].item() == 0.0
    assert all(torch.isfinite(value) for value in losses.values())


def test_08_pred_equals_gt_new_losses_are_near_zero() -> None:
    pet = torch.rand(1, 1, 8, 8, 8)
    mask, valid, _ = build_hotspot_mask(pet)
    hot = hotspot_losses(pet, pet, mask, valid)
    high = torch.randn(1, 7, 4, 4, 4)
    directional = aligned_high_losses(high, high, torch.full((7,), 1e-12))
    assert hot["hotspot"] <= 1.01e-3
    assert hot["underestimation"] == 0
    assert directional["aligned_amplitude"] < 1e-6
    assert directional["orthogonal_error"] < 1e-5


def test_09_zero_energy_bands_are_finite_and_invalid() -> None:
    zeros = torch.zeros(1, 7, 4, 4, 4)
    result = aligned_high_losses(zeros, zeros, torch.full((7,), 1e-8))
    assert not bool(result["valid"].any())
    assert all(
        torch.isfinite(result[key])
        for key in ("aligned_amplitude", "orthogonal_error", "valid_band_fraction")
    )


def test_10_aligned_beta_is_one_for_exact_nonzero_prediction() -> None:
    truth = torch.randn(1, 7, 8, 8, 8)
    result = aligned_high_losses(truth, truth, torch.full((7,), 1e-14))
    assert torch.allclose(result["beta"], torch.ones_like(result["beta"]), atol=1e-5)


def test_11_orthogonal_error_is_zero_for_exact_prediction() -> None:
    truth = torch.randn(1, 7, 8, 8, 8)
    result = aligned_high_losses(truth, truth, torch.full((7,), 1e-14))
    assert result["orthogonal_error"] < 1e-5


def test_12_gain_always_obeys_exponential_bounds() -> None:
    module = CaseAdaptiveWaveletCorrector(8, 8, hidden_dim=8, gamma=0.15)
    torch.nn.init.normal_(module.mlp[-1].weight, std=100.0)
    low = torch.randn(3, 8, 2, 2, 2)
    high = torch.randn(3, 8, 2, 2, 2)
    residual = torch.randn(3, 7, 4, 4, 4)
    _, gain = module(low, high, residual, residual)
    lower, upper = module.gain_bounds
    assert torch.all(gain >= lower)
    assert torch.all(gain <= upper)


def test_13_s1_parameters_enter_optimizer_and_ddp() -> None:
    model = _model(True)
    optimizer = torch.optim.Adam(model.parameters())
    optimizer_ids = {id(value) for group in optimizer.param_groups for value in group["params"]}
    corrector = [
        value
        for name, value in model.named_parameters()
        if "case_adaptive_wavelet_corrector" in name
    ]
    assert corrector and all(id(value) in optimizer_ids for value in corrector)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        port = handle.getsockname()[1]
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}?use_libuv=0",
        rank=0,
        world_size=1,
    )
    try:
        wrapped = DistributedDataParallel(model)
        assert any(
            "case_adaptive_wavelet_corrector" in name
            for name, _ in wrapped.named_parameters()
        )
    finally:
        dist.destroy_process_group()


def test_14_every_trainable_parameter_occurs_exactly_once() -> None:
    model = _model(True)
    optimizer = torch.optim.Adam(model.parameters())
    ids = [id(value) for group in optimizer.param_groups for value in group["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(value) for value in model.parameters() if value.requires_grad}


def test_15_lr_matches_epoch_0_19_20_799_formula() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1e-4)
    scheduler = WarmupCosineStepScheduler(optimizer, steps_per_epoch=78)
    assert scheduler.lr_for_epoch(0) == pytest.approx(5e-6)
    assert scheduler.lr_for_epoch(19) == pytest.approx(1e-4)
    assert scheduler.lr_for_epoch(20) == pytest.approx(1e-4)
    expected = 1e-6 + 0.5 * 99e-6 * (
        1 + torch.cos(torch.tensor(torch.pi * 779 / 780)).item()
    )
    assert scheduler.lr_for_epoch(799) == pytest.approx(expected)


def test_16_cosine_has_no_restart() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1e-4)
    scheduler = WarmupCosineStepScheduler(optimizer, steps_per_epoch=3)
    values = [scheduler.lr_for_epoch(epoch) for epoch in range(20, 800)]
    assert all(left >= right for left, right in zip(values, values[1:]))


def test_17_scheduler_checkpoint_roundtrip_is_exact() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1e-4)
    first = WarmupCosineStepScheduler(optimizer, steps_per_epoch=7)
    for _ in range(39):
        first.prepare_step()
        first.step()
    second = WarmupCosineStepScheduler(optimizer, steps_per_epoch=7)
    second.load_state_dict(first.state_dict())
    assert second.state_dict() == first.state_dict()
    assert second.prepare_step() == first.prepare_step()


def test_18_amp_contract_forward_backward_is_finite() -> None:
    model = _model(True)
    mri, pet = _pair()
    with torch.autocast(device_type="cpu", enabled=False):
        result = model.forward_train(
            mri,
            pet,
            0.1,
            torch.ones(7) * 0.1,
            t=torch.tensor([5]),
            noise_low=torch.zeros(1, 1, 4, 4, 4),
            noise_high=torch.zeros(1, 7, 4, 4, 4),
        )
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_19_no_gt_inference_runs_and_signature_has_no_pet() -> None:
    model = _model(True).eval()
    mri, _ = _pair()
    assert "pet" not in inspect.signature(model.infer).parameters
    with torch.no_grad():
        result = model.infer(mri, 0.1, torch.ones(7) * 0.1, num_steps=1)
    assert result["B_raw"].shape == mri.shape


def test_20_validation_metrics_do_not_build_backward_graph() -> None:
    model = _model(True).eval()
    mri, pet = _pair()
    with torch.no_grad():
        inference = model.infer(mri, 0.1, torch.ones(7) * 0.1, num_steps=1)
        metrics = formal_case_metrics(model, mri, pet, inference)
    assert metrics
    assert all(not value.requires_grad for value in metrics.values())
