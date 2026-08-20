from pathlib import Path

import pytest
import torch

from wrb3d.training import EMA, save_checkpoint
from wrb3d.training.psnr_refine import (
    collect_mse_gradient_ratios,
    load_ema_as_base_initialization,
    mse_weight_for_epoch,
    refinement_constraints,
    solve_mse_weight,
    update_refinement_best,
    validated_auxiliary_weights,
)
from wrb3d.models import WaveletResidualBridgeModel


def _source_config() -> dict:
    return {
        "experiment": {"id": "S2", "name": "source"},
        "model": {"case_adaptive_wavelet_corrector": {"enabled": False}},
    }


def _auxiliary() -> dict[str, float]:
    return {
        "hotspot": 0.07,
        "underestimation": 0.02,
        "aligned_amplitude": 0.01,
        "orthogonal_error": 0.01,
    }


def test_s3_loads_embedded_ema_as_new_base(tmp_path: Path) -> None:
    source = torch.nn.Linear(3, 2)
    with torch.no_grad():
        source.weight.fill_(1.0)
        source.bias.fill_(2.0)
    ema = EMA(source, decay=0.9)
    with torch.no_grad():
        source.weight.fill_(4.0)
        source.bias.fill_(5.0)
    ema.update(source)
    checkpoint = tmp_path / "epoch_0589.pt"
    save_checkpoint(
        checkpoint,
        source,
        ema=ema,
        epoch=589,
        global_step=123,
        config=_source_config(),
        extra={"calibrated_auxiliary_weights": _auxiliary()},
    )
    target = torch.nn.Linear(3, 2)
    provenance = load_ema_as_base_initialization(
        checkpoint,
        target,
        config=_source_config(),
        expected_epoch=589,
    )
    for key, value in target.state_dict().items():
        assert torch.equal(value, ema.module.state_dict()[key])
    assert provenance["source_epoch"] == 589
    assert provenance["optimizer_restored"] is False
    assert provenance["calibrated_auxiliary_weights"] == _auxiliary()


def test_s3_rejects_incomplete_source_auxiliary_weights() -> None:
    with pytest.raises(RuntimeError):
        validated_auxiliary_weights({"hotspot": 0.1})


def test_s3_mse_weight_calibration_and_warmup() -> None:
    weight, audit = solve_mse_weight([0.5] * 32)
    assert weight == pytest.approx(0.2)
    assert audit["target_weighted_ratio"] == 0.10
    assert mse_weight_for_epoch(0, weight, 10) == pytest.approx(0.02)
    assert mse_weight_for_epoch(9, weight, 10) == pytest.approx(0.2)
    assert mse_weight_for_epoch(79, weight, 10) == pytest.approx(0.2)


def test_s3_collects_graph_connected_raw_mse_ratio() -> None:
    torch.manual_seed(17)
    model = WaveletResidualBridgeModel(
        channels=(4, 8),
        condition_dim=8,
        num_timesteps=10,
        prediction_target="residual_x0",
        loss_kwargs={"high_band_stds": torch.ones(7), "image_domain": "raw"},
        corrector_kwargs={"enabled": False},
    )
    batch = {
        "mri": torch.rand(1, 1, 8, 8, 8),
        "pet": torch.rand(1, 1, 8, 8, 8),
        "case_id": ["train-only-case"],
    }
    result = collect_mse_gradient_ratios(
        model,
        [batch],
        torch.tensor(0.1),
        torch.ones(7) * 0.1,
        device=torch.device("cpu"),
        maximum_batches=1,
        t_sampling="uniform_internal",
        endpoint_probability=0.0,
    )
    assert len(result["ratios"]) == 1
    assert result["ratios"][0] > 0
    assert result["case_ids"] == ["train-only-case"]


def test_s3_guarded_psnr_selection_is_fail_closed() -> None:
    baseline = {
        "msssim3d": 0.875,
        "hotspot_mae": 0.14,
        "raw_out_of_range_ratio": 0.09,
    }
    selection = {
        "target_psnr": 24.0,
        "msssim_max_drop": 0.001,
        "hotspot_mae_max_ratio": 1.02,
        "raw_out_of_range_max_increase": 0.002,
    }
    good = {
        "psnr": 24.1,
        "mae": 0.034,
        "msssim3d": 0.8741,
        "hotspot_mae": 0.142,
        "raw_out_of_range_ratio": 0.091,
    }
    best: dict = {}
    improved, checks = update_refinement_best(best, good, baseline, selection, 14)
    assert all(checks.values())
    assert "psnr_guarded" in improved
    assert "psnr_target_guarded" in improved
    bad = dict(good, psnr=24.5, msssim3d=0.870)
    improved, checks = update_refinement_best(best, bad, baseline, selection, 19)
    assert not checks["msssim_guard"]
    assert "psnr_guarded" not in improved
    assert best["psnr_guarded"]["epoch"] == 14
    assert refinement_constraints(bad, baseline, selection) == checks
