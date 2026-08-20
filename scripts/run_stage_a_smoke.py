"""Deterministic synthetic Gate-A3 smoke; never presented as clinical evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from wrb3d.models import WaveletResidualBridgeModel
from wrb3d.training import save_checkpoint
from wrb3d.utils import load_config


def _case(seed: int, size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    mri = torch.rand((1, 1, size, size, size), generator=generator)
    grid = torch.linspace(-1, 1, size)
    d, h, w = torch.meshgrid(grid, grid, grid, indexing="ij")
    blob = torch.exp(-5.0 * (d.square() + h.square() + w.square())).reshape(1, 1, size, size, size)
    pet = (0.62 * mri + 0.28 * blob + 0.04 * torch.sin(7.0 * mri)).clamp(0, 1)
    return mri, pet


def _evaluate(model, cases, low_cov, high_cov):
    rows = []
    with torch.no_grad():
        for mri, pet in cases:
            low, high, _ = model.dwt(mri)
            result = model.forward_train(
                mri,
                pet,
                low_cov,
                high_cov,
                t=torch.tensor([model.bridge.num_timesteps // 2]),
                noise_low=torch.zeros_like(low),
                noise_high=torch.zeros_like(high),
            )
            rows.append({name: float(value) for name, value in result["logs"].items() if name.startswith("loss_")})
    keys = rows[0]
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def _train(model, cases, steps, learning_rate, low_cov, high_cov):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start = _evaluate(model, cases, low_cov, high_cov)
    first_gradient = None
    begin = time.perf_counter()
    for step in range(steps):
        mri, pet = cases[step % len(cases)]
        low, high, _ = model.dwt(mri)
        optimizer.zero_grad(set_to_none=True)
        result = model.forward_train(
            mri,
            pet,
            low_cov,
            high_cov,
            t=torch.tensor([model.bridge.num_timesteps // 2]),
            noise_low=torch.zeros_like(low),
            noise_high=torch.zeros_like(high),
        )
        result["loss"].backward()
        if first_gradient is None:
            first_gradient = {
                "low": sum(
                    float(parameter.grad.abs().sum())
                    for parameter in model.low_model.parameters()
                    if parameter.grad is not None
                ),
                "high": sum(
                    float(parameter.grad.abs().sum())
                    for parameter in model.high_model.parameters()
                    if parameter.grad is not None
                ),
            }
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    elapsed = time.perf_counter() - begin
    end = _evaluate(model, cases, low_cov, high_cov)
    return {
        "initial": start,
        "final": end,
        "first_step_gradient_l1": first_gradient,
        "steps": steps,
        "elapsed_seconds": elapsed,
        "mean_step_seconds": elapsed / steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/stage_a_smoke")
    parser.add_argument("--single-steps", type=int, default=80)
    parser.add_argument("--small-set-steps", type=int, default=120)
    args = parser.parse_args()
    torch.manual_seed(19)
    torch.set_num_threads(1)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    low_cov = torch.tensor(0.02)
    high_cov = torch.full((7,), 0.02)
    single_model = WaveletResidualBridgeModel(channels=(4, 8), condition_dim=16, num_timesteps=50)
    single = _train(single_model, [_case(1)], args.single_steps, 5e-3, low_cov, high_cov)
    small_model = WaveletResidualBridgeModel(channels=(4, 8), condition_dim=16, num_timesteps=50)
    cases = [_case(seed) for seed in (2, 3, 4)]
    small = _train(small_model, cases, args.small_set_steps, 3e-3, low_cov, high_cov)
    inference_input = _case(8)[0]
    inference_timings = {}
    inference_results = {}
    for sampling_steps in (5, 15, 50):
        begin = time.perf_counter()
        inference_results[sampling_steps] = single_model.infer(
            inference_input, low_cov, high_cov, num_steps=sampling_steps
        )
        inference_timings[str(sampling_steps)] = time.perf_counter() - begin
    mri_only = inference_results[5]
    report = {
        "scope": "synthetic_cpu_smoke_only",
        "single_batch": single,
        "small_dataset_three_cases": small,
        "mri_only_inference": {
            "shape": list(mri_only["B_raw"].shape),
            "finite": bool(torch.isfinite(mri_only["B_raw"]).all()),
            "sampling_timesteps": mri_only["sampling_timesteps"],
            "cpu_tiny_volume_seconds": inference_timings,
            "timing_scope": "one 8x8x8 case, tiny channels=(4,8), CPU only",
        },
        "parameter_counts_tiny": single_model.parameter_counts(),
    }
    report["single_batch_pass"] = all(
        single["final"][key] < single["initial"][key]
        for key in ("loss_total", "loss_low_residual", "loss_high_residual", "loss_raw_image")
    ) and all(value > 0 for value in single["first_step_gradient_l1"].values())
    report["small_dataset_pass"] = all(
        small["final"][key] < small["initial"][key]
        for key in ("loss_total", "loss_low_residual", "loss_high_residual", "loss_raw_image")
    )
    (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_checkpoint(
        output / "synthetic_smoke.pt",
        single_model,
        config=load_config("configs/model/synthetic_smoke.yaml"),
    )
    mri_only_dir = output / "mri_only_input"
    mri_only_dir.mkdir(exist_ok=True)
    torch_mri = _case(8)[0][0, 0].numpy()
    import numpy as np

    np.savez_compressed(mri_only_dir / "synthetic_mri_only.npz", mri=torch_mri, patient_id="smoke")
    synthetic_root = Path("outputs/synthetic_data")
    for split, seed in (("train", 21), ("val", 22), ("test", 23)):
        split_dir = synthetic_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_mri, split_pet = _case(seed)
        np.savez_compressed(
            split_dir / f"patient{seed}.npz",
            mri=split_mri[0, 0].numpy(),
            pet=split_pet[0, 0].numpy(),
            patient_id=str(seed),
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
