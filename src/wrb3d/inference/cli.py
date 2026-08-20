"""MRI-only command-line inference; PET is never loaded by this module."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..datasets import MRIOnlyVolumeDataset
from ..training import EMA, load_checkpoint
from ..utils import build_model, load_config, load_covariances


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PET from MRI only")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, choices=(5, 15, 50), default=15)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument(
        "--allow-checkpoint-mismatch",
        action="store_true",
        help="explicit legacy conversion override; do not use for formal evaluation",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config).to(device)
    ema = EMA(model, float(config["train"].get("ema_decay", 0.9999)))
    checkpoint_info = load_checkpoint(
        args.checkpoint,
        model,
        ema=ema,
        config=config,
        allow_semantic_mismatch=args.allow_checkpoint_mismatch,
        map_location=device,
    )
    inference_model = ema.module if checkpoint_info["ema_loaded"] else model
    inference_model.eval()
    low_covariance, high_covariance, _ = load_covariances(config)
    dataset = MRIOnlyVolumeDataset(args.input, config["data"].get("patient_id_regex"))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    for batch in loader:
        mri = batch["mri"].to(device)
        result = inference_model.infer(
            mri,
            low_covariance.to(device),
            high_covariance.to(device),
            num_steps=args.steps,
            stochastic=args.stochastic,
            generator=generator,
        )
        case = batch["case_id"][0]
        arrays = {
            "pet_raw": result["B_raw"][0, 0].cpu().numpy(),
            "pet_clamp": result["B_clamp"][0, 0].cpu().numpy(),
            "pet_soft": result["B_soft"][0, 0].cpu().numpy(),
        }
        affine = batch.get("affine")
        affine_array = affine[0].cpu().numpy() if affine is not None else None
        archive = {
            **arrays,
            "sampling_timesteps": np.asarray(result["sampling_timesteps"]),
            "seed": args.seed,
            "stochastic": args.stochastic,
            "checkpoint_semantic_mismatch": checkpoint_info["semantic_mismatch"],
            "checkpoint_semantic_fingerprint": checkpoint_info["source_semantic_fingerprint"] or "",
        }
        if affine_array is not None:
            archive["source_affine"] = affine_array
        np.savez_compressed(output / f"{case}.npz", **archive)
        if bool(batch["is_nifti"][0]):
            import nibabel as nib

            source = nib.load(batch["source_path"][0])
            header = source.header.copy()
            for name, array in arrays.items():
                nib.save(
                    nib.Nifti1Image(array.astype(np.float32), affine_array, header=header),
                    output / f"{case}_{name}.nii.gz",
                )


if __name__ == "__main__":
    main()
