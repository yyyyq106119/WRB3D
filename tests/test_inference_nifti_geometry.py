import sys

import nibabel as nib
import numpy as np
import torch
import yaml

from wrb3d.inference.cli import main
from wrb3d.training import save_checkpoint
from wrb3d.utils import build_model, load_config


def test_nifti_inference_preserves_input_geometry(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    affine = np.array(
        [[2.0, 0.0, 0.0, -12.0], [0.0, 2.5, 0.0, 8.0], [0.0, 0.0, 3.0, 4.0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    volume = np.linspace(0, 1, 8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8)
    source = input_dir / "P12345_MRI_preprocessed.nii.gz"
    nib.save(nib.Nifti1Image(volume, affine), source)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "root": str(input_dir),
                    "normalization": {"mode": "preprocessed_0_1"},
                    "require_patient_manifests": False,
                },
                "wavelet": {
                    "type": "fixed_orthonormal_haar3d",
                    "levels": 1,
                    "band_order": ["LLL", "HLL", "LHL", "HHL", "LLH", "HLH", "LHH", "HHH"],
                    "pad_mode": "replicate",
                },
                "bridge": {"num_timesteps": 4, "schedule": "linear"},
                "covariance": {
                    "statistics_path": None,
                    "allow_unverified_for_smoke": True,
                    "low": 0.05,
                    "high": [0.05] * 7,
                },
                "model": {
                    "input_channels": 1,
                    "channels": [4, 8],
                    "condition_dim": 16,
                    "prediction_target": "residual_x0",
                    "low_to_high_condition": "feature_gating",
                },
                "projection": {"mode": "none", "beta": 10.0},
                "loss": {"high_band_stds": [1.0] * 7},
                "train": {"ema_decay": 0.9},
            }
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    torch.manual_seed(73)
    model = build_model(config)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, config=config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wrb3d-infer",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--steps",
            "5",
            "--device",
            "cpu",
        ],
    )
    main()
    generated = nib.load(str(output_dir / "P12345_pet_raw.nii.gz"))
    assert generated.shape == volume.shape
    assert np.allclose(generated.affine, affine)
    with np.load(output_dir / "P12345.npz", allow_pickle=False) as archive:
        assert np.allclose(archive["source_affine"], affine)
        assert archive["pet_raw"].shape == volume.shape
