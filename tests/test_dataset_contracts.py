import numpy as np

from wrb3d.datasets import MRIOnlyVolumeDataset, PairedVolumeDataset


def test_paired_and_mri_only_dataset_contracts(tmp_path):
    for split in ("train", "test"):
        directory = tmp_path / split
        directory.mkdir()
        np.savez_compressed(
            directory / "patient12345.npz",
            mri=np.zeros((8, 8, 8), np.float32),
            pet=np.ones((8, 8, 8), np.float32),
            patient_id="12345",
        )
    paired = PairedVolumeDataset(tmp_path, "train", patch_size=(8, 8, 8))[0]
    mri_only = MRIOnlyVolumeDataset(tmp_path / "test")[0]
    assert set(paired) >= {"mri", "pet", "case_id", "patient_id"}
    assert "pet" not in mri_only
    assert paired["mri"].shape == (1, 8, 8, 8)

