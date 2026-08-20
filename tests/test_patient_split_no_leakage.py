import json

import pytest

from wrb3d.datasets import (
    audit_patient_splits,
    verify_split_manifests,
    write_split_manifests,
)


def test_patient_split_no_leakage():
    clean = {
        "train": [{"patient_id": "1"}],
        "val": [{"patient_id": "2"}],
        "test": [{"patient_id": "3"}],
    }
    assert audit_patient_splits(clean)["patients_per_split"] == {"train": 1, "val": 1, "test": 1}
    leaking = {"train": [{"patient_id": "1"}], "val": [{"patient_id": "1"}]}
    with pytest.raises(RuntimeError, match="patient leakage"):
        audit_patient_splits(leaking)


def test_patient_manifests_fail_closed_when_stale(tmp_path):
    samples = {
        "train": [{"patient_id": "1"}],
        "val": [{"patient_id": "2"}],
        "test": [{"patient_id": "3"}],
    }
    write_split_manifests(samples, tmp_path)
    assert verify_split_manifests(samples, tmp_path)["content_hash_count"] == 0
    (tmp_path / "val_patients.json").write_text(json.dumps(["9"]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale val patient manifest"):
        verify_split_manifests(samples, tmp_path)
