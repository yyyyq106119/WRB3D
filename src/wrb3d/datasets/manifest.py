"""Patient-level split auditing and immutable JSON manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


def infer_patient_id(case_id: str | Path, explicit: str | None = None, regex: str | None = None) -> str:
    if explicit:
        return str(explicit)
    text = Path(str(case_id)).stem
    if regex:
        match = re.search(regex, text)
        if match:
            return match.group(1) if match.groups() else match.group(0)
    match = re.search(r"\d{5,}", text)
    if match:
        return match.group(0)
    return re.split(r"[_\.\s-]+", text)[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_patient_splits(split_samples: dict[str, list[dict]]) -> dict[str, object]:
    patient_sets: dict[str, set[str]] = {}
    hashes: dict[str, str] = {}
    identity_records: list[dict[str, object]] = []
    for split, samples in split_samples.items():
        patients: set[str] = set()
        for sample in samples:
            patient = str(sample["patient_id"])
            patients.add(patient)
            identity_row: dict[str, object] = {
                "split": split,
                "patient_id": patient,
                "case_id": str(sample.get("case_id", "")),
            }
            for key in ("mri_path", "pet_path"):
                path = sample.get(key)
                if path is None:
                    identity_row[f"{key}_sha256"] = None
                    continue
                path = Path(path)
                if path.exists():
                    digest = _sha256(path)
                    owner = hashes.get(digest)
                    if owner is not None and owner != split:
                        raise RuntimeError(f"duplicate file content crosses {owner}/{split}: {path}")
                    hashes[digest] = split
                    identity_row[f"{key}_sha256"] = digest
                else:
                    identity_row[f"{key}_sha256"] = None
            identity_records.append(identity_row)
        patient_sets[split] = patients
    names = sorted(patient_sets)
    overlaps = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            shared = sorted(patient_sets[left] & patient_sets[right])
            overlaps[f"{left}__{right}"] = shared
            if shared:
                raise RuntimeError(f"patient leakage across {left}/{right}: {shared[:10]}")
    identity_text = json.dumps(
        sorted(
            identity_records,
            key=lambda row: (str(row["split"]), str(row["patient_id"]), str(row["case_id"])),
        ),
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "patients_per_split": {name: len(values) for name, values in patient_sets.items()},
        "overlaps": overlaps,
        "content_hash_count": len(hashes),
        "dataset_identity_sha256": hashlib.sha256(identity_text.encode("utf-8")).hexdigest(),
    }


def write_split_manifests(
    split_samples: dict[str, list[dict]], output_dir: str | Path
) -> dict[str, object]:
    audit = audit_patient_splits(split_samples)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split, samples in split_samples.items():
        patients = sorted({str(sample["patient_id"]) for sample in samples})
        (output / f"{split}_patients.json").write_text(
            json.dumps(patients, indent=2), encoding="utf-8"
        )
    (output / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def verify_split_manifests(
    split_samples: dict[str, list[dict]], manifest_dir: str | Path
) -> dict[str, object]:
    """Fail closed if fixed manifests are absent, leaky, or stale."""
    output = Path(manifest_dir)
    expected = {split: {str(row["patient_id"]) for row in rows} for split, rows in split_samples.items()}
    recorded: dict[str, set[str]] = {}
    for split in split_samples:
        path = output / f"{split}_patients.json"
        if not path.exists():
            raise FileNotFoundError(
                f"required patient manifest is missing: {path}. Run wrb3d-build-manifests first."
            )
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, list):
            raise ValueError(f"patient manifest must contain a JSON list: {path}")
        recorded[split] = {str(value) for value in values}
        if recorded[split] != expected[split]:
            missing = sorted(expected[split] - recorded[split])
            extra = sorted(recorded[split] - expected[split])
            raise RuntimeError(
                f"stale {split} patient manifest; missing={missing[:10]}, extra={extra[:10]}"
            )
    names = sorted(recorded)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            shared = recorded[left] & recorded[right]
            if shared:
                raise RuntimeError(f"patient leakage across {left}/{right}: {sorted(shared)[:10]}")
    current = audit_patient_splits(split_samples)
    audit_path = output / "split_audit.json"
    if not audit_path.exists():
        raise FileNotFoundError(
            f"required content-identity manifest is missing: {audit_path}. "
            "Run wrb3d-build-manifests again."
        )
    saved = json.loads(audit_path.read_text(encoding="utf-8"))
    if saved.get("dataset_identity_sha256") != current["dataset_identity_sha256"]:
        raise RuntimeError(
            "stale split content manifest: dataset files, pairing, or patient identity changed"
        )
    return current
