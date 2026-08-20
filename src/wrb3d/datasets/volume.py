"""Self-contained paired NPZ/NIfTI readers and a separate MRI-only contract."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import infer_patient_id

_MRI_SUFFIXES = ("_MRI_preprocessed", "_Reg_MRI", "_MRI_reference", "_mri", "_MRI")
_PET_SUFFIXES = ("_PET_preprocessed", "_PET_in_MRI", "_Reg_PET", "_pet", "_PET")


def _is_nifti(path: Path) -> bool:
    return path.name.lower().endswith((".nii", ".nii.gz"))


def _stem(path: Path) -> str:
    name = path.name
    return name[:-7] if name.lower().endswith(".nii.gz") else path.stem


def _case_id(path: Path, kind: str) -> str:
    stem = _stem(path)
    suffixes = _MRI_SUFFIXES if kind == "mri" else _PET_SUFFIXES
    for suffix in suffixes:
        if stem.lower().endswith(suffix.lower()):
            return stem[: -len(suffix)]
    if path.parent.name.lower() not in {"train", "val", "test", "test_mri"}:
        return path.parent.name
    return stem


def _score(path: Path, kind: str) -> int:
    name = path.name.lower()
    if any(token in name for token in ("mask", "roi", "seg", "label", "preview")):
        return -10**9
    if kind == "mri":
        return sum(
            weight for token, weight in (("mri_preprocessed", 1000), ("reg_mri", 900), ("mri", 500))
            if token in name
        ) - (1000 if "pet" in name else 0)
    return sum(
        weight for token, weight in (("pet_preprocessed", 1000), ("pet_in_mri", 900), ("pet", 500))
        if token in name
    ) - (700 if "mri" in name and "pet_in_mri" not in name else 0)


def _stage_root(root: str | Path, stage: str) -> Path:
    root = Path(root)
    nested = root / stage
    return nested if nested.is_dir() else root


def discover_pairs(root: str | Path, stage: str) -> list[dict[str, Any]]:
    actual = _stage_root(root, stage)
    if not actual.exists():
        return []
    nifti = [path for path in actual.rglob("*") if path.is_file() and _is_nifti(path)]
    by_parent: dict[Path, list[Path]] = {}
    for path in nifti:
        by_parent.setdefault(path.parent, []).append(path)
    pairs: list[dict[str, Any]] = []
    for parent, paths in sorted(by_parent.items(), key=lambda item: str(item[0])):
        mri_paths = [path for path in paths if _score(path, "mri") > 0]
        pet_paths = [path for path in paths if _score(path, "pet") > 0]
        pet_by_case: dict[str, list[Path]] = {}
        for path in pet_paths:
            pet_by_case.setdefault(_case_id(path, "pet").lower(), []).append(path)
        for mri_path in sorted(mri_paths):
            case = _case_id(mri_path, "mri")
            candidates = pet_by_case.get(case.lower(), [])
            if not candidates and len(mri_paths) == 1 and len(pet_paths) == 1:
                candidates = pet_paths
            if candidates:
                pet_path = max(candidates, key=lambda value: _score(value, "pet"))
                pairs.append(
                    {
                        "format": "nifti",
                        "case_id": case,
                        "mri_path": mri_path,
                        "pet_path": pet_path,
                    }
                )
    if pairs:
        return sorted(pairs, key=lambda sample: sample["case_id"])
    output = []
    for path in sorted(actual.rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "mri" in archive and "pet" in archive:
                    output.append(
                        {"format": "npz", "case_id": path.stem, "mri_path": path, "pet_path": path}
                    )
        except (OSError, ValueError):
            continue
    return output


def discover_mri(root: str | Path, stage: str = "test") -> list[dict[str, Any]]:
    actual = _stage_root(root, stage)
    if not actual.exists():
        return []
    paths = [
        path
        for path in actual.rglob("*")
        if path.is_file() and _is_nifti(path) and _score(path, "mri") > 0
    ]
    if paths:
        return [
            {
                "format": "nifti",
                "case_id": _case_id(path, "mri"),
                "mri_path": path,
                "pet_path": None,
            }
            for path in sorted(paths)
        ]
    output = []
    for path in sorted(actual.rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "mri" in archive:
                    output.append(
                        {"format": "npz", "case_id": path.stem, "mri_path": path, "pet_path": None}
                    )
        except (OSError, ValueError):
            continue
    return output


def _load_pair(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if sample["format"] == "npz":
        with np.load(sample["mri_path"], allow_pickle=False) as archive:
            mri = archive["mri"].astype(np.float32)
            pet = archive["pet"].astype(np.float32)
            metadata = {
                "patient_id": str(archive["patient_id"].item()) if "patient_id" in archive else None,
                "spacing": archive["spacing"].astype(np.float32) if "spacing" in archive else None,
                "roi": (
                    archive["roi"].astype(np.float32)
                    if "roi" in archive
                    else archive["mask"].astype(np.float32) if "mask" in archive else None
                ),
                "affine": archive["affine"].astype(np.float64) if "affine" in archive else None,
                "orientation": None,
            }
        return mri, pet, metadata
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("NIfTI input requires the optional 'nifti' dependency") from exc
    mri_image = nib.load(str(sample["mri_path"]))
    pet_image = nib.load(str(sample["pet_path"]))
    mri_affine = np.asarray(mri_image.affine, dtype=np.float64)
    pet_affine = np.asarray(pet_image.affine, dtype=np.float64)
    mri_orientation = tuple(str(value) for value in nib.aff2axcodes(mri_affine))
    pet_orientation = tuple(str(value) for value in nib.aff2axcodes(pet_affine))
    mri_spacing = np.asarray(mri_image.header.get_zooms()[:3], dtype=np.float64)
    pet_spacing = np.asarray(pet_image.header.get_zooms()[:3], dtype=np.float64)
    if mri_orientation != pet_orientation:
        raise ValueError(
            f"orientation mismatch for {sample['case_id']}: {mri_orientation} vs {pet_orientation}"
        )
    if not np.allclose(mri_spacing, pet_spacing, rtol=1e-5, atol=1e-5):
        raise ValueError(
            f"registration spacing mismatch for {sample['case_id']}: "
            f"{mri_spacing.tolist()} vs {pet_spacing.tolist()}"
        )
    if not np.allclose(mri_affine, pet_affine, rtol=1e-5, atol=1e-4):
        raise ValueError(f"affine registration mismatch for {sample['case_id']}")
    return (
        mri_image.get_fdata(dtype=np.float32),
        pet_image.get_fdata(dtype=np.float32),
        {
            "patient_id": None,
            "spacing": mri_spacing.astype(np.float32),
            "roi": None,
            "affine": mri_affine,
            "orientation": mri_orientation,
        },
    )


def _load_mri(sample: dict[str, Any]) -> tuple[np.ndarray, str | None, dict[str, Any]]:
    if sample["format"] == "npz":
        with np.load(sample["mri_path"], allow_pickle=False) as archive:
            patient = str(archive["patient_id"].item()) if "patient_id" in archive else None
            metadata = {
                "affine": archive["affine"].astype(np.float64) if "affine" in archive else None,
                "spacing": archive["spacing"].astype(np.float32) if "spacing" in archive else None,
                "source_path": str(sample["mri_path"]),
                "is_nifti": False,
            }
            return archive["mri"].astype(np.float32), patient, metadata
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("NIfTI input requires the optional 'nifti' dependency") from exc
    image = nib.load(str(sample["mri_path"]))
    return image.get_fdata(dtype=np.float32), None, {
        "affine": np.asarray(image.affine, dtype=np.float64),
        "spacing": np.asarray(image.header.get_zooms()[:3], dtype=np.float32),
        "source_path": str(sample["mri_path"]),
        "is_nifti": True,
    }


def _validate_volume(volume: np.ndarray, case: str, kind: str, tolerance: float) -> None:
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError(f"{case} {kind} must be a finite [D,H,W] volume")
    if float(volume.min()) < -tolerance or float(volume.max()) > 1.0 + tolerance:
        raise ValueError(
            f"{case} {kind} is outside expected preprocessed [0,1] range: "
            f"[{float(volume.min()):.5g},{float(volume.max()):.5g}]"
        )


def _crop_pair(
    arrays: list[np.ndarray], patch_size: tuple[int, int, int] | None, random_crop: bool
) -> list[np.ndarray]:
    if patch_size is None:
        return arrays
    shape = arrays[0].shape
    pads = []
    for size, target in zip(shape, patch_size):
        total = max(0, target - size)
        pads.append((total // 2, total - total // 2))
    if any(left or right for left, right in pads):
        arrays = [np.pad(value, pads, mode="edge") for value in arrays]
        shape = arrays[0].shape
    starts = [
        random.randint(0, size - target) if random_crop and size > target else (size - target) // 2
        for size, target in zip(shape, patch_size)
    ]
    slices = tuple(slice(start, start + target) for start, target in zip(starts, patch_size))
    return [value[slices] for value in arrays]


class PairedVolumeDataset(Dataset):
    """Paired training/evaluation data. A is always MRI and B is always PET."""

    def __init__(
        self,
        root: str | Path,
        stage: str = "train",
        patch_size: tuple[int, int, int] | None = None,
        patient_id_regex: str | None = None,
        range_tolerance: float = 1e-3,
    ) -> None:
        self.stage = stage
        self.patch_size = tuple(int(v) for v in patch_size) if patch_size else None
        self.patient_id_regex = patient_id_regex
        self.range_tolerance = float(range_tolerance)
        self.samples = discover_pairs(root, stage)
        if not self.samples:
            raise FileNotFoundError(f"no paired NPZ/NIfTI MRI-PET samples under {root!s} for {stage}")

    def __len__(self) -> int:
        return len(self.samples)

    def manifest_records(self) -> list[dict[str, Any]]:
        """Return split-audit metadata without loading MRI/PET voxel arrays."""
        records: list[dict[str, Any]] = []
        for sample in self.samples:
            explicit = None
            if sample["format"] == "npz":
                with np.load(sample["mri_path"], allow_pickle=False) as archive:
                    if "patient_id" in archive:
                        explicit = str(archive["patient_id"].item())
            records.append(
                {
                    "case_id": sample["case_id"],
                    "patient_id": infer_patient_id(
                        sample["case_id"], explicit, self.patient_id_regex
                    ),
                    "mri_path": sample["mri_path"],
                    "pet_path": sample["pet_path"],
                }
            )
        return records

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        mri, pet, metadata = _load_pair(sample)
        if mri.shape != pet.shape:
            raise ValueError(f"shape mismatch for {sample['case_id']}: {mri.shape} vs {pet.shape}")
        _validate_volume(mri, sample["case_id"], "MRI", self.range_tolerance)
        _validate_volume(pet, sample["case_id"], "PET", self.range_tolerance)
        arrays = [mri, pet]
        has_roi = metadata["roi"] is not None
        if has_roi:
            if metadata["roi"].shape != mri.shape:
                raise ValueError(f"ROI shape mismatch for {sample['case_id']}")
            arrays.append(metadata["roi"])
        arrays = _crop_pair(arrays, self.patch_size, self.stage == "train")
        mri, pet = arrays[:2]
        patient_id = infer_patient_id(
            sample["case_id"], metadata["patient_id"], self.patient_id_regex
        )
        output: dict[str, Any] = {
            "mri": torch.from_numpy(mri[None].copy()),
            "pet": torch.from_numpy(pet[None].copy()),
            "case_id": sample["case_id"],
            "patient_id": patient_id,
            "has_roi": has_roi,
        }
        if has_roi:
            output["roi"] = torch.from_numpy(arrays[2][None].copy())
        return output


class MRIOnlyVolumeDataset(Dataset):
    """Generation dataset whose instances cannot expose PET to inference."""

    def __init__(self, root: str | Path, patient_id_regex: str | None = None) -> None:
        self.patient_id_regex = patient_id_regex
        self.samples = discover_mri(root)
        if not self.samples:
            raise FileNotFoundError(f"no MRI NPZ/NIfTI inputs under {root!s}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        mri, explicit, metadata = _load_mri(sample)
        _validate_volume(mri, sample["case_id"], "MRI", 1e-3)
        output: dict[str, Any] = {
            "mri": torch.from_numpy(mri[None].copy()),
            "case_id": sample["case_id"],
            "patient_id": infer_patient_id(sample["case_id"], explicit, self.patient_id_regex),
            "source_path": metadata["source_path"],
            "is_nifti": metadata["is_nifti"],
        }
        if metadata["affine"] is not None:
            output["affine"] = torch.from_numpy(metadata["affine"].copy())
        if metadata["spacing"] is not None:
            output["spacing"] = torch.from_numpy(metadata["spacing"].copy())
        return output
