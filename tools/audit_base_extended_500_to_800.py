"""Fail-closed audit for the original Base 500-to-800 continuation experiment.

This tool is deliberately read-only.  It never trains, updates, converts, or
rewrites a checkpoint.  Its only successful outcome is an evidence-backed
READY/NOT READY decision that can be consumed by a separate dry-run gate.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from wrb3d.utils import load_config


EXPERIMENT_NAME = "BASE_EXTENDED_500_TO_800"
EXPECTED_SOURCE_PATH = Path(
    "/data4/wangchangmiao/yyq/WaveletResidualBridge3D/"
    "outputs/stage_a/from_scratch/latest.pt"
)
EXPECTED_SOURCE_SHA256 = (
    "7e9e4965b888d2f0e155c23ca2ce8cab113b2cea070762e2d447e12dfdad24d6"
)
EXPECTED_MODEL_STATE_SHA256 = (
    "3bfb9993d9543e748e47625b936e7d658c1f723a2b99332763719f2c3e08429c"
)
EXPECTED_EPOCH = 499
EXPECTED_GLOBAL_STEP = 39000

BANNED_KEYWORDS = (
    "gate0",
    "gate1",
    "gate2",
    "e1",
    "e2",
    "ampcal",
    "finetune",
    "slope",
    "cancel",
    "cancellation",
    "normalized_residual",
    "residual_norm",
    "canonical",
)

ALLOWED_CONFIG_DIFF_PREFIXES = (
    "_config_path",
    "experiment.name",
    "train.epochs",
    "train.max_epochs",
    "train.output_dir",
    "train.checkpoint_save_interval",
    "train.validate_every",
    "train.validation_interval",
    "train.log_every",
    "train.logging",
)

MODEL_KEYS = ("model", "base_model", "model_state_dict", "state_dict")
EMA_KEYS = ("ema", "ema_model", "model_ema")
OPTIMIZER_KEYS = ("optimizer", "optimizer_state_dict")
SCHEDULER_KEYS = ("scheduler", "lr_scheduler", "scheduler_state_dict")
SCALER_KEYS = ("scaler", "grad_scaler", "amp_scaler")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _first_mapping(payload: Mapping[str, Any], keys: Iterable[str]) -> tuple[str | None, Mapping[str, Any] | None]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return key, value
    return None, None


def _all_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        result: list[str] = []
        for key, item in value.items():
            result.extend(_all_strings(str(key)))
            result.extend(_all_strings(item))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_all_strings(item))
        return result
    return []


def _banned_matches(path: Path, payload: Mapping[str, Any] | None) -> list[str]:
    texts = [str(path).lower()]
    if payload is not None:
        config = payload.get("config")
        semantics = payload.get("semantic_config")
        for value in (config, semantics):
            if isinstance(value, Mapping):
                texts.extend(item.lower() for item in _all_strings(value))
    return sorted(
        {
            keyword
            for keyword in BANNED_KEYWORDS
            if any(keyword in text for text in texts)
        }
    )


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key in sorted(value, key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten(value[key], path))
        return output
    if isinstance(value, tuple):
        value = list(value)
    return {prefix: value}


def config_differences(
    checkpoint_config: Mapping[str, Any] | None,
    original_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if checkpoint_config is None:
        missing = {
            "path": "<checkpoint.config>",
            "checkpoint": "<missing>",
            "original": "<present>",
            "allowed": False,
        }
        return [missing], [missing]
    left = _flatten(checkpoint_config)
    right = _flatten(original_config)
    rows: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for path in sorted(set(left) | set(right)):
        left_value = left.get(path, "<missing>")
        right_value = right.get(path, "<missing>")
        if left_value == right_value:
            continue
        allowed = any(
            path == prefix or path.startswith(f"{prefix}.")
            for prefix in ALLOWED_CONFIG_DIFF_PREFIXES
        )
        row = {
            "path": path,
            "checkpoint": left_value,
            "original": right_value,
            "allowed": allowed,
        }
        rows.append(row)
        if not allowed:
            blockers.append(row)
    return rows, blockers


def _optimizer_learning_rates(state: Mapping[str, Any] | None) -> list[float]:
    if state is None:
        return []
    groups = state.get("param_groups")
    if not isinstance(groups, list):
        return []
    return [
        float(group["lr"])
        for group in groups
        if isinstance(group, Mapping) and isinstance(group.get("lr"), (int, float))
    ]


def _scheduler_metadata(
    key: str | None, state: Mapping[str, Any] | None, config: Mapping[str, Any] | None
) -> dict[str, Any]:
    train = (
        config.get("train", {})
        if isinstance(config, Mapping) and isinstance(config.get("train"), Mapping)
        else {}
    )
    configured = train.get("scheduler")
    return {
        "key": key,
        "type": (
            configured.get("type")
            if isinstance(configured, Mapping)
            else configured
        ),
        "last_epoch": state.get("last_epoch") if state is not None else None,
        "total_steps": (
            state.get("total_steps")
            if state is not None
            else None
        ),
        "T_max": state.get("T_max") if state is not None else None,
    }


def inspect_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    expected_model_sha256: str,
    expected_epoch: int,
    expected_global_step: int,
    original_config: Mapping[str, Any],
    require_scheduler_state: bool = True,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    base_record: dict[str, Any] = {
        "checkpoint_path": str(resolved),
        "exists": resolved.is_file(),
        "file_size": resolved.stat().st_size if resolved.is_file() else None,
        "modified_time": (
            datetime.fromtimestamp(
                resolved.stat().st_mtime, tz=timezone.utc
            ).isoformat()
            if resolved.is_file()
            else None
        ),
    }
    if not resolved.is_file():
        return {
            **base_record,
            "load_error": "checkpoint file is missing",
            "blockers": ["checkpoint_file_missing"],
            "identity_verified": False,
            "allowed_use": False,
        }

    file_hash = sha256_file(resolved)
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
    except Exception as error:  # pragma: no cover - exact torch errors are version-specific.
        return {
            **base_record,
            "checkpoint_sha256": file_hash,
            "load_error": f"{type(error).__name__}: {error}",
            "blockers": ["checkpoint_unreadable"],
            "identity_verified": False,
            "allowed_use": False,
        }
    if not isinstance(payload, Mapping):
        return {
            **base_record,
            "checkpoint_sha256": file_hash,
            "load_error": "checkpoint root is not a mapping",
            "blockers": ["checkpoint_root_not_mapping"],
            "identity_verified": False,
            "allowed_use": False,
        }

    model_key, model_state = _first_mapping(payload, MODEL_KEYS)
    ema_key, _ = _first_mapping(payload, EMA_KEYS)
    optimizer_key, optimizer_state = _first_mapping(payload, OPTIMIZER_KEYS)
    scheduler_key, scheduler_state = _first_mapping(payload, SCHEDULER_KEYS)
    scaler_key, _ = _first_mapping(payload, SCALER_KEYS)
    checkpoint_config = (
        payload.get("config") if isinstance(payload.get("config"), Mapping) else None
    )
    config_path = (
        checkpoint_config.get("_config_path")
        if checkpoint_config is not None
        else None
    )
    git_commit = payload.get("git_commit")
    if git_commit is None and isinstance(payload.get("code_identity"), Mapping):
        git_commit = payload["code_identity"].get("git_commit")
    model_hash = (
        state_dict_sha256(model_state)
        if model_state is not None
        else None
    )
    differences, forbidden_differences = config_differences(
        checkpoint_config, original_config
    )
    banned = _banned_matches(resolved, payload)
    epoch = int(payload.get("epoch", -1))
    global_step = int(payload.get("global_step", -1))
    learning_rates = _optimizer_learning_rates(optimizer_state)
    scheduler = _scheduler_metadata(
        scheduler_key, scheduler_state, checkpoint_config
    )

    checks = {
        "file_sha256_exact": file_hash == expected_sha256,
        "model_state_sha256_exact": model_hash == expected_model_sha256,
        "epoch_499": epoch == expected_epoch,
        "global_step_exact": global_step == expected_global_step,
        "base_model_key_present": model_key in MODEL_KEYS,
        "base_model_key_not_ema": model_key not in EMA_KEYS,
        "optimizer_state_present": optimizer_key is not None,
        "scheduler_state_present": (
            scheduler_key is not None if require_scheduler_state else True
        ),
        "scaler_state_present": scaler_key is not None,
        "checkpoint_config_present": checkpoint_config is not None,
        "config_diff_allowed_only": not forbidden_differences,
        "no_gate_e1_e2_or_finetune_provenance": not banned,
        "learning_rate_present": bool(learning_rates),
        "learning_rate_nonterminal": (
            bool(learning_rates)
            and all(float(value) > 0.0 for value in learning_rates)
        ),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        **base_record,
        "checkpoint_sha256": file_hash,
        "epoch": epoch,
        "global_step": global_step,
        "model_base_key": model_key,
        "model_state_sha256": model_hash,
        "ema_key": ema_key,
        "optimizer_state_present": optimizer_key is not None,
        "scheduler_state_present": scheduler_key is not None,
        "scaler_state_present": scaler_key is not None,
        "random_state_present": any(
            key in payload
            for key in ("rng_state", "random_state", "torch_rng_state")
        ),
        "sampler_state_present": any(
            key in payload for key in ("sampler", "sampler_state")
        ),
        "config_path": config_path,
        "git_commit": git_commit,
        "optimizer_type": (
            checkpoint_config.get("train", {}).get("optimizer")
            if checkpoint_config is not None
            and isinstance(checkpoint_config.get("train"), Mapping)
            else None
        ),
        "current_learning_rates": learning_rates,
        "scheduler": scheduler,
        "banned_provenance_keywords": banned,
        "config_differences": differences,
        "forbidden_config_differences": forbidden_differences,
        "checks": checks,
        "blockers": blockers,
        "identity_verified": all(
            checks[name]
            for name in (
                "file_sha256_exact",
                "model_state_sha256_exact",
                "epoch_499",
                "global_step_exact",
                "base_model_key_present",
                "base_model_key_not_ema",
                "no_gate_e1_e2_or_finetune_provenance",
            )
        ),
        "allowed_use": all(checks.values()),
    }


def _candidate_paths(
    search_roots: Iterable[Path], explicit_checkpoint: Path
) -> list[Path]:
    candidates: set[Path] = {explicit_checkpoint.expanduser()}
    for root in search_roots:
        root = root.expanduser()
        if not root.exists():
            continue
        for pattern in ("*.pt", "*.pth", "*.ckpt"):
            candidates.update(root.rglob(pattern))
    return sorted(candidates, key=lambda item: str(item))


def _write_csv(path: Path, records: list[Mapping[str, Any]]) -> None:
    fields = [
        "checkpoint_path",
        "file_size",
        "modified_time",
        "epoch",
        "global_step",
        "model_base_key",
        "ema_key",
        "optimizer_state_present",
        "scheduler_state_present",
        "scaler_state_present",
        "config_path",
        "git_commit",
        "checkpoint_sha256",
        "banned_provenance_keywords",
        "identity_verified",
        "allowed_use",
        "blockers",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            for key in ("banned_provenance_keywords", "blockers"):
                row[key] = ";".join(str(value) for value in row.get(key, []))
            writer.writerow(row)


def _write_diff_report(
    path: Path, record: Mapping[str, Any] | None
) -> None:
    rows = list(record.get("config_differences", [])) if record else []
    lines = [
        "# Original Base config vs extended config diff",
        "",
        "Only experiment/logging/output/checkpoint cadence/validation cadence/"
        "maximum epoch fields may differ.",
        "",
        "| Path | Checkpoint value | Original config value | Allowed |",
        "|---|---|---|---:|",
    ]
    if not rows:
        lines.append("| — | No auditable checkpoint config | — | False |")
    for row in rows:
        lines.append(
            f"| `{row['path']}` | `{row['checkpoint']}` | "
            f"`{row['original']}` | {row['allowed']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_audit_report(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    selected = payload.get("identity_candidate")
    blockers = payload.get("blockers", [])
    lines = [
        "# BASE_EXTENDED pretrain audit",
        "",
        f"- Experiment: `{EXPERIMENT_NAME}`",
        f"- Status: **{payload['status']}**",
        f"- Candidates inspected: {len(payload['candidates'])}",
        f"- Expected source: `{EXPECTED_SOURCE_PATH}`",
        f"- Expected source SHA256: `{payload['expected_source_sha256']}`",
    ]
    if isinstance(selected, Mapping):
        lines.extend(
            [
                f"- Identity candidate: `{selected['checkpoint_path']}`",
                f"- Observed SHA256: `{selected.get('checkpoint_sha256')}`",
                f"- loaded_weight_type: `base`",
                f"- source_epoch: `{selected.get('epoch')}`",
                f"- source_global_step: `{selected.get('global_step')}`",
                f"- Optimizer present: `{selected.get('optimizer_state_present')}`",
                f"- Scheduler present: `{selected.get('scheduler_state_present')}`",
                f"- AMP scaler present: `{selected.get('scaler_state_present')}`",
                f"- Current LR: `{selected.get('current_learning_rates')}`",
            ]
        )
    lines.extend(["", "## Blockers", ""])
    if blockers:
        lines.extend(f"- `{item}`" for item in blockers)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "READY"
                if payload["status"] == "READY"
                else "NOT READY"
            ),
            "",
        ]
    )
    if payload["status"] != "READY":
        lines.append(
            "BLOCKED: original 500-epoch Base checkpoint identity and complete "
            "training state cannot yet be verified."
        )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    original_config = load_config(args.original_config)
    candidates = [
        inspect_checkpoint(
            path,
            expected_sha256=args.expected_sha256,
            expected_model_sha256=args.expected_model_sha256,
            expected_epoch=args.expected_epoch,
            expected_global_step=args.expected_global_step,
            original_config=original_config,
            require_scheduler_state=not args.allow_missing_scheduler,
        )
        for path in _candidate_paths(args.search_root, args.checkpoint)
    ]
    identity_candidates = [
        row for row in candidates if row.get("identity_verified") is True
    ]
    allowed = [row for row in candidates if row.get("allowed_use") is True]
    identity_candidate = (
        identity_candidates[0] if len(identity_candidates) == 1 else None
    )
    blockers: list[str] = []
    if len(identity_candidates) != 1:
        blockers.append(
            f"expected_exactly_one_identity_candidate_observed_{len(identity_candidates)}"
        )
    if identity_candidate is not None:
        blockers.extend(str(value) for value in identity_candidate.get("blockers", []))
    if len(allowed) != 1:
        blockers.append(f"expected_exactly_one_allowed_candidate_observed_{len(allowed)}")
    blockers.append("dry_run_not_executed")
    status = "NOT READY"
    payload = {
        "schema_version": 1,
        "kind": "base_extended_500_to_800_pretrain_audit",
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "expected_source_path": str(args.checkpoint),
        "expected_source_sha256": args.expected_sha256,
        "expected_model_state_sha256": args.expected_model_sha256,
        "expected_epoch": args.expected_epoch,
        "expected_global_step": args.expected_global_step,
        "require_scheduler_state": not args.allow_missing_scheduler,
        "identity_candidate": identity_candidate,
        "allowed_candidates": allowed,
        "blockers": sorted(set(blockers)),
        "candidates": candidates,
    }
    (output / "checkpoint_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "checkpoint_audit.csv", candidates)
    _write_diff_report(
        output / "ORIGINAL_BASE_CONFIG_VS_EXTENDED_CONFIG_DIFF.md",
        identity_candidate,
    )
    _write_audit_report(
        output / "BASE_EXTENDED_PRETRAIN_AUDIT.md",
        payload,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=EXPECTED_SOURCE_PATH,
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        action="append",
        default=[],
        help="recursively inspect checkpoint files under this root; repeatable",
    )
    parser.add_argument(
        "--original-config",
        type=Path,
        default=Path("configs/train/stage_a_4gpu.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/BASE_EXTENDED_500_TO_800"),
    )
    parser.add_argument("--expected-sha256", default=EXPECTED_SOURCE_SHA256)
    parser.add_argument(
        "--expected-model-sha256", default=EXPECTED_MODEL_STATE_SHA256
    )
    parser.add_argument("--expected-epoch", type=int, default=EXPECTED_EPOCH)
    parser.add_argument(
        "--expected-global-step", type=int, default=EXPECTED_GLOBAL_STEP
    )
    parser.add_argument(
        "--allow-missing-scheduler",
        action="store_true",
        help=(
            "diagnostic only; never use this for the prescribed continuation "
            "experiment"
        ),
    )
    return parser.parse_args()


def main() -> None:
    payload = run_audit(parse_args())
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    raise SystemExit(0 if payload["status"] == "READY" else 2)


if __name__ == "__main__":
    main()
