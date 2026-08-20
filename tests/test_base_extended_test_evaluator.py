from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "evaluate_base_extended_test.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_base_extended_test", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_tensor_endpoint_metrics_are_exact_for_equal_tensors():
    target = torch.tensor([[[[[1.0, 2.0, 3.0, 4.0]]]]])
    metrics = MODULE._tensor_endpoint_metrics(target.clone(), target)
    assert metrics["mae"] == 0.0
    assert abs(metrics["pearson"] - 1.0) < 1e-6
    assert abs(metrics["slope"] - 1.0) < 1e-6
    assert abs(metrics["energy_ratio"] - 1.0) < 1e-6


def test_patient_safe_summary_averages_cases_within_patient_first():
    rows = [
        {
            "case_id": "a1",
            "patient_id": "a",
            "sampling_steps": 5,
            "mae": 1.0,
        },
        {
            "case_id": "a2",
            "patient_id": "a",
            "sampling_steps": 5,
            "mae": 3.0,
        },
        {
            "case_id": "b1",
            "patient_id": "b",
            "sampling_steps": 5,
            "mae": 6.0,
        },
    ]
    summary = MODULE._patient_safe_summary(rows)["5"]
    assert summary["case_count"] == 3
    assert summary["patient_count"] == 2
    assert summary["metrics"]["mae"] == 4.0


def test_finite_rows_rejects_nonfinite_metric():
    rows = [
        {
            "case_id": "x",
            "patient_id": "x",
            "sampling_steps": 1,
            "mae": float("nan"),
        }
    ]
    assert MODULE._finite_rows(rows) is False


def test_semantic_fingerprint_is_key_order_independent():
    left = {"b": [2, 3], "a": {"x": 1}}
    right = {"a": {"x": 1}, "b": [2, 3]}
    assert MODULE._semantic_fingerprint(left) == MODULE._semantic_fingerprint(right)


def test_core_semantics_ignore_formal_e2_bookkeeping_only():
    core = {key: {"value": key} for key in MODULE.CORE_SEMANTIC_KEYS}
    source = {
        **core,
        "initialization": None,
        "run_contract": {"mode": "formal_20epoch"},
        "runtime_approval": {"artifact": "old"},
        "ema": {"enabled": True},
    }
    receiving = {
        **core,
        "initialization": {"mode": "different"},
        "run_contract": {"mode": "diagnostic_320epoch"},
        "runtime_approval": None,
        "ema": {"enabled": False},
    }
    assert MODULE._core_semantics(source) == MODULE._core_semantics(receiving)


def test_core_semantics_detect_loss_change():
    source = {key: {"value": key} for key in MODULE.CORE_SEMANTIC_KEYS}
    receiving = {key: {"value": key} for key in MODULE.CORE_SEMANTIC_KEYS}
    receiving["loss"] = {"value": "changed"}
    differences = MODULE._semantic_differences(
        MODULE._core_semantics(source), MODULE._core_semantics(receiving)
    )
    assert [row["path"] for row in differences] == ["loss.value"]


def test_pairwise_numeric_string_and_float_are_semantically_equal():
    left = {"loss": {"eps": 1e-6, "variance_floor": 1e-8}}
    right = {"loss": {"eps": "1e-06", "variance_floor": "1e-08"}}
    normalized_left, normalized_right, paths = (
        MODULE._pairwise_numeric_equivalence(left, right)
    )
    assert normalized_left == normalized_right
    assert paths == ["loss.eps", "loss.variance_floor"]


def test_pairwise_different_numeric_values_are_not_equal():
    left = {"loss": {"eps": 1e-6}}
    right = {"loss": {"eps": "2e-06"}}
    normalized_left, normalized_right, paths = (
        MODULE._pairwise_numeric_equivalence(left, right)
    )
    assert normalized_left != normalized_right
    assert paths == []
