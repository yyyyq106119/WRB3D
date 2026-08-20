"""Validation-only paired S1/S2 analysis and Pareto tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

import numpy as np


VIEWS = {
    "best_val_whole_mae": "best_val_whole_mae.pt",
    "best_val_hotspot_mae": "best_val_hotspot_mae.pt",
    "best_val_peak_bias": "best_val_peak_bias.pt",
    "epoch_0799": "epoch_0799.pt",
}


def _checkpoint_epoch(path: Path) -> int:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload["epoch"])


def _patients(run: Path, epoch: int) -> dict[str, dict[str, float]]:
    path = run / f"validation_patients_epoch_{epoch:04d}_base.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {str(row["patient_id"]): row["metrics"] for row in rows}


def _wilcoxon(differences: np.ndarray) -> dict[str, float | str]:
    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
        return {
            "statistic": float(result.statistic),
            "pvalue": float(result.pvalue),
            "method": "scipy_exact_or_approx",
        }
    except ImportError:
        values = differences[differences != 0]
        order = np.argsort(np.abs(values))
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(1, len(values) + 1)
        positive = float(ranks[values > 0].sum())
        negative = float(ranks[values < 0].sum())
        n = len(values)
        mean = n * (n + 1) / 4
        variance = n * (n + 1) * (2 * n + 1) / 24
        z = (positive - mean) / math.sqrt(max(variance, 1e-12))
        return {
            "statistic": min(positive, negative),
            "pvalue": math.erfc(abs(z) / math.sqrt(2.0)),
            "method": "normal_approx_without_tie_correction",
        }


def _paired(
    s1: np.ndarray, s2: np.ndarray, *, higher_is_better: bool, rng: np.random.Generator
) -> dict:
    difference = s1 - s2
    indices = rng.integers(0, len(difference), size=(10_000, len(difference)))
    bootstrap = difference[indices].mean(axis=1)
    improvement = difference if higher_is_better else -difference
    return {
        "mean_s1_minus_s2": float(difference.mean()),
        "median_s1_minus_s2": float(np.median(difference)),
        "win_rate_s1": float((improvement > 0).mean()),
        "paired_bootstrap_95ci_mean_s1_minus_s2": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "wilcoxon": _wilcoxon(difference),
    }


def _metric_vector(
    rows: dict[str, dict[str, float]], metric: str
) -> np.ndarray:
    if metric == "abs_peak_bias":
        return np.array(
            [abs(rows[key]["hotspot_peak_intensity_bias"]) for key in sorted(rows)]
        )
    if metric == "high_mean_orthogonal_error":
        return np.array(
            [
                sum(
                    rows[key][f"high_band_{band}_orthogonal_error_ratio"]
                    for band in range(7)
                )
                / 7
                for key in sorted(rows)
            ]
        )
    if metric == "high_mean_pearson":
        return np.array(
            [
                sum(rows[key][f"high_band_{band}_pearson"] for band in range(7)) / 7
                for key in sorted(rows)
            ]
        )
    return np.array([rows[key][metric] for key in sorted(rows)])


def _load_validation_rows(run: Path, experiment: str) -> list[dict]:
    rows = []
    for line in (run / "validation_metrics.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["model"] == "base":
            rows.append({"experiment": experiment, **row})
    return rows


def _pareto(rows: list[dict]) -> list[dict]:
    fields = (
        ("mae", False),
        ("hotspot_mae", False),
        ("hotspot_peak_intensity_bias_abs", False),
        ("msssim3d", True),
        ("high_mean_orthogonal_error", False),
        ("hotspot_false_positive_energy", False),
    )
    prepared = []
    for row in rows:
        value = dict(row)
        value["hotspot_peak_intensity_bias_abs"] = abs(
            value["hotspot_peak_intensity_bias"]
        )
        prepared.append(value)
    output = []
    for candidate in prepared:
        dominated = False
        for other in prepared:
            if other is candidate:
                continue
            no_worse = all(
                other[field] >= candidate[field]
                if maximize
                else other[field] <= candidate[field]
                for field, maximize in fields
            )
            strictly_better = any(
                other[field] > candidate[field]
                if maximize
                else other[field] < candidate[field]
                for field, maximize in fields
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            output.append(candidate)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--s1", default="outputs/S1_case_adaptive_corrector_hotspot_aligned_800e"
    )
    parser.add_argument(
        "--s2", default="outputs/S2_no_corrector_hotspot_aligned_800e"
    )
    parser.add_argument("--output-dir", default="reports/validation_comparison")
    args = parser.parse_args()
    s1_run = Path(args.s1)
    s2_run = Path(args.s2)
    for run in (s1_run, s2_run):
        if not (run / "FORMAL_TRAINING_COMPLETE.json").is_file():
            raise RuntimeError(f"formal run is not complete: {run}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)
    metrics = {
        "whole_mae": ("mae", False),
        "hotspot_mae": ("hotspot_mae", False),
        "hotspot_underestimation": ("hotspot_underestimation", False),
        "abs_peak_bias": ("abs_peak_bias", False),
        "msssim3d": ("msssim3d", True),
        "high_mean_orthogonal_error": ("high_mean_orthogonal_error", False),
        "high_mean_pearson": ("high_mean_pearson", True),
        "false_positive_hotspot_energy": ("hotspot_false_positive_energy", False),
    }
    comparison: dict[str, dict] = {}
    for view, filename in VIEWS.items():
        s1_epoch = _checkpoint_epoch(s1_run / filename)
        s2_epoch = _checkpoint_epoch(s2_run / filename)
        s1_rows = _patients(s1_run, s1_epoch)
        s2_rows = _patients(s2_run, s2_epoch)
        if set(s1_rows) != set(s2_rows) or len(s1_rows) != 43:
            raise RuntimeError(f"{view} is not a paired fixed-43-patient comparison")
        view_result = {
            "s1_epoch": s1_epoch,
            "s2_epoch": s2_epoch,
            "patient_count": 43,
            "metrics": {},
        }
        for label, (metric, maximize) in metrics.items():
            view_result["metrics"][label] = _paired(
                _metric_vector(s1_rows, metric),
                _metric_vector(s2_rows, metric),
                higher_is_better=maximize,
                rng=rng,
            )
        lower = math.exp(-0.15)
        upper = math.exp(0.15)
        gain_values = np.array(
            [
                s1_rows[patient][f"corrector_gain_band_{band}"]
                for patient in sorted(s1_rows)
                for band in range(7)
            ]
        )
        saturation = float(
            ((gain_values <= lower + 1e-3) | (gain_values >= upper - 1e-3)).mean()
        )
        m = view_result["metrics"]
        whole = m["whole_mae"]
        whole_not_significantly_worse = not (
            whole["mean_s1_minus_s2"] > 0 and whole["wilcoxon"]["pvalue"] < 0.05
        )
        s2_fp_mean = _metric_vector(s2_rows, "hotspot_false_positive_energy").mean()
        s1_fp_mean = _metric_vector(s1_rows, "hotspot_false_positive_energy").mean()
        criteria = {
            "hotspot_mae_decreased": m["hotspot_mae"]["mean_s1_minus_s2"] < 0,
            "hotspot_underestimation_decreased": m["hotspot_underestimation"][
                "mean_s1_minus_s2"
            ]
            < 0,
            "absolute_peak_bias_decreased": m["abs_peak_bias"]["mean_s1_minus_s2"] < 0,
            "whole_mae_not_significantly_worse": whole_not_significantly_worse,
            "msssim_not_decreased": m["msssim3d"]["mean_s1_minus_s2"] >= 0,
            "orthogonal_error_decreased": m["high_mean_orthogonal_error"][
                "mean_s1_minus_s2"
            ]
            < 0,
            "high_pearson_not_worse_by_more_than_0_002": m["high_mean_pearson"][
                "mean_s1_minus_s2"
            ]
            >= -0.002,
            "false_positive_energy_increase_at_most_5pct": s1_fp_mean
            <= 1.05 * max(s2_fp_mean, 1e-12),
            "gain_not_boundary_saturated": saturation < 0.05,
        }
        view_result["gain_boundary_saturation_fraction"] = saturation
        view_result["corrector_criteria"] = criteria
        view_result["corrector_success"] = all(criteria.values())
        comparison[view] = view_result
    all_rows = _load_validation_rows(s1_run, "S1") + _load_validation_rows(s2_run, "S2")
    pareto = _pareto(all_rows)
    pareto_fields = [
        "experiment",
        "epoch",
        "mae",
        "hotspot_mae",
        "hotspot_peak_intensity_bias_abs",
        "msssim3d",
        "high_mean_orthogonal_error",
        "hotspot_false_positive_energy",
    ]
    with (output / "pareto_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=pareto_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in pareto_fields} for row in pareto)
    payload = {
        "schema_version": 1,
        "kind": "S1_vs_S2_validation_only",
        "test_used": False,
        "views": comparison,
        "limitation": (
            "S1 and S2 isolate only the case-adaptive corrector contribution. "
            "Without a same-budget scratch S0, this matrix cannot prove the full "
            "new objective is better than the original scratch objective."
        ),
    }
    (output / "paired_comparison.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Validation Comparison Report",
        "",
        "This report is validation-only; the 88-case test set was not read.",
        "",
        payload["limitation"],
        "",
    ]
    for view, result in comparison.items():
        lines.extend(
            [
                f"## {view}",
                "",
                f"- S1 epoch: {result['s1_epoch']}",
                f"- S2 epoch: {result['s2_epoch']}",
                f"- All nine corrector criteria pass: {result['corrector_success']}",
                "",
            ]
        )
    (output / "VALIDATION_COMPARISON_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
