#!/usr/bin/env python3
"""Summarize calibration from a frozen CHERISH prediction table."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


TASK_C_CLASSES = [
    "IHC2/FISH_negative",
    "IHC2/FISH_positive",
    "IHC3",
]

TASK_C_PROB_COLS = [
    "task_c_prob_ihc2_fish_negative",
    "task_c_prob_ihc2_fish_positive",
    "task_c_prob_ihc3",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def binary_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def multiclass_log_loss(y_index: np.ndarray, probs: np.ndarray) -> float:
    probs = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    return float(-np.log(probs[np.arange(len(y_index)), y_index]).mean())


def auc_rank(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    sorted_score = score[order]
    start = 0
    while start < len(score):
        end = start
        while end + 1 < len(score) and sorted_score[end + 1] == sorted_score[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end + 2) / 2.0
        start = end + 1

    pos_rank_sum = ranks[y == 1].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ece_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    rows = []
    for idx in range(n_bins):
        left = idx / n_bins
        right = (idx + 1) / n_bins
        if idx == n_bins - 1:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p >= left) & (p < right)
        n = int(mask.sum())
        if n:
            mean_pred = float(p[mask].mean())
            observed = float(y[mask].mean())
            abs_gap = abs(observed - mean_pred)
        else:
            mean_pred = float("nan")
            observed = float("nan")
            abs_gap = float("nan")
        rows.append(
            {
                "bin": idx + 1,
                "bin_left": left,
                "bin_right": right,
                "n": n,
                "mean_prediction": mean_pred,
                "observed_fraction": observed,
                "abs_gap": abs_gap,
            }
        )
    return pd.DataFrame(rows)


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = ece_bins(y, p, n_bins)
    n_total = bins["n"].sum()
    if n_total == 0:
        return float("nan")
    return float(((bins["n"] / n_total) * bins["abs_gap"].fillna(0)).sum())


def mce(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = ece_bins(y, p, n_bins)
    if (bins["n"] > 0).sum() == 0:
        return float("nan")
    return float(bins.loc[bins["n"] > 0, "abs_gap"].max())


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), float("nan")

    x = logit(p)
    X = np.column_stack([np.ones(len(x)), x])
    prevalence = np.clip(y.mean(), 1e-6, 1 - 1e-6)
    beta = np.array([math.log(prevalence / (1 - prevalence)), 1.0], dtype=float)

    for _ in range(100):
        mu = sigmoid(X @ beta)
        w = np.clip(mu * (1 - mu), 1e-9, None)
        hessian = X.T @ (w[:, None] * X)
        gradient = X.T @ (y - mu)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        beta += step
        if float(np.max(np.abs(step))) < 1e-8:
            break

    return float(beta[0]), float(beta[1])


def summarize_binary(endpoint: str, y: np.ndarray, p: np.ndarray) -> dict[str, float | str | int]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    intercept, slope = calibration_intercept_slope(y, p)
    return {
        "endpoint": endpoint,
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "event_rate_or_accuracy": float(y.mean()),
        "mean_prediction": float(p.mean()),
        "mean_prediction_minus_observed": float(p.mean() - y.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "ece_10bin": ece(y, p),
        "mce_10bin": mce(y, p),
        "log_loss": binary_log_loss(y, p),
        "roc_auc": auc_rank(y, p),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    n_boot: int,
    seed: int,
) -> list[dict[str, float | str]]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    rows = []
    for name, fn in metrics.items():
        values = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            try:
                val = fn(y[idx], p[idx])
            except Exception:
                val = float("nan")
            if np.isfinite(val):
                values.append(val)
        if values:
            arr = np.asarray(values)
            rows.append(
                {
                    "metric": name,
                    "bootstrap_n": len(arr),
                    "ci_lower": float(np.quantile(arr, 0.025)),
                    "ci_upper": float(np.quantile(arr, 0.975)),
                }
            )
    return rows


def add_reliability_rows(
    rows: list[pd.DataFrame], endpoint: str, y: np.ndarray, p: np.ndarray
) -> None:
    out = ece_bins(y, p, n_bins=10)
    out.insert(0, "endpoint", endpoint)
    rows.append(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("04_supplementary_results/01_external_validation/predictions/ensemble_slide_predictions_labeled.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/locked_calibration_summary"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args()

    project_root = Path.cwd()
    input_path = args.input if args.input.is_absolute() else project_root / args.input
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    probs = df[TASK_C_PROB_COLS].to_numpy(dtype=float)
    y_index = df["task_c_label_name"].map({label: idx for idx, label in enumerate(TASK_C_CLASSES)}).to_numpy()
    confidence = probs.max(axis=1)
    pred_index = probs.argmax(axis=1)
    correct = (pred_index == y_index).astype(int)

    metric_rows = []
    reliability_rows: list[pd.DataFrame] = []
    ci_rows = []

    top_summary = summarize_binary("task_c_top_class_confidence", correct, confidence)
    top_summary["multiclass_log_loss"] = multiclass_log_loss(y_index, probs)
    top_summary["multiclass_brier"] = float(
        np.mean(np.sum((probs - np.eye(len(TASK_C_CLASSES))[y_index]) ** 2, axis=1))
    )
    metric_rows.append(top_summary)
    add_reliability_rows(reliability_rows, "task_c_top_class_confidence", correct, confidence)
    for row in bootstrap_ci(
        correct,
        confidence,
        {
            "accuracy": lambda y, p: float(np.mean(y)),
            "mean_confidence": lambda y, p: float(np.mean(p)),
            "ece_10bin": ece,
            "brier": lambda y, p: float(np.mean((p - y) ** 2)),
            "calibration_slope": lambda y, p: calibration_intercept_slope(y, p)[1],
        },
        args.n_bootstrap,
        args.seed,
    ):
        row["endpoint"] = "task_c_top_class_confidence"
        ci_rows.append(row)

    for label, prob_col in zip(TASK_C_CLASSES, TASK_C_PROB_COLS):
        y = (df["task_c_label_name"] == label).astype(int).to_numpy()
        p = df[prob_col].to_numpy(dtype=float)
        endpoint = f"task_c_one_vs_rest__{label}"
        metric_rows.append(summarize_binary(endpoint, y, p))
        add_reliability_rows(reliability_rows, endpoint, y, p)

    subset = df[df["task_c_binary_subset_mask"] == 1].copy()
    subset_y = (subset["task_c_binary_subset_label_name"] == "IHC2/FISH_positive").astype(int).to_numpy()
    subset_p = subset["task_c_binary_subset_prob_positive"].to_numpy(dtype=float)
    subset_endpoint = "task_c_ihc2_fish_binary_subset"
    metric_rows.append(summarize_binary(subset_endpoint, subset_y, subset_p))
    add_reliability_rows(reliability_rows, subset_endpoint, subset_y, subset_p)
    for row in bootstrap_ci(
        subset_y,
        subset_p,
        {
            "roc_auc": auc_rank,
            "brier": lambda y, p: float(np.mean((p - y) ** 2)),
            "ece_10bin": ece,
            "log_loss": binary_log_loss,
            "calibration_slope": lambda y, p: calibration_intercept_slope(y, p)[1],
        },
        args.n_bootstrap,
        args.seed + 1,
    ):
        row["endpoint"] = subset_endpoint
        ci_rows.append(row)

    task_a_y = (df["task_a_label_name"] == "HER2_positive").astype(int).to_numpy()
    task_a_p = df["task_a_prob_positive"].to_numpy(dtype=float)
    metric_rows.append(summarize_binary("task_a_auxiliary_head", task_a_y, task_a_p))
    add_reliability_rows(reliability_rows, "task_a_auxiliary_head", task_a_y, task_a_p)

    task_b = df[df["task_b_mask"] == 1].copy()
    task_b_y = (task_b["task_b_label_name"] == "IHC2/FISH_positive").astype(int).to_numpy()
    task_b_p = task_b["task_b_prob_positive"].to_numpy(dtype=float)
    metric_rows.append(summarize_binary("task_b_auxiliary_head", task_b_y, task_b_p))
    add_reliability_rows(reliability_rows, "task_b_auxiliary_head", task_b_y, task_b_p)

    center_rows = []
    for center, sub in df.groupby("external_center", dropna=False):
        sub_probs = sub[TASK_C_PROB_COLS].to_numpy(dtype=float)
        sub_y_index = sub["task_c_label_name"].map({label: idx for idx, label in enumerate(TASK_C_CLASSES)}).to_numpy()
        sub_confidence = sub_probs.max(axis=1)
        sub_correct = (sub_probs.argmax(axis=1) == sub_y_index).astype(int)
        row = summarize_binary(f"task_c_top_class_confidence__{center}", sub_correct, sub_confidence)
        row["center"] = center
        row["analysis_endpoint"] = "task_c_top_class_confidence"
        center_rows.append(row)

        sub_subset = sub[sub["task_c_binary_subset_mask"] == 1].copy()
        if len(sub_subset):
            sub_y = (sub_subset["task_c_binary_subset_label_name"] == "IHC2/FISH_positive").astype(int).to_numpy()
            sub_p = sub_subset["task_c_binary_subset_prob_positive"].to_numpy(dtype=float)
            row = summarize_binary(f"task_c_ihc2_fish_binary_subset__{center}", sub_y, sub_p)
            row["center"] = center
            row["analysis_endpoint"] = "task_c_ihc2_fish_binary_subset"
            center_rows.append(row)

    summary = pd.DataFrame(metric_rows)
    summary.to_csv(output_dir / "locked_calibration_metric_summary.csv", index=False)

    reliability = pd.concat(reliability_rows, ignore_index=True)
    reliability.to_csv(output_dir / "locked_calibration_reliability_bins_10bin.csv", index=False)

    pd.DataFrame(center_rows).to_csv(output_dir / "locked_calibration_center_stratified.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(output_dir / "locked_calibration_bootstrap_ci.csv", index=False)

    manifest = pd.DataFrame(
        [
            {
                "input_file": str(args.input),
                "n_rows": int(len(df)),
                "output_dir": str(args.output_dir),
                "ece_bins": 10,
                "bootstrap_resamples_for_key_endpoints": int(args.n_bootstrap),
                "bootstrap_seed": int(args.seed),
                "model_or_threshold_changed": "no",
                "external_label_dependent_recalibration": "no",
            }
        ]
    )
    manifest.to_csv(output_dir / "locked_calibration_summary_manifest.csv", index=False)

    print(f"Wrote locked calibration summary outputs to {output_dir}")


if __name__ == "__main__":
    main()
