"""Metric helpers for CHERISH prediction tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

EPS_STABLE = 1e-6

TASK_C_PROB_COLUMNS = [
    "task_c_prob_ihc2_fish_negative",
    "task_c_prob_ihc2_fish_positive",
    "task_c_prob_ihc3",
]

TASK_C_LABELS = ["IHC2/FISH_negative", "IHC2/FISH_positive", "IHC3"]

PATHWAY_LABELS = [
    "negative_side_likely",
    "boundary_likely_non_amplified",
    "boundary_likely_amplified",
    "strong_positive_like",
]

PATHWAY_MASS_COLUMNS = [
    "pathway_mass_negative_side_likely",
    "pathway_mass_boundary_likely_non_amplified",
    "pathway_mass_boundary_likely_amplified",
    "pathway_mass_strong_positive_like",
]

PATHWAY_REFERENCE_ORDER = [
    "IHC0/1+",
    "IHC2/FISH_negative",
    "IHC2/FISH_positive",
    "IHC3",
]

PATHWAY_REFERENCE_ALIASES = {
    "Negative-side IHC0/1+": "IHC0/1+",
    "HER2_negative_side": "IHC0/1+",
    "IHC0": "IHC0/1+",
    "IHC 0": "IHC0/1+",
    "IHC1": "IHC0/1+",
    "IHC1+": "IHC0/1+",
    "IHC 1+": "IHC0/1+",
    "IHC0/1+": "IHC0/1+",
    "IHC 0/1+": "IHC0/1+",
    "IHC2/FISH_negative": "IHC2/FISH_negative",
    "IHC2/FISH-": "IHC2/FISH_negative",
    "IHC 2+/FISH-": "IHC2/FISH_negative",
    "IHC2/FISH-negative": "IHC2/FISH_negative",
    "IHC 2+/FISH-negative": "IHC2/FISH_negative",
    "IHC 2+/FISH−": "IHC2/FISH_negative",
    "IHC2/FISH_non_amplified": "IHC2/FISH_negative",
    "IHC2/FISH non-amplified": "IHC2/FISH_negative",
    "IHC2/FISH_positive": "IHC2/FISH_positive",
    "IHC2/FISH+": "IHC2/FISH_positive",
    "IHC 2+/FISH+": "IHC2/FISH_positive",
    "IHC2/FISH-positive": "IHC2/FISH_positive",
    "IHC 2+/FISH-positive": "IHC2/FISH_positive",
    "IHC2/FISH_amplified": "IHC2/FISH_positive",
    "IHC2/FISH amplified": "IHC2/FISH_positive",
    "IHC3": "IHC3",
    "IHC3+": "IHC3",
    "IHC 3+": "IHC3",
}

PATHWAY_EXPECTED_READOUT = {
    "IHC0/1+": "negative_side_likely",
    "IHC2/FISH_negative": "boundary_likely_non_amplified",
    "IHC2/FISH_positive": "boundary_likely_amplified",
    "IHC3": "strong_positive_like",
}

PATHWAY_CLINICALLY_MATCHED_READOUT = {
    "IHC0/1+": "negative-side mass-profile readout",
    "IHC2/FISH_negative": "conditional boundary-amplification score < 0.5",
    "IHC2/FISH_positive": "conditional boundary-amplification score >= 0.5",
    "IHC3": "strong-positive boundary-state evidence",
}

STATE_PRED_TO_PATHWAY_LABEL = {
    0: "boundary_likely_non_amplified",
    1: "boundary_likely_amplified",
    2: "strong_positive_like",
}


def pathway_readout_values(
    task_a_prob_positive: float,
    task_c_prob_ihc2_fish_negative: float,
    task_c_prob_ihc2_fish_positive: float,
    task_c_prob_ihc3: float,
    *,
    eps: float = EPS_STABLE,
) -> dict[str, float | int | str]:
    """Compute H&E pathway scores and the descriptive mass-profile label."""

    alpha = float(task_a_prob_positive)
    c_negative = float(task_c_prob_ihc2_fish_negative)
    c_positive = float(task_c_prob_ihc2_fish_positive)
    c_ihc3 = float(task_c_prob_ihc3)
    masses = np.asarray(
        [
            1.0 - alpha,
            alpha * c_negative,
            alpha * c_positive,
            alpha * c_ihc3,
        ],
        dtype=float,
    )
    pred_index = int(masses.argmax())
    return {
        "pathway_score_negative": float(1.0 - alpha),
        "pathway_score_boundary": float(alpha * (c_negative + c_positive)),
        "pathway_score_boundary_amp": float(c_positive / (c_negative + c_positive + float(eps))),
        "pathway_score_strong": float(alpha * c_ihc3),
        "pathway_mass_negative_side_likely": float(masses[0]),
        "pathway_mass_boundary_likely_non_amplified": float(masses[1]),
        "pathway_mass_boundary_likely_amplified": float(masses[2]),
        "pathway_mass_strong_positive_like": float(masses[3]),
        "pathway_pred_index": pred_index,
        "pathway_pred_label": PATHWAY_LABELS[pred_index],
    }


def add_pathway_readouts(df: pd.DataFrame, *, eps: float = EPS_STABLE) -> pd.DataFrame:
    """Add reportable pathway scores from polarity and state probabilities."""

    required = {"task_a_prob_positive", *TASK_C_PROB_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")

    out = df.copy()
    alpha = out["task_a_prob_positive"].astype(float).to_numpy()
    c_negative = out["task_c_prob_ihc2_fish_negative"].astype(float).to_numpy()
    c_positive = out["task_c_prob_ihc2_fish_positive"].astype(float).to_numpy()
    c_ihc3 = out["task_c_prob_ihc3"].astype(float).to_numpy()

    out["pathway_score_negative"] = 1.0 - alpha
    out["pathway_score_boundary"] = alpha * (c_negative + c_positive)
    out["pathway_score_boundary_amp"] = c_positive / (c_negative + c_positive + float(eps))
    out["pathway_score_strong"] = alpha * c_ihc3

    mass = np.column_stack(
        [
            1.0 - alpha,
            alpha * c_negative,
            alpha * c_positive,
            alpha * c_ihc3,
        ]
    )
    for index, column in enumerate(PATHWAY_MASS_COLUMNS):
        out[column] = mass[:, index]
    pred_index = mass.argmax(axis=1)
    out["pathway_pred_index"] = pred_index.astype(int)
    out["pathway_pred_label"] = [PATHWAY_LABELS[int(index)] for index in pred_index]
    out["pathway_pred_label_mass_profile"] = out["pathway_pred_label"]

    state_prob = np.column_stack([c_negative, c_positive, c_ihc3])
    state_pred_index = state_prob.argmax(axis=1)
    out["pathway_state_pred_index"] = state_pred_index.astype(int)
    out["pathway_state_pred_label"] = [TASK_C_LABELS[int(index)] for index in state_pred_index]
    out["pathway_boundary_amplification_pred"] = (out["pathway_score_boundary_amp"].astype(float) >= 0.5).astype(int)
    out["pathway_strong_positive_evidence_pred"] = (state_pred_index == 2).astype(int)
    return out


def _canonical_pathway_reference(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return PATHWAY_REFERENCE_ALIASES.get(text, text if text in PATHWAY_EXPECTED_READOUT else None)


def pathway_readout_summary(
    df: pd.DataFrame,
    *,
    reference_col: str = "reference_pathway_label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize manuscript-aligned, clinically matched pathway readouts.

    The four reportable rows are not a native flat four-class argmax. They follow
    the clinical denominator used for the manuscript summaries: IHC0/1+ cases are
    evaluated by the negative-side mass-profile readout, IHC2/FISH cases by the
    conditional boundary-amplification score, and IHC3 cases by the final
    boundary-state strong-positive evidence.
    """

    if reference_col not in df.columns:
        raise ValueError(f"Prediction table is missing columns: ['{reference_col}']")

    eval_df = add_pathway_readouts(df)
    eval_df["reference_pathway_label_canonical"] = eval_df[reference_col].map(_canonical_pathway_reference)
    eval_df = eval_df[eval_df["reference_pathway_label_canonical"].notna()].copy()
    if eval_df.empty:
        raise ValueError(f"No mappable reference labels found in column: {reference_col}")

    rows = []
    matrix_rows: list[tuple[str, str]] = []
    for reference_label in PATHWAY_REFERENCE_ORDER:
        expected = PATHWAY_EXPECTED_READOUT[reference_label]
        subset = eval_df[eval_df["reference_pathway_label_canonical"] == reference_label]
        total = int(len(subset))
        if total:
            if reference_label == "IHC0/1+":
                predicted = subset["pathway_pred_label"].astype(str)
            elif reference_label == "IHC2/FISH_negative":
                predicted = pd.Series(
                    np.where(
                        subset["pathway_score_boundary_amp"].astype(float).to_numpy() >= 0.5,
                        "boundary_likely_amplified",
                        "boundary_likely_non_amplified",
                    ),
                    index=subset.index,
                )
            elif reference_label == "IHC2/FISH_positive":
                predicted = pd.Series(
                    np.where(
                        subset["pathway_score_boundary_amp"].astype(float).to_numpy() >= 0.5,
                        "boundary_likely_amplified",
                        "boundary_likely_non_amplified",
                    ),
                    index=subset.index,
                )
            else:
                predicted = subset["pathway_state_pred_index"].astype(int).map(STATE_PRED_TO_PATHWAY_LABEL)
            matched = int((predicted == expected).sum())
            matrix_rows.extend((reference_label, str(label)) for label in predicted)
        else:
            matched = 0
        rows.append(
            {
                "reference_label": reference_label,
                "readout_used": PATHWAY_CLINICALLY_MATCHED_READOUT[reference_label],
                "expected_readout": expected,
                "matched": matched,
                "total": total,
                "matched_total": f"{matched}/{total}",
                "percent": float(matched / total * 100.0) if total else np.nan,
            }
        )
    summary_df = pd.DataFrame(rows)

    matrix_df = pd.DataFrame(
        0,
        index=[f"true_{label}" for label in PATHWAY_REFERENCE_ORDER],
        columns=[f"pred_{label}" for label in PATHWAY_LABELS],
        dtype=int,
    )
    for reference_label, predicted_label in matrix_rows:
        matrix_df.loc[f"true_{reference_label}", f"pred_{predicted_label}"] += 1
    return summary_df, matrix_df


def pathway_mass_profile_summary(
    df: pd.DataFrame,
    *,
    reference_col: str = "reference_pathway_label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize the descriptive four-mass profile by argmax.

    This helper is useful for auditing probability-mass routing, but it should
    not be confused with the clinically matched manuscript readout summary.
    """

    if reference_col not in df.columns:
        raise ValueError(f"Prediction table is missing columns: ['{reference_col}']")

    eval_df = add_pathway_readouts(df)
    eval_df["reference_pathway_label_canonical"] = eval_df[reference_col].map(_canonical_pathway_reference)
    eval_df = eval_df[eval_df["reference_pathway_label_canonical"].notna()].copy()
    if eval_df.empty:
        raise ValueError(f"No mappable reference labels found in column: {reference_col}")

    rows = []
    for reference_label in PATHWAY_REFERENCE_ORDER:
        expected = PATHWAY_EXPECTED_READOUT[reference_label]
        subset = eval_df[eval_df["reference_pathway_label_canonical"] == reference_label]
        total = int(len(subset))
        matched = int((subset["pathway_pred_label"] == expected).sum()) if total else 0
        rows.append(
            {
                "reference_label": reference_label,
                "expected_readout": expected,
                "matched": matched,
                "total": total,
                "matched_total": f"{matched}/{total}",
                "percent": float(matched / total * 100.0) if total else np.nan,
            }
        )
    summary_df = pd.DataFrame(rows)

    matrix_df = pd.DataFrame(
        0,
        index=[f"true_{label}" for label in PATHWAY_REFERENCE_ORDER],
        columns=[f"pred_{label}" for label in PATHWAY_LABELS],
        dtype=int,
    )
    for reference_label, predicted_label in zip(
        eval_df["reference_pathway_label_canonical"],
        eval_df["pathway_pred_label"],
    ):
        matrix_df.loc[f"true_{reference_label}", f"pred_{predicted_label}"] += 1
    return summary_df, matrix_df


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        if right < 1.0:
            mask = (confidences >= left) & (confidences < right)
        else:
            mask = (confidences >= left) & (confidences <= right)
        if np.any(mask):
            ece += abs(float(accuracies[mask].mean()) - float(confidences[mask].mean())) * float(mask.mean())
    return float(ece)


def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    target = np.eye(y_prob.shape[1], dtype=float)[y_true]
    return float(np.mean(np.sum((y_prob - target) ** 2, axis=1)))


def task_c_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"task_c_label_index", *TASK_C_PROB_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")

    if "task_c_mask" in df.columns:
        eval_df = df[df["task_c_mask"].astype(float) > 0].copy()
    else:
        eval_df = df.copy()
    y_true = eval_df["task_c_label_index"].astype(int).to_numpy()
    y_prob = eval_df[TASK_C_PROB_COLUMNS].astype(float).to_numpy()
    y_pred = y_prob.argmax(axis=1)

    metrics = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier": multiclass_brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
    }
    try:
        metrics["macro_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        metrics["macro_auc"] = np.nan
    metric_df = pd.DataFrame([metrics])

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in TASK_C_LABELS],
        columns=[f"pred_{label}" for label in TASK_C_LABELS],
    )
    return metric_df, cm_df


def binary_metrics(df: pd.DataFrame, label_col: str, score_col: str, mask_col: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {label_col, score_col}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")

    eval_df = df.copy()
    if mask_col and mask_col in eval_df.columns:
        eval_df = eval_df[eval_df[mask_col].astype(float) > 0].copy()
    y_true = eval_df[label_col].astype(int).to_numpy()
    y_score = eval_df[score_col].astype(float).to_numpy()
    y_pred = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "n": int(len(y_true)),
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan
    metric_df = pd.DataFrame([metrics])
    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["true_negative", "true_positive"],
        columns=["pred_negative", "pred_positive"],
    )
    return metric_df, cm_df


def write_metric_outputs(metric_df: pd.DataFrame, cm_df: pd.DataFrame, output_dir: str | Path, prefix: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_df.to_csv(output_dir / f"{prefix}_metrics.csv", index=False)
    cm_df.to_csv(output_dir / f"{prefix}_confusion_matrix.csv")
