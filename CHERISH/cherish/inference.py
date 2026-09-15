"""Locked CHERISH ensemble inference utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .dataset import read_feature_bag
from .metrics import pathway_readout_values
from .model import MultitaskHER2MILHier

TASK_C_LABELS = {
    0: "IHC2/FISH_negative",
    1: "IHC2/FISH_positive",
    2: "IHC3",
}


def discover_checkpoints(checkpoint_root: str | Path) -> list[dict[str, Path]]:
    """Discover run directories containing a checkpoint and model config."""

    root = Path(checkpoint_root)
    discovered: list[dict[str, Path]] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        checkpoint_path = run_dir / "best_by_val_task_c_macro_auc.pt"
        config_path = run_dir / "config.json"
        if checkpoint_path.exists() and config_path.exists():
            discovered.append(
                {
                    "run_name": run_dir.name,
                    "checkpoint_path": checkpoint_path,
                    "config_path": config_path,
                }
            )
    return discovered


def read_checkpoint_manifest(path: str | Path) -> list[dict[str, Path]]:
    """Read an explicit checkpoint manifest CSV."""

    df = pd.read_csv(path)
    required = {"checkpoint_path", "config_path"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Checkpoint manifest is missing columns: {missing}")

    checkpoints: list[dict[str, Path]] = []
    for index, row in df.iterrows():
        checkpoints.append(
            {
                "run_name": str(row.get("run_name", row.get("fold", f"checkpoint_{index}"))),
                "checkpoint_path": Path(row["checkpoint_path"]),
                "config_path": Path(row["config_path"]),
            }
        )
    return checkpoints


def _infer_input_dim(state_dict: dict[str, torch.Tensor]) -> int:
    return int(state_dict["input_proj.0.weight"].shape[1])


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | Path,
    device: str | torch.device,
) -> tuple[MultitaskHER2MILHier, dict]:
    """Load a CHERISH checkpoint with model parameters from JSON config."""

    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint_payload, dict) and "state_dict" in checkpoint_payload:
        state_dict = checkpoint_payload["state_dict"]
    else:
        state_dict = checkpoint_payload
    config = _load_json(config_path)
    model_kwargs = dict(config.get("model", {}))
    model = MultitaskHER2MILHier(input_dim=_infer_input_dim(state_dict), **model_kwargs)
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("task_c_alpha.") and not key.startswith("task_c_beta.")
    }
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, config


def sample_feature_bag(
    features: np.ndarray,
    coords: np.ndarray,
    *,
    max_tiles: int | None,
    sampling: str,
    seed: int,
    dataset_index: int,
    repeat_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply validation-style tile sampling."""

    if max_tiles is None or features.shape[0] <= int(max_tiles):
        indices = np.arange(features.shape[0], dtype=int)
    elif sampling == "random":
        rng = np.random.default_rng(int(seed) + int(repeat_index) * 1000 + int(dataset_index) * 17)
        indices = np.asarray(rng.choice(features.shape[0], size=int(max_tiles), replace=False), dtype=int)
    else:
        indices = np.arange(int(max_tiles), dtype=int)
    return features[indices], coords[indices]


def infer_single_slide_from_arrays(
    model: MultitaskHER2MILHier,
    features: np.ndarray,
    coords: np.ndarray,
    device: str | torch.device,
) -> dict[str, np.ndarray]:
    """Run one model on one feature bag."""

    feats_tensor = torch.from_numpy(features.astype(np.float32)).to(device)
    coords_tensor = torch.from_numpy(coords.astype(np.float32)).to(device)
    with torch.inference_mode():
        out = model(feats_tensor, coords=coords_tensor, return_features=True)

    return {
        "coords": coords.astype(np.float32),
        "attention": out["task_c_attn"].detach().cpu().numpy().astype(np.float32),
        "task_a_prob": torch.sigmoid(out["task_a_logit"]).detach().cpu().numpy().astype(np.float32),
        "task_b_prob": torch.sigmoid(out["task_b_logit"]).detach().cpu().numpy().astype(np.float32),
        "task_c_prob": torch.softmax(out["task_c_logit"], dim=0).detach().cpu().numpy().astype(np.float32),
        "task_c_alpha_logit": torch.atleast_1d(out["task_c_alpha_logit"]).detach().cpu().numpy().astype(np.float32),
        "task_c_beta_logit": torch.atleast_1d(out["task_c_beta_logit"]).detach().cpu().numpy().astype(np.float32),
        "task_c_alpha_prob": torch.sigmoid(out["task_c_alpha_logit"]).detach().cpu().numpy().astype(np.float32),
        "task_c_beta_prob": torch.sigmoid(out["task_c_beta_logit"]).detach().cpu().numpy().astype(np.float32),
        "task_c_base_prob": out["task_c_base_prob"].detach().cpu().numpy().astype(np.float32),
        "task_c_residual_logit": out["task_c_residual_logit"].detach().cpu().numpy().astype(np.float32),
    }


def _mean_and_std(items: list[dict[str, np.ndarray]], key: str) -> tuple[np.ndarray, np.ndarray]:
    values = np.stack([np.atleast_1d(item[key]) for item in items], axis=0)
    return values.mean(axis=0), values.std(axis=0)


def aggregate_fold_outputs(fold_outputs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Equal-weight average across locked checkpoints."""

    if not fold_outputs:
        raise ValueError("fold_outputs must not be empty")
    task_c_prob_mean, task_c_prob_std = _mean_and_std(fold_outputs, "task_c_prob")
    attention_mean, attention_std = _mean_and_std(fold_outputs, "attention")
    task_a_prob_mean, task_a_prob_std = _mean_and_std(fold_outputs, "task_a_prob")
    task_b_prob_mean, task_b_prob_std = _mean_and_std(fold_outputs, "task_b_prob")
    alpha_logit_mean, alpha_logit_std = _mean_and_std(fold_outputs, "task_c_alpha_logit")
    beta_logit_mean, beta_logit_std = _mean_and_std(fold_outputs, "task_c_beta_logit")
    alpha_prob_mean, alpha_prob_std = _mean_and_std(fold_outputs, "task_c_alpha_prob")
    beta_prob_mean, beta_prob_std = _mean_and_std(fold_outputs, "task_c_beta_prob")
    base_prob_mean, base_prob_std = _mean_and_std(fold_outputs, "task_c_base_prob")
    residual_logit_mean, residual_logit_std = _mean_and_std(fold_outputs, "task_c_residual_logit")
    return {
        "coords": fold_outputs[0]["coords"],
        "attention_mean": attention_mean,
        "attention_std": attention_std,
        "task_a_prob_mean": task_a_prob_mean,
        "task_a_prob_std": task_a_prob_std,
        "task_b_prob_mean": task_b_prob_mean,
        "task_b_prob_std": task_b_prob_std,
        "task_c_prob_mean": task_c_prob_mean,
        "task_c_prob_std": task_c_prob_std,
        "task_c_alpha_logit_mean": alpha_logit_mean,
        "task_c_alpha_logit_std": alpha_logit_std,
        "task_c_beta_logit_mean": beta_logit_mean,
        "task_c_beta_logit_std": beta_logit_std,
        "task_c_alpha_prob_mean": alpha_prob_mean,
        "task_c_alpha_prob_std": alpha_prob_std,
        "task_c_beta_prob_mean": beta_prob_mean,
        "task_c_beta_prob_std": beta_prob_std,
        "task_c_base_prob_mean": base_prob_mean,
        "task_c_base_prob_std": base_prob_std,
        "task_c_residual_logit_mean": residual_logit_mean,
        "task_c_residual_logit_std": residual_logit_std,
    }


def _scalar(value: np.ndarray | float | int) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def build_prediction_row(slide_id: str, row: pd.Series, aggregated: dict[str, np.ndarray]) -> dict[str, object]:
    prob = np.asarray(aggregated["task_c_prob_mean"], dtype=np.float32).reshape(-1)
    prob_std = np.asarray(aggregated["task_c_prob_std"], dtype=np.float32).reshape(-1)
    base = np.asarray(aggregated["task_c_base_prob_mean"], dtype=np.float32).reshape(-1)
    residual = np.asarray(aggregated["task_c_residual_logit_mean"], dtype=np.float32).reshape(-1)
    task_a_prob = _scalar(aggregated["task_a_prob_mean"])
    pred_index = int(prob.argmax())
    pathway = pathway_readout_values(
        task_a_prob,
        float(prob[0]),
        float(prob[1]),
        float(prob[2]),
    )
    prediction = {
        "slide_id": slide_id,
        "n_tiles": int(aggregated["coords"].shape[0]),
        "task_a_prob_positive": task_a_prob,
        "task_a_prob_morphology_axis": task_a_prob,
        "task_a_pred_positive": int(task_a_prob >= 0.5),
        "task_a_pred_morphology_axis": int(task_a_prob >= 0.5),
        "task_b_prob_positive": _scalar(aggregated["task_b_prob_mean"]),
        "task_b_pred_positive": int(_scalar(aggregated["task_b_prob_mean"]) >= 0.5),
        "task_c_prob_ihc2_fish_negative": float(prob[0]),
        "task_c_prob_ihc2_fish_positive": float(prob[1]),
        "task_c_prob_ihc3": float(prob[2]),
        "task_c_prob_std_ihc2_fish_negative": float(prob_std[0]),
        "task_c_prob_std_ihc2_fish_positive": float(prob_std[1]),
        "task_c_prob_std_ihc3": float(prob_std[2]),
        "task_c_pred_index": pred_index,
        "task_c_pred_label": TASK_C_LABELS[pred_index],
        "task_c_positive_branch": float(prob[1] + prob[2]),
        "task_c_amplification_strong_branch": float(prob[1] + prob[2]),
        "task_c_alpha_logit": _scalar(aggregated["task_c_alpha_logit_mean"]),
        "task_c_beta_logit": _scalar(aggregated["task_c_beta_logit_mean"]),
        "task_c_alpha_prob_positive_branch": _scalar(aggregated["task_c_alpha_prob_mean"]),
        "task_c_alpha_prob_morphology_axis": _scalar(aggregated["task_c_alpha_prob_mean"]),
        "task_c_beta_prob_ihc2fish_positive_given_positive": _scalar(aggregated["task_c_beta_prob_mean"]),
        "task_c_beta_prob_ihc2fish_positive_given_amplification_strong_branch": _scalar(aggregated["task_c_beta_prob_mean"]),
        "task_c_base_prob_ihc2_fish_negative": float(base[0]),
        "task_c_base_prob_ihc2_fish_positive": float(base[1]),
        "task_c_base_prob_ihc3": float(base[2]),
        "task_c_residual_logit_ihc2_fish_negative": float(residual[0]),
        "task_c_residual_logit_ihc2_fish_positive": float(residual[1]),
        "task_c_residual_logit_ihc3": float(residual[2]),
        "task_a_label_bin": row.get("task_a_label_bin"),
        "task_b_label_bin": row.get("task_b_label_bin"),
        "task_b_mask": row.get("task_b_mask"),
        "task_c_label_index": row.get("task_c_label_index"),
        "task_c_mask": row.get("task_c_mask"),
        "reference_pathway_label": row.get("reference_pathway_label", row.get("pathway_label")),
    }
    prediction.update(pathway)
    return prediction


def run_locked_ensemble_inference(
    manifest_df: pd.DataFrame,
    checkpoints: list[dict[str, Path]],
    output_dir: str | Path,
    device: str | torch.device = "cpu",
    max_slides: int | None = None,
    max_tiles: int | None = None,
    sampling: str = "random",
    seed: int = 42,
    write_per_fold: bool = False,
) -> pd.DataFrame:
    """Run the locked equal-weight ensemble and write prediction CSVs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if max_slides is not None:
        manifest_df = manifest_df.head(int(max_slides)).copy()

    loaded = []
    for item in checkpoints:
        model, config = load_model_from_checkpoint(item["checkpoint_path"], item["config_path"], device)
        loaded.append((str(item["run_name"]), model, config))
    if not loaded:
        raise ValueError("No checkpoints were provided")

    rows: list[dict[str, object]] = []
    per_fold_rows: dict[str, list[dict[str, object]]] = {name: [] for name, _, _ in loaded}
    for dataset_index, (_, row) in enumerate(manifest_df.iterrows()):
        slide_id = str(row["slide_id"])
        feature_bag_path = row.get("feature_bag_path")
        if feature_bag_path is None:
            raise KeyError("Manifest must include a feature_bag_path column")
        features, coords = read_feature_bag(feature_bag_path, require_coords=True)
        assert coords is not None
        sampled_features, sampled_coords = sample_feature_bag(
            features,
            coords,
            max_tiles=max_tiles,
            sampling=sampling,
            seed=seed,
            dataset_index=dataset_index,
        )
        fold_outputs = []
        for run_name, model, _ in loaded:
            fold_output = infer_single_slide_from_arrays(model, sampled_features, sampled_coords, device)
            fold_outputs.append(fold_output)
            if write_per_fold:
                per_fold_rows[run_name].append(build_prediction_row(slide_id, row, aggregate_fold_outputs([fold_output])))
        aggregated = aggregate_fold_outputs(fold_outputs)
        rows.append(build_prediction_row(slide_id, row, aggregated))

    predictions = pd.DataFrame(rows)
    predictions.to_csv(output_dir / "ensemble_slide_predictions.csv", index=False)
    if write_per_fold:
        per_fold_dir = output_dir / "per_fold"
        per_fold_dir.mkdir(parents=True, exist_ok=True)
        for run_name, run_rows in per_fold_rows.items():
            pd.DataFrame(run_rows).to_csv(per_fold_dir / f"{run_name}_slide_predictions.csv", index=False)
    return predictions
