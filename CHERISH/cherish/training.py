"""Training and validation utilities for CHERISH."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from .dataset import MultitaskHER2Dataset, collate_fn_multitask
from .metrics import TASK_C_PROB_COLUMNS
from .model import MultitaskHER2MILHier


def hierarchy_weight_for_epoch(
    epoch: int,
    *,
    start: float = 0.05,
    end: float = 0.20,
    warmup_epochs: int = 6,
) -> float:
    """Linearly warm hierarchy consistency from 0.05 to 0.20, then hold fixed."""

    if warmup_epochs <= 1:
        return float(end)
    progress = min(max(int(epoch), 0), int(warmup_epochs) - 1) / float(int(warmup_epochs) - 1)
    return float(start + (end - start) * progress)


def build_split(
    manifest_df: pd.DataFrame,
    split_df: pd.DataFrame,
    *,
    val_fold: int,
    split_col: str = "fold",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = manifest_df.merge(split_df[["slide_id", split_col]], on="slide_id", how="left")
    if df[split_col].isna().any():
        missing = df[df[split_col].isna()]["slide_id"].head(5).tolist()
        raise ValueError(f"Missing split assignments for slides: {missing}")
    train_df = df[df[split_col].astype(int) != int(val_fold)].reset_index(drop=True)
    val_df = df[df[split_col].astype(int) == int(val_fold)].reset_index(drop=True)
    return train_df, val_df


def infer_input_dim(dataset: MultitaskHER2Dataset) -> int:
    sample = dataset[0][0]
    return int(sample.shape[1])


def binary_pos_weight(labels: pd.Series) -> torch.Tensor:
    labels = labels.astype(float)
    pos = float(labels.sum())
    neg = float(len(labels) - pos)
    return torch.tensor(1.0 if pos <= 0 else neg / pos, dtype=torch.float32)


def multiclass_weight(indices: pd.Series, num_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(indices.astype(int), minlength=num_classes).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    weights = counts.sum() / counts
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def compute_hierarchy_loss(out: dict[str, torch.Tensor], task_b_mask: torch.Tensor) -> torch.Tensor:
    p_a = torch.sigmoid(out["task_a_logit"]).squeeze()
    p_b = torch.sigmoid(out["task_b_logit"]).squeeze()
    p_c = torch.softmax(out["task_c_logit"], dim=0)
    loss = F.mse_loss(p_a, p_c[1] + p_c[2])
    if float(task_b_mask.detach().cpu().item()) > 0.0:
        denom = (p_c[1] + p_c[2]).clamp_min(1e-6)
        loss = loss + F.mse_loss(p_b, p_c[1] / denom)
    return loss


def run_epoch(
    model: MultitaskHER2MILHier,
    loader: DataLoader,
    device: str | torch.device,
    optimizer: torch.optim.Optimizer | None,
    criterion_a: torch.nn.Module,
    criterion_b: torch.nn.Module,
    criterion_c: torch.nn.Module,
    *,
    task_a_weight: float = 0.35,
    task_b_weight: float = 0.35,
    task_c_weight: float = 1.0,
    hierarchy_weight: float = 0.2,
) -> dict[str, float | None]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_items = 0
    y_c: list[int] = []
    p_c: list[list[float]] = []
    y_a: list[int] = []
    p_a: list[float] = []
    y_b: list[int] = []
    p_b: list[float] = []

    for batch in loader:
        features, coords, labels_a, labels_b, labels_c, mask_b, mask_c, _ = batch
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        batch_loss = torch.zeros((), device=device)
        with torch.set_grad_enabled(is_train):
            for i, feats in enumerate(features):
                feats = feats.to(device)
                slide_coords = coords[i].to(device) if coords[i] is not None else None
                label_a = labels_a[i].to(device).view(1)
                label_b = labels_b[i].to(device).view(1)
                label_c = labels_c[i].to(device).view(1)
                b_mask = mask_b[i].to(device).view(1)
                c_mask = mask_c[i].to(device).view(1)

                out = model(feats, coords=slide_coords, return_features=False)
                loss_a = criterion_a(out["task_a_logit"].view(1), label_a)
                loss_b = criterion_b(out["task_b_logit"].view(1), label_b) if float(b_mask.item()) > 0 else out["task_b_logit"].new_zeros(())
                if float(c_mask.item()) > 0:
                    loss_c = criterion_c(out["task_c_logit"].view(1, -1), label_c.long())
                    hierarchy_loss = compute_hierarchy_loss(out, b_mask)
                else:
                    loss_c = out["task_c_logit"].new_zeros(())
                    hierarchy_loss = out["task_c_logit"].new_zeros(())
                loss = task_a_weight * loss_a + task_b_weight * loss_b + task_c_weight * loss_c + hierarchy_weight * hierarchy_loss
                batch_loss = batch_loss + loss

                with torch.no_grad():
                    y_a.append(int(label_a.detach().cpu().item()))
                    p_a.append(float(torch.sigmoid(out["task_a_logit"]).detach().cpu().item()))
                    if float(b_mask.item()) > 0:
                        y_b.append(int(label_b.detach().cpu().item()))
                        p_b.append(float(torch.sigmoid(out["task_b_logit"]).detach().cpu().item()))
                    if float(c_mask.item()) > 0:
                        y_c.append(int(label_c.detach().cpu().item()))
                        p_c.append(torch.softmax(out["task_c_logit"], dim=0).detach().cpu().numpy().astype(float).tolist())
                total_items += 1

            batch_loss = batch_loss / max(1, len(features))
            if optimizer is not None:
                batch_loss.backward()
                optimizer.step()
        total_loss += float(batch_loss.detach().cpu().item()) * len(features)

    metrics: dict[str, float | None] = {"loss": total_loss / max(1, total_items)}
    if y_c:
        y_c_arr = np.asarray(y_c, dtype=int)
        p_c_arr = np.asarray(p_c, dtype=float)
        pred_c = p_c_arr.argmax(axis=1)
        metrics["task_c_macro_f1"] = float(f1_score(y_c_arr, pred_c, average="macro", zero_division=0))
        metrics["task_c_balanced_accuracy"] = float(balanced_accuracy_score(y_c_arr, pred_c))
        try:
            metrics["task_c_macro_auc"] = float(roc_auc_score(y_c_arr, p_c_arr, multi_class="ovr", average="macro"))
        except ValueError:
            metrics["task_c_macro_auc"] = None
    if y_a and len(np.unique(y_a)) > 1:
        metrics["task_a_auc"] = float(roc_auc_score(np.asarray(y_a), np.asarray(p_a)))
    if y_b and len(np.unique(y_b)) > 1:
        metrics["task_b_auc"] = float(roc_auc_score(np.asarray(y_b), np.asarray(p_b)))
    return metrics


def train_one_fold(
    cfg: dict,
    manifest_df: pd.DataFrame,
    split_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    val_fold: int,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    train_df, val_df = build_split(
        manifest_df,
        split_df,
        val_fold=val_fold,
        split_col=data_cfg.get("split_col", "fold"),
    )
    train_ds = MultitaskHER2Dataset(
        train_df,
        max_tiles=data_cfg.get("max_tiles"),
        sampling=data_cfg.get("sampling", "random"),
        seed=data_cfg.get("seed", 42),
        return_coords=data_cfg.get("return_coords", True),
    )
    val_ds = MultitaskHER2Dataset(
        val_df,
        max_tiles=data_cfg.get("max_tiles"),
        sampling=data_cfg.get("sampling", "random"),
        seed=data_cfg.get("seed", 42),
        return_coords=data_cfg.get("return_coords", True),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_fn_multitask,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_fn_multitask,
    )

    model = MultitaskHER2MILHier(input_dim=infer_input_dim(train_ds), **model_cfg).to(device)
    criterion_a = torch.nn.BCEWithLogitsLoss(pos_weight=binary_pos_weight(train_df["task_a_label_bin"]).to(device))
    task_b_train = train_df[train_df["task_b_mask"].astype(float) > 0]
    criterion_b = torch.nn.BCEWithLogitsLoss(pos_weight=binary_pos_weight(task_b_train["task_b_label_bin"]).to(device))
    task_c_train = train_df[train_df["task_c_mask"].astype(float) > 0]
    criterion_c = torch.nn.CrossEntropyLoss(weight=multiclass_weight(task_c_train["task_c_label_index"]).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    best_metric = -np.inf
    best_epoch: int | None = None
    epochs_without_improvement = 0
    patience = train_cfg.get("early_stop_patience")
    patience = None if patience is None else int(patience)
    best_path = output_dir / "best_by_val_task_c_macro_auc.pt"
    history = []
    for epoch in range(int(train_cfg.get("epochs", 20))):
        train_ds.set_epoch(epoch)
        hierarchy_weight = hierarchy_weight_for_epoch(
            epoch,
            start=float(train_cfg.get("hierarchy_weight_start", 0.05)),
            end=float(train_cfg.get("hierarchy_weight", 0.20)),
            warmup_epochs=int(train_cfg.get("hierarchy_weight_warmup_epochs", 6)),
        )
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            criterion_a,
            criterion_b,
            criterion_c,
            hierarchy_weight=hierarchy_weight,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            None,
            criterion_a,
            criterion_b,
            criterion_c,
            hierarchy_weight=hierarchy_weight,
        )
        row = {
            "epoch": epoch,
            "hierarchy_loss_weight_epoch": hierarchy_weight,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        score = val_metrics.get("task_c_macro_auc")
        if score is not None and float(score) > best_metric:
            best_metric = float(score)
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_without_improvement += 1
        if patience is not None and epochs_without_improvement >= patience:
            break

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    config_out = {"model": model_cfg, "data": data_cfg, "training": train_cfg, "val_fold": int(val_fold)}
    (output_dir / "config.json").write_text(json.dumps(config_out, indent=2), encoding="utf-8")
    return {
        "best_metric": None if best_metric == -np.inf else best_metric,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "checkpoint_path": str(best_path),
    }
