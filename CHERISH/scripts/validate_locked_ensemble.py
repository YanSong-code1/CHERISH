#!/usr/bin/env python3
"""Run locked CHERISH validation and write metric summaries.

This wrapper performs locked ensemble inference followed by metric summaries.
It does not recalibrate, refit thresholds, or tune model outputs using
validation or external-test labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Deidentified labeled slide manifest CSV.")
    parser.add_argument("--checkpoint-manifest", type=Path, default=None, help="CSV with run_name, checkpoint_path, config_path.")
    parser.add_argument("--checkpoint-root", type=Path, default=None, help="Root containing one subdirectory per checkpoint run.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-slides", type=int, default=None)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--sampling", type=str, default="random", choices=["random", "first"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-per-fold", action="store_true")
    parser.add_argument("--skip-task-a", action="store_true", help="Do not compute task A binary metrics.")
    parser.add_argument("--skip-task-b", action="store_true", help="Do not compute task B binary metrics.")
    parser.add_argument("--skip-task-c", action="store_true", help="Do not compute task C multiclass metrics.")
    parser.add_argument("--skip-pathway", action="store_true", help="Do not compute four-category pathway readout summaries.")
    parser.add_argument("--pathway-reference-col", type=str, default="reference_pathway_label")
    return parser.parse_args()


def _write_available_metrics(
    predictions,
    output_dir: Path,
    *,
    skip_task_a: bool,
    skip_task_b: bool,
    skip_task_c: bool,
    skip_pathway: bool,
    pathway_reference_col: str,
) -> list[str]:
    from cherish.metrics import binary_metrics, pathway_readout_summary, task_c_metrics, write_metric_outputs

    metric_dir = output_dir / "metrics"
    written: list[str] = []
    if not skip_task_c:
        metric_df, cm_df = task_c_metrics(predictions)
        write_metric_outputs(metric_df, cm_df, metric_dir, "task_c")
        written.append("task_c")
    if not skip_task_a:
        metric_df, cm_df = binary_metrics(predictions, "task_a_label_bin", "task_a_prob_positive")
        write_metric_outputs(metric_df, cm_df, metric_dir, "task_a")
        written.append("task_a")
    if not skip_task_b:
        metric_df, cm_df = binary_metrics(predictions, "task_b_label_bin", "task_b_prob_positive", mask_col="task_b_mask")
        write_metric_outputs(metric_df, cm_df, metric_dir, "task_b")
        written.append("task_b")
    if not skip_pathway and pathway_reference_col in predictions.columns:
        metric_df, cm_df = pathway_readout_summary(predictions, reference_col=pathway_reference_col)
        write_metric_outputs(metric_df, cm_df, metric_dir, "pathway")
        written.append("pathway")
    return written


def main() -> None:
    args = parse_args()
    import pandas as pd

    from cherish.inference import discover_checkpoints, read_checkpoint_manifest, run_locked_ensemble_inference

    if args.checkpoint_manifest is None and args.checkpoint_root is None:
        raise SystemExit("Provide either --checkpoint-manifest or --checkpoint-root.")

    manifest_df = pd.read_csv(args.manifest)
    checkpoints = (
        read_checkpoint_manifest(args.checkpoint_manifest)
        if args.checkpoint_manifest is not None
        else discover_checkpoints(args.checkpoint_root)
    )
    predictions = run_locked_ensemble_inference(
        manifest_df=manifest_df,
        checkpoints=checkpoints,
        output_dir=args.output_dir,
        device=args.device,
        max_slides=args.max_slides,
        max_tiles=args.max_tiles,
        sampling=args.sampling,
        seed=args.seed,
        write_per_fold=args.write_per_fold,
    )
    metric_tasks = _write_available_metrics(
        predictions,
        args.output_dir,
        skip_task_a=args.skip_task_a,
        skip_task_b=args.skip_task_b,
        skip_task_c=args.skip_task_c,
        skip_pathway=args.skip_pathway,
        pathway_reference_col=args.pathway_reference_col,
    )
    manifest = {
        "prediction_file": str(args.output_dir / "ensemble_slide_predictions.csv"),
        "metric_dir": str(args.output_dir / "metrics"),
        "n_slides": int(len(predictions)),
        "metric_tasks": metric_tasks,
        "locked_equal_weight_ensemble": True,
        "threshold_or_calibration_refit": False,
    }
    (args.output_dir / "validation_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
