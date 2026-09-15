#!/usr/bin/env python3
"""Compute CHERISH metrics from a prediction CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--task", type=str, required=True, choices=["task_a", "task_b", "task_c", "pathway"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-col", type=str, default="reference_pathway_label")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import pandas as pd

    from cherish.metrics import binary_metrics, pathway_readout_summary, task_c_metrics, write_metric_outputs

    df = pd.read_csv(args.predictions)
    if args.task == "task_c":
        metric_df, cm_df = task_c_metrics(df)
    elif args.task == "task_a":
        metric_df, cm_df = binary_metrics(df, "task_a_label_bin", "task_a_prob_positive")
    elif args.task == "pathway":
        metric_df, cm_df = pathway_readout_summary(df, reference_col=args.reference_col)
    else:
        metric_df, cm_df = binary_metrics(df, "task_b_label_bin", "task_b_prob_positive", mask_col="task_b_mask")
    write_metric_outputs(metric_df, cm_df, args.output_dir, args.task)
    print(f"Wrote {args.task} metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
