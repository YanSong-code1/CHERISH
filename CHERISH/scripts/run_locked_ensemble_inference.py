#!/usr/bin/env python3
"""Run locked equal-weight CHERISH ensemble inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Deidentified slide manifest CSV.")
    parser.add_argument("--checkpoint-manifest", type=Path, default=None, help="CSV with run_name, checkpoint_path, config_path.")
    parser.add_argument("--checkpoint-root", type=Path, default=None, help="Root containing one subdirectory per checkpoint run.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-slides", type=int, default=None)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--sampling", type=str, default="random", choices=["random", "first"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-per-fold", action="store_true")
    return parser.parse_args()


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
    print(f"Wrote {len(predictions)} ensemble predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
