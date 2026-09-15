#!/usr/bin/env python3
"""Train one CHERISH cross-validation fold from deidentified inputs."""

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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--val-fold", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import pandas as pd
    import yaml

    from cherish.training import train_one_fold

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest_df = pd.read_csv(args.manifest)
    split_df = pd.read_csv(args.splits)
    result = train_one_fold(
        cfg=cfg,
        manifest_df=manifest_df,
        split_df=split_df,
        output_dir=args.output_dir,
        val_fold=args.val_fold,
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
