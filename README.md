# CHERISH

Code used for the CHERISH study:

https://github.com/YanSong-code1/CHERISH

The package contains the model, training utilities, locked ensemble inference, clinical-pathway readouts, and core performance metrics. It does not contain patient data, whole-slide images, private feature bags, trained checkpoints, or third-party model weights.

## Layout

- `cherish/model.py` — hierarchical multitask MIL model and gated attention.
- `cherish/dataset.py` — manifest-backed feature-bag dataset.
- `cherish/training.py` — one-fold training and validation utilities.
- `cherish/inference.py` — locked equal-weight ensemble inference.
- `cherish/metrics.py` — binary, three-state, and pathway-readout metrics.
- `scripts/` — command-line entry points.
- `configs/` — example YAML files.
- `examples/` — input schemas only.

## Input

The model takes precomputed CONCH v1.5 tile features. A feature file is an `.h5`, `.hdf5`, or `.npz` file containing:

- `features`: a float array with shape `[n_tiles, feature_dim]`;
- `coords`: an optional `[n_tiles, 2]` coordinate array, required for the manuscript configuration.

The manifest format is shown in `examples/manifest_schema.csv` and described in `examples/manifest_schema_dictionary.csv`. Replace the example paths with approved local paths before running the code.

## Outputs

The model returns the polarity probability, the positive-boundary probability, the three-state probabilities, and the conditional ERBB2 amplification probability for IHC 2+ cases.

The four clinical HER2 readouts are derived from prespecified clinical-pathway outputs. No independently trained four-class probability vector is constructed. The conditional IHC 2+/FISH amplification score is calculated from the two boundary-state probabilities and uses a 0.5 threshold when a binary call is requested.

## Install

```bash
python -m pip install -r requirements.txt
```

## Train one fold

```bash
python scripts/train_cherish_cv.py \
  --config configs/training_example.yaml \
  --manifest path/to/training_manifest.csv \
  --splits path/to/fold_assignments.csv \
  --val-fold 0 \
  --output-dir outputs/fold_0 \
  --device cuda:0
```

## Run locked inference

```bash
python scripts/run_locked_ensemble_inference.py \
  --manifest path/to/deidentified_manifest.csv \
  --checkpoint-manifest path/to/checkpoint_manifest.csv \
  --output-dir outputs/locked_external \
  --device cuda:0
```

The formal estimator is the equal-weight five-checkpoint ensemble. The inference script does not retune thresholds on validation or external-test labels.

## Compute metrics

```bash
python scripts/compute_metrics.py \
  --predictions outputs/locked_external/ensemble_slide_predictions.csv \
  --task task_c \
  --output-dir outputs/locked_external/metrics
```

For four-category clinical-pathway summaries, use `--task pathway` and provide the reference-label column with `--reference-col`.

## Data and model weights

CONCH, TRIDENT, TIAToolbox, HoVer-Net, and cell2location are external dependencies. Their weights and licenses are not included here. Raw WSI files, institutional data, private feature bags, and trained checkpoints must be supplied separately by an approved user.

## Citation

Please cite the CHERISH manuscript and this repository when using the code.
