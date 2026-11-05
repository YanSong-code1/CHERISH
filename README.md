# CHERISH

Hierarchical multiple instance learning for HER2 state recognition from H&E-stained breast cancer biopsies.

The model follows the clinical HER2 testing algorithm with three tasks:

- **Task A**: HER2-negative (IHC 0/1+ and nonamplified IHC 2+) vs HER2-positive
- **Task B**: amplified IHC 2+ vs IHC 3+ within HER2-positive disease
- **Task C**: three-state probability over nonamplified IHC 2+, amplified IHC 2+, and IHC 3+

Task C uses a structured base distribution derived from Task A and Task B bag representations, plus a morphological residual. The four-state readout is pathway-matched to each clinical category:

- IHC 0/1+: negative-side combined score from Task A
- IHC 2+: conditional amplification score from Task C, threshold 0.50
- IHC 3+: largest Task C probability among the three positive states

## Installation

Python 3.12+. Install dependencies:

```bash
pip install -r requirements.txt
```

CONCH v1.5 weights are not included. Obtain them separately under the CC BY-NC-ND 4.0 license.

## Usage

### Training

Five-fold cross-validation with patient-level stratified splits:

```bash
python scripts/train_cherish_cv.py --config configs/training_example.yaml
```

### Inference

Locked five-fold ensemble inference:

```bash
python scripts/run_locked_ensemble_inference.py \
    --manifest examples/manifest_schema.csv \
    --checkpoint-manifest examples/checkpoint_manifest_schema.csv \
    --feature-dir /path/to/conch_features \
    --output results.csv
```

Feature files are HDF5 with datasets `features` (N × 512, float16) and `coords` (N × 2, int32).

### Validation

```bash
python scripts/validate_locked_ensemble.py \
    --manifest examples/manifest_schema.csv \
    --predictions results.csv \
    --split examples/split_schema.csv \
    --output validation_metrics.json
```

## Data schema

Manifests are CSV files. Required columns:

- `slide_id`, `patient_id`
- `task_a_label_bin`: 0 = IHC 0/1+ or nonamplified IHC 2+; 1 = amplified IHC 2+ or IHC 3+
- `task_b_label_bin`: 0 = amplified IHC 2+; 1 = IHC 3+ (ignored if `task_b_mask = 0`)
- `task_c_label_index`: 0 = nonamplified IHC 2+; 1 = amplified IHC 2+; 2 = IHC 3+
- `task_b_mask`, `task_c_mask`: 1 if HER2-positive

See `examples/manifest_schema_dictionary.csv` for the full dictionary. Splits are in `examples/split_schema.csv` (`patient_id`, `fold`).

## Architecture

Four gated attention pools (shared, Task A, Task B, Task C) aggregate CONCH tile embeddings into bag representations. Task C derives its base distribution from separate linear projections of the Task A and Task B bags:

```
alpha = sigmoid(task_c_alpha(task_a_bag))
beta  = sigmoid(task_c_beta(task_b_bag))
q = [1 - alpha, alpha * beta, alpha * (1 - beta)]
```

A residual head refines this base, and the result is normalized to three-state probabilities. The four-state pathway masses are:

```
[1 - p_A, p_A * p_C[0], p_A * p_C[1], p_A * p_C[2]]
```

The conditional amplification score within IHC 2+ disease is `p_C[1] / (p_C[0] + p_C[1] + 1e-6)`.

## Environment

Tested on Ubuntu 22.04 with an NVIDIA RTX 5090, CUDA 12.8, and PyTorch 2.10.0+cu128. See `environment.json` for the full configuration. 

## Citation

.......coming soon

## License

Non-commercial academic research use only. See LICENSE.

Institutional images, features, and trained weights are not included. They are available for non-commercial research on request under a data transfer agreement and ethics approval.

CONCH weights are subject to the original CC BY-NC-ND 4.0 license and must not be redistributed.
