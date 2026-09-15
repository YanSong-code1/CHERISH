"""Generic manifest-backed datasets for CHERISH feature bags."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def read_feature_bag(path: str | Path, require_coords: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """Read a feature bag from `.h5`, `.hdf5`, or `.npz`."""

    path = Path(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as handle:
            features = handle["features"][:].astype(np.float32)
            coords = handle["coords"][:].astype(np.float32) if "coords" in handle else None
    elif path.suffix.lower() == ".npz":
        payload = np.load(path)
        features = payload["features"].astype(np.float32)
        coords = payload["coords"].astype(np.float32) if "coords" in payload else None
    else:
        raise ValueError(f"Unsupported feature file format: {path}")

    if require_coords and coords is None:
        raise KeyError(f"coords not found in feature bag: {path}")
    return features, coords


class MultitaskHER2Dataset(Dataset):
    """Dataset for deidentified CHERISH slide manifests."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_col: str = "feature_bag_path",
        id_col: str = "slide_id",
        max_tiles: int | None = None,
        sampling: str = "random",
        seed: int = 42,
        return_coords: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.feature_col = feature_col
        self.id_col = id_col
        self.max_tiles = None if max_tiles is None else int(max_tiles)
        self.sampling = str(sampling)
        self.seed = int(seed)
        self.return_coords = bool(return_coords)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.df)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + self.epoch * 1000 + idx * 17)

    def _sample_indices(self, n_tiles: int, idx: int) -> np.ndarray:
        if self.max_tiles is None or n_tiles <= self.max_tiles:
            return np.arange(n_tiles, dtype=int)
        if self.sampling == "random":
            return np.asarray(self._rng(idx).choice(n_tiles, size=self.max_tiles, replace=False), dtype=int)
        return np.arange(self.max_tiles, dtype=int)

    @staticmethod
    def _label_or_default(row: pd.Series, column: str, default: float | int) -> float | int:
        if column not in row or pd.isna(row[column]):
            return default
        return row[column]

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        features, coords = read_feature_bag(row[self.feature_col], require_coords=self.return_coords)
        indices = self._sample_indices(int(features.shape[0]), idx)
        features = features[indices]
        if coords is not None:
            coords = coords[indices]

        slide_id = str(row[self.id_col]) if self.id_col in row else str(idx)
        task_a = float(self._label_or_default(row, "task_a_label_bin", 0.0))
        task_b = float(self._label_or_default(row, "task_b_label_bin", 0.0))
        task_c = int(self._label_or_default(row, "task_c_label_index", -1))
        task_b_mask = float(self._label_or_default(row, "task_b_mask", 0.0))
        task_c_mask = float(self._label_or_default(row, "task_c_mask", 1.0 if task_c >= 0 else 0.0))

        if self.return_coords:
            assert coords is not None
            return (
                torch.from_numpy(features),
                torch.from_numpy(coords),
                torch.tensor(task_a, dtype=torch.float32),
                torch.tensor(task_b, dtype=torch.float32),
                torch.tensor(task_c, dtype=torch.long),
                torch.tensor(task_b_mask, dtype=torch.float32),
                torch.tensor(task_c_mask, dtype=torch.float32),
                slide_id,
            )
        return (
            torch.from_numpy(features),
            None,
            torch.tensor(task_a, dtype=torch.float32),
            torch.tensor(task_b, dtype=torch.float32),
            torch.tensor(task_c, dtype=torch.long),
            torch.tensor(task_b_mask, dtype=torch.float32),
            torch.tensor(task_c_mask, dtype=torch.float32),
            slide_id,
        )


def collate_fn_multitask(batch):
    features, coords, task_a, task_b, task_c, task_b_mask, task_c_mask, slide_ids = zip(*batch)
    return (
        list(features),
        list(coords),
        torch.stack(task_a).to(dtype=torch.float32),
        torch.stack(task_b).to(dtype=torch.float32),
        torch.stack(task_c).to(dtype=torch.long),
        torch.stack(task_b_mask).to(dtype=torch.float32),
        torch.stack(task_c_mask).to(dtype=torch.float32),
        list(slide_ids),
    )
