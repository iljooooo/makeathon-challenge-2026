"""
Dataset and DataLoader utilities for deforestation detection.

Provides:
- StreamingParquetDataset: PyTorch IterableDataset for memory-efficient streaming
- ProcessedDataModule: High-level API for all models (PyTorch, XGBoost, sklearn)
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterator
import torch
from torch.utils.data import IterableDataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pickle


FEATURE_COLUMNS = [
    "evi",
    "ndmi",
    "bsi",
    "ndre",
    "evi_zscore_6mo",
    "ndmi_zscore_6mo",
    "bsi_zscore_6mo",
    "ndre_zscore_6mo",
    "evi_diff_mom",
    "ndmi_diff_mom",
    "bsi_diff_mom",
    "ndre_diff_mom",
    "vv_desc",
    "vv_desc_zscore_6mo",
    "vv_desc_diff_mom",
    "vv_roughness_drop_desc",
    "latent_shift_yoy_1y",
    "latent_shift_yoy_2y",
    "treecover2000_pct",
    "lossyear_hansen",
]

TARGET_COLUMN = "deforestation_target"
FILTER_COLUMN = "forest_2020"
METADATA_COLUMNS = ["tile_id", "split", "year", "month"]

VAL_TILES = ["18NWG_6_6", "18NWH_1_4", "18NXH_6_8", "18NWM_9_4"]
TEST_TILES = ["18NVJ_1_6", "18NYH_2_1", "33NTE_5_1", "47QMA_6_2", "48PWA_0_6"]


def assign_split(tile_id: str) -> str:
    if tile_id in VAL_TILES:
        return "val"
    elif tile_id in TEST_TILES:
        return "test"
    return "train"


class StreamingParquetDataset(IterableDataset):
    """
    PyTorch IterableDataset that streams parquet tiles.

    Memory-efficient: Loads max `max_parquets_in_memory` tiles at a time.

    Args:
        parquet_dir: Directory containing {tile_id}.parquet files
        split: "train", "val", or "test"
        feature_cols: List of feature columns
        imputer: Fitted SimpleImputer
        scaler: Fitted StandardScaler
        max_parquets_in_memory: Maximum parquet files loaded simultaneously
    """

    def __init__(
        self,
        parquet_dir: Path,
        split: str,
        feature_cols: List[str],
        imputer: SimpleImputer,
        scaler: StandardScaler,
        max_parquets_in_memory: int = 4,
    ):
        self.parquet_dir = Path(parquet_dir)
        self.split = split
        self.feature_cols = feature_cols
        self.imputer = imputer
        self.scaler = scaler
        self.max_parquets_in_memory = max_parquets_in_memory

        parquet_files = sorted(self.parquet_dir.glob("*.parquet"))

        self.tile_parquet_files = {}
        for pf in parquet_files:
            tile_id = pf.stem
            self.tile_parquet_files[tile_id] = pf

        self.split_tiles = self._get_split_tiles()

    def _get_split_tiles(self) -> List[str]:
        split_tiles = []
        for tile_id in self.tile_parquet_files:
            assigned = assign_split(tile_id)
            if assigned == self.split:
                split_tiles.append(tile_id)
        return split_tiles

    def _load_and_process_parquet(
        self, tile_id: str
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        pf = self.tile_parquet_files[tile_id]
        keep_cols = (
            METADATA_COLUMNS + self.feature_cols + [TARGET_COLUMN, FILTER_COLUMN]
        )

        df = pl.read_parquet(pf, columns=keep_cols)

        df = df.filter(pl.col(FILTER_COLUMN) == 1)
        df = df.filter(pl.col(TARGET_COLUMN).is_not_nan())
        df = df.drop(FILTER_COLUMN)

        if df.height == 0:
            return

        X = df.select(self.feature_cols).to_numpy()
        y = df.select(TARGET_COLUMN).to_numpy().ravel()

        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)

        for i in range(len(X_scaled)):
            yield (
                torch.tensor(X_scaled[i], dtype=torch.float32),
                torch.tensor(y[i], dtype=torch.float32),
            )

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            tile_ids = self.split_tiles
        else:
            tiles_per_worker = len(self.split_tiles) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * tiles_per_worker
            end = (
                start + tiles_per_worker
                if worker_id < worker_info.num_workers - 1
                else len(self.split_tiles)
            )
            tile_ids = self.split_tiles[start:end]

        for tile_id in tile_ids:
            yield from self._load_and_process_parquet(tile_id)

    def __len__(self) -> int:
        return len(self.split_tiles)


class ProcessedDataModule:
    """
    High-level data module for all models.

    Provides:
    - get_dataloader(): PyTorch DataLoader for neural networks
    - get_numpy(): NumPy arrays for XGBoost/sklearn
    - get_class_weights(): Class weights for imbalanced loss

    Args:
        processed_dir: Directory containing {train,val,test}.npz files
        parquet_dir: Directory containing {tile_id}.parquet files (for streaming)
        imputer: Fitted SimpleImputer (or path to preprocessors.pkl)
        scaler: Fitted StandardScaler (or path to preprocessors.pkl)
    """

    def __init__(
        self,
        processed_dir: Optional[Path] = None,
        parquet_dir: Optional[Path] = None,
        imputer: Optional[SimpleImputer] = None,
        scaler: Optional[StandardScaler] = None,
    ):
        self.processed_dir = Path(processed_dir) if processed_dir else None
        self.parquet_dir = Path(parquet_dir) if parquet_dir else None

        if imputer is None or scaler is None:
            if self.processed_dir:
                self.imputer, self.scaler, self.feature_cols = (
                    self._load_preprocessors()
                )
            else:
                raise ValueError("Must provide either imputer/scaler or processed_dir")
        else:
            self.imputer = imputer
            self.scaler = scaler
            self.feature_cols = FEATURE_COLUMNS

        self._cached_data = {}

    def _load_preprocessors(self) -> Tuple[SimpleImputer, StandardScaler, List[str]]:
        with open(self.processed_dir / "preprocessors.pkl", "rb") as f:
            data = pickle.load(f)
        return data["imputer"], data["scaler"], data["feature_cols"]

    def _load_numpy_split(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        if split not in self._cached_data:
            data = np.load(self.processed_dir / f"{split}.npz")
            self._cached_data[split] = (data["X"], data["y"])
        return self._cached_data[split]

    def get_numpy(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get NumPy arrays for a split.

        Args:
            split: "train", "val", or "test"

        Returns:
            X: (n_samples, n_features) array
            y: (n_samples,) array
        """
        return self._load_numpy_split(split)

    def get_dataloader(
        self,
        split: str,
        batch_size: int = 1024,
        num_workers: int = 4,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Get PyTorch DataLoader for a split.

        Args:
            split: "train", "val", or "test"
            batch_size: Batch size
            num_workers: Number of parallel data loaders
            shuffle: Shuffle batches (only for train)

        Returns:
            DataLoader
        """
        if self.parquet_dir is None:
            X, y = self.get_numpy(split)
            dataset = torch.utils.data.TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle if split == "train" else False,
                num_workers=0,
            )

        dataset = StreamingParquetDataset(
            parquet_dir=self.parquet_dir,
            split=split,
            feature_cols=self.feature_cols,
            imputer=self.imputer,
            scaler=self.scaler,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_class_weights(self) -> Dict[int, float]:
        """Get class weights computed from training data."""
        data = np.load(self.processed_dir / "class_weights.npz")
        weights = data["weights"]
        return {0: float(weights[0]), 1: float(weights[1])}


def get_class_weights_numpy(y: np.ndarray) -> Dict[int, float]:
    """Compute class weights from labels array."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = {}
    for cls, count in zip(unique, counts):
        weights[cls] = total / (2 * count)
    return weights


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test dataset loading")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()

    dm = ProcessedDataModule(processed_dir=args.processed_dir)

    print("[INFO] Testing numpy loading...")
    X_train, y_train = dm.get_numpy("train")
    print(f"Train: X={X_train.shape}, y={y_train.shape}")

    print("[INFO] Testing DataLoader...")
    train_loader = dm.get_dataloader("train", batch_size=args.batch_size)
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        print(f"Batch {batch_idx}: X={X_batch.shape}, y={y_batch.shape}")
        if batch_idx >= 2:
            break

    print("[INFO] Class weights:", dm.get_class_weights())
