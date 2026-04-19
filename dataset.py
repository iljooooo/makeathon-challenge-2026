"""
Dataset utilities for deforestation detection.

Loads preprocessed parquet files, applies imputation and scaling on-the-fly.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Tuple, Optional, Dict
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


class ProcessedDataModule:
    """
    High-level data module for all models.

    Loads parquet files, fits imputer/scaler on train data,
    applies to val/test.

    Args:
        processed_dir: Directory containing train/, val/, test/ subdirectories
    """

    def __init__(self, processed_dir: Path):
        self.processed_dir = Path(processed_dir)
        self.feature_cols = self._load_feature_cols()
        self.imputer = None
        self.scaler = None
        self._cached_splits = {}

    def _load_feature_cols(self) -> List[str]:
        """Load feature columns from features.parquet."""
        feature_path = self.processed_dir / "features.parquet"
        if feature_path.exists():
            df = pl.read_parquet(feature_path)
            return df["feature"].to_list()
        return FEATURE_COLUMNS.copy()

    def _fit_preprocessors(
        self, X_train: np.ndarray
    ) -> Tuple[SimpleImputer, StandardScaler]:
        """Fit imputer and scaler on training data."""
        print(
            f"[INFO] Fitting imputer and scaler on {X_train.shape[0]:,} training samples..."
        )

        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        X_imputed = imputer.fit_transform(X_train)

        scaler = StandardScaler()
        scaler.fit(X_imputed)

        print(f"[INFO] Median imputer fitted for {len(self.feature_cols)} features")
        print(f"[INFO] Scaler mean: {scaler.mean_[:5]}... (first 5)")

        return imputer, scaler

    def _load_split_parquets(self, split: str) -> pl.DataFrame:
        """Load all parquet files for a split."""
        split_dir = self.processed_dir / split
        if not split_dir.exists():
            return pl.DataFrame()

        parquet_files = sorted(split_dir.glob("*.parquet"))
        if not parquet_files:
            return pl.DataFrame()

        dfs = [pl.read_parquet(pf) for pf in parquet_files]
        return pl.concat(dfs)

    def _prepare_numpy(
        self, df: pl.DataFrame, fit: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert polars df to numpy arrays."""
        X = df.select(self.feature_cols).to_numpy()
        y = df.select(TARGET_COLUMN).to_numpy().ravel()

        if fit:
            self.imputer, self.scaler = self._fit_preprocessors(X)

        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed).astype(np.float32)
        y = y.astype(np.float32)

        return X_scaled, y

    def get_numpy(self, split: str, fit: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get NumPy arrays for a split.

        Args:
            split: "train", "val", or "test"
            fit: If True, fit imputer/scaler on this data

        Returns:
            X: (n_samples, n_features) array
            y: (n_samples,) array
        """
        if split in self._cached_splits:
            return self._cached_splits[split]

        df = self._load_split_parquets(split)
        if df.height == 0:
            return np.array([]), np.array([])

        X, y = self._prepare_numpy(df, fit=fit)

        del df

        self._cached_splits[split] = (X, y)
        return X, y

    def clear_cache(self):
        """Clear cached split data to free memory."""
        self._cached_splits.clear()

    def get_class_weights(self) -> Dict[int, float]:
        """Get class weights from metadata."""
        meta_path = self.processed_dir / "metadata.parquet"
        if not meta_path.exists():
            # Compute from train data
            X_train, y_train = self.get_numpy("train", fit=True)
            total = len(y_train)
            pos = y_train.sum()
            neg = total - pos
            return {
                0: total / (2 * neg) if neg > 0 else 1.0,
                1: total / (2 * pos) if pos > 0 else 1.0,
            }

        meta = pl.read_parquet(meta_path)
        train_count = meta.filter(
            (pl.col("split") == "train") & (pl.col("metric") == "count")
        )["value"][0]
        train_positive = meta.filter(
            (pl.col("split") == "train") & (pl.col("metric") == "positive")
        )["value"][0]
        train_negative = train_count - train_positive

        return {
            0: train_count / (2 * train_negative) if train_negative > 0 else 1.0,
            1: train_count / (2 * train_positive) if train_positive > 0 else 1.0,
        }

    def save_preprocessors(self, output_path: Path = None):
        """Save imputer and scaler."""
        if output_path is None:
            output_path = self.processed_dir / "preprocessors.pkl"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            pickle.dump(
                {
                    "imputer": self.imputer,
                    "scaler": self.scaler,
                    "feature_cols": self.feature_cols,
                },
                f,
            )
        print(f"[INFO] Saved preprocessors to {output_path}")

    def load_preprocessors(self, input_path: Path = None):
        """Load imputer and scaler."""
        if input_path is None:
            input_path = self.processed_dir / "preprocessors.pkl"
        input_path = Path(input_path)

        with open(input_path, "rb") as f:
            data = pickle.load(f)
        self.imputer = data["imputer"]
        self.scaler = data["scaler"]
        self.feature_cols = data["feature_cols"]
        print(f"[INFO] Loaded preprocessors from {input_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test dataset loading")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))

    args = parser.parse_args()

    dm = ProcessedDataModule(processed_dir=args.processed_dir)

    print("[INFO] Loading train data and fitting preprocessors...")
    X_train, y_train = dm.get_numpy("train", fit=True)
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Positive class: {y_train.sum():,} ({y_train.mean() * 100:.2f}%)")

    print("[INFO] Loading val data...")
    X_val, y_val = dm.get_numpy("val")
    print(f"Val: X={X_val.shape}, y={y_val.shape}")

    print("[INFO] Class weights:", dm.get_class_weights())

    dm.save_preprocessors()
