"""
Preprocessing pipeline for deforestation detection.

Analyzes NaN patterns, filters features, imputes missing values,
scales features, and saves processed data.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm

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

METADATA_COLUMNS = ["tile_id", "split", "year", "month"]
TARGET_COLUMN = "deforestation_target"
FILTER_COLUMN = "forest_2020"

TRAIN_TILES = [
    "47QMB_0_8",
    "48PWV_7_8",
    "48PUT_0_8",
    "48QWD_2_2",
    "48PXC_7_7",
    "48PYB_3_6",
    "47QQV_2_4",
    "48QVE_3_0",
    "18NXH_6_8",
    "18NWG_6_6",
    "18NWJ_8_9",
    "18NXJ_7_6",
    "18NWH_1_4",
    "18NYH_9_9",
    "19NBD_4_4",
    "18NWM_9_4",
]

VAL_TILES = ["18NWG_6_6", "18NWH_1_4", "18NXH_6_8", "18NWM_9_4"]

TEST_TILES = ["18NVJ_1_6", "18NYH_2_1", "33NTE_5_1", "47QMA_6_2", "48PWA_0_6"]


def analyze_nan_orbits(parquet_dir: Path) -> Dict[str, float]:
    """Analyze NaN percentage in ascending vs descending orbit features."""
    parquet_dir = Path(parquet_dir)
    parquet_files = sorted(parquet_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {parquet_dir}")

    print("[INFO] Analyzing NaN patterns in orbit features...")

    total_rows = 0
    vv_desc_nan = 0
    vv_asc_nan = 0

    for pf in tqdm(parquet_files, desc="Scanning parquets"):
        df = pl.scan_parquet(pf).select(["vv_desc", "vv_asc"]).collect()
        total_rows += df.height
        vv_desc_nan += df.filter(pl.col("vv_desc").is_nan()).height
        vv_asc_nan += df.filter(pl.col("vv_asc").is_nan()).height

    vv_desc_pct = vv_desc_nan / total_rows * 100
    vv_asc_pct = vv_asc_nan / total_rows * 100

    print(f"[INFO] Total rows: {total_rows:,}")
    print(f"[INFO] vv_desc NaN: {vv_desc_pct:.2f}%")
    print(f"[INFO] vv_asc  NaN: {vv_asc_pct:.2f}%")

    return {
        "vv_desc_nan_pct": vv_desc_pct,
        "vv_asc_nan_pct": vv_asc_pct,
        "total_rows": total_rows,
    }


def get_dropped_columns() -> List[str]:
    """Return columns to drop (high NaN orbit + weak labels + metadata)."""
    return [
        "vv_asc",
        "vv_asc_zscore_6mo",
        "vv_asc_diff_mom",
        "vv_roughness_drop_asc",
        "gladl_alert",
        "glads2_alert",
        "radd_alert",
        "pixel_x",
        "pixel_y",
    ]


def get_feature_columns() -> List[str]:
    """Return features to keep for training."""
    return FEATURE_COLUMNS.copy()


def assign_split(tile_id: str) -> str:
    """Assign tile to train/val/test split."""
    if tile_id in VAL_TILES:
        return "val"
    elif tile_id in TEST_TILES:
        return "test"
    else:
        return "train"


def filter_and_split(
    parquet_dir: Path,
    output_dir: Path,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load parquets, filter forest_2020==1, split into train/val/test."""
    parquet_dir = Path(parquet_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    print(f"[INFO] Found {len(parquet_files)} parquet files")

    dropped_cols = get_dropped_columns()
    keep_cols = METADATA_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN, FILTER_COLUMN]

    train_dfs, val_dfs, test_dfs = [], [], []

    for pf in tqdm(parquet_files, desc="Loading & filtering"):
        df = pl.read_parquet(pf, columns=keep_cols)

        df = df.filter(pl.col(FILTER_COLUMN) == 1)
        df = df.filter(pl.col(TARGET_COLUMN).is_not_nan())

        df = df.drop(FILTER_COLUMN)

        tile_id = df.select("tile_id").unique().to_series()[0]
        split = assign_split(tile_id)
        df = df.with_columns(pl.lit(split).alias("split"))

        if split == "train":
            train_dfs.append(df)
        elif split == "val":
            val_dfs.append(df)
        else:
            test_dfs.append(df)

    print(
        f"[INFO] Train tiles: {len(train_dfs)}, Val tiles: {len(val_dfs)}, Test tiles: {len(test_dfs)}"
    )

    train_df = pl.concat(train_dfs) if train_dfs else pl.DataFrame()
    val_df = pl.concat(val_dfs) if val_dfs else pl.DataFrame()
    test_df = pl.concat(test_dfs) if test_dfs else pl.DataFrame()

    return train_df, val_df, test_df


def fit_preprocessors(
    train_df: pl.DataFrame,
    feature_cols: List[str],
) -> Tuple[SimpleImputer, StandardScaler]:
    """Fit imputer and scaler on training data."""
    print(
        f"[INFO] Fitting imputer and scaler on {train_df.height:,} training samples..."
    )

    X_train = train_df.select(feature_cols).to_numpy()

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_imputed = imputer.fit_transform(X_train)

    scaler = StandardScaler()
    scaler.fit(X_imputed)

    print(
        f"[INFO] Imputer statistics (median) computed for {len(feature_cols)} features"
    )
    print(f"[INFO] Scaler mean: {scaler.mean_[:5]}... (showing first 5)")
    print(f"[INFO] Scaler std: {scaler.scale_[:5]}... (showing first 5)")

    return imputer, scaler


def transform_dataset(
    df: pl.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply imputer and scaler, return X, y, metadata."""
    if df.height == 0:
        return np.array([]), np.array([]), np.array([])

    X = df.select(feature_cols).to_numpy()
    y = df.select(TARGET_COLUMN).to_numpy().ravel()
    tile_ids = df.select("tile_id").to_series().to_numpy()

    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)

    return X_scaled, y, tile_ids


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute class weights for imbalanced classification."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)

    weights = {}
    for cls, count in zip(unique, counts):
        weights[cls] = total / (2 * count)

    print(f"[INFO] Class distribution: {dict(zip(unique, counts))}")
    print(f"[INFO] Class weights: {weights}")

    return weights


def save_processed_data(
    output_dir: Path,
    split: str,
    X: np.ndarray,
    y: np.ndarray,
    metadata: np.ndarray,
) -> None:
    """Save processed data as .npz file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / f"{split}.npz",
        X=X.astype(np.float32),
        y=y.astype(np.float32),
        metadata=metadata,
    )
    print(f"[INFO] Saved {split}.npz: X={X.shape}, y={y.shape}")


def save_preprocessors(
    output_dir: Path,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    feature_cols: List[str],
) -> None:
    """Save imputer and scaler to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "preprocessors.pkl", "wb") as f:
        pickle.dump(
            {
                "imputer": imputer,
                "scaler": scaler,
                "feature_cols": feature_cols,
            },
            f,
        )
    print(f"[INFO] Saved preprocessors to {output_dir / 'preprocessors.pkl'}")


def load_preprocessors(
    output_dir: Path,
) -> Tuple[SimpleImputer, StandardScaler, List[str]]:
    """Load imputer and scaler from disk."""
    with open(output_dir / "preprocessors.pkl", "rb") as f:
        data = pickle.load(f)
    return data["imputer"], data["scaler"], data["feature_cols"]


def preprocess_pipeline(
    parquet_dir: Path = None,
    output_dir: Path = None,
    skip_analysis: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Run full preprocessing pipeline.

    Returns:
        Dictionary with 'train', 'val', 'test' splits as (X, y) tuples.
    """
    if parquet_dir is None:
        parquet_dir = Path(__file__).parent / "data" / "parquet_with_baseline"
    if output_dir is None:
        output_dir = Path(__file__).parent / "data" / "processed"

    parquet_dir = Path(parquet_dir)
    output_dir = Path(output_dir)

    if not skip_analysis:
        analyze_nan_orbits(parquet_dir)

    train_df, val_df, test_df = filter_and_split(parquet_dir, output_dir)

    feature_cols = get_feature_columns()
    imputer, scaler = fit_preprocessors(train_df, feature_cols)

    save_preprocessors(output_dir, imputer, scaler, feature_cols)

    print("[INFO] Transforming datasets...")

    X_train, y_train, meta_train = transform_dataset(
        train_df, imputer, scaler, feature_cols
    )
    X_val, y_val, meta_val = transform_dataset(val_df, imputer, scaler, feature_cols)
    X_test, y_test, meta_test = transform_dataset(
        test_df, imputer, scaler, feature_cols
    )

    save_processed_data(output_dir, "train", X_train, y_train, meta_train)
    save_processed_data(output_dir, "val", X_val, y_val, meta_val)
    save_processed_data(output_dir, "test", X_test, y_test, meta_test)

    print(
        f"[INFO] Train: {X_train.shape[0]:,} samples, positive: {y_train.sum():,} ({y_train.mean() * 100:.2f}%)"
    )
    print(
        f"[INFO] Val: {X_val.shape[0]:,} samples, positive: {y_val.sum():,} ({y_val.mean() * 100:.2f}%)"
    )
    print(
        f"[INFO] Test: {X_test.shape[0]:,} samples, positive: {y_test.sum():,} ({y_test.mean() * 100:.2f}%)"
    )

    class_weights = compute_class_weights(y_train)

    np.savez(
        output_dir / "class_weights.npz",
        weights=np.array([class_weights[0], class_weights[1]]),
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
        "class_weights": class_weights,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess parquet data for training")
    parser.add_argument("--parquet-dir", type=Path, help="Input parquet directory")
    parser.add_argument(
        "--output-dir", type=Path, help="Output directory for processed data"
    )
    parser.add_argument(
        "--skip-analysis", action="store_true", help="Skip NaN orbit analysis"
    )

    args = parser.parse_args()

    preprocess_pipeline(
        parquet_dir=args.parquet_dir,
        output_dir=args.output_dir,
        skip_analysis=args.skip_analysis,
    )
