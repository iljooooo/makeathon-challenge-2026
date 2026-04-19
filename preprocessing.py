"""
Preprocessing pipeline for deforestation detection.

Fast streaming version: reads parquet tiles, filters, splits, and saves.
No heavy computation - just filtering and splitting.
"""

import polars as pl
from pathlib import Path
from typing import List
import numpy as np
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

METADATA_COLUMNS = ["tile_id", "year", "month"]
TARGET_COLUMN = "deforestation_target"
FILTER_COLUMN = "forest_2020"

VAL_TILES = ["18NWG_6_6", "18NWH_1_4", "18NXH_6_8", "18NWM_9_4"]
TEST_TILES = ["18NVJ_1_6", "18NYH_2_1", "33NTE_5_1", "47QMA_6_2", "48PWA_0_6"]


def get_feature_columns() -> List[str]:
    return FEATURE_COLUMNS.copy()


def assign_split(tile_id: str) -> str:
    if tile_id in VAL_TILES:
        return "val"
    elif tile_id in TEST_TILES:
        return "test"
    return "train"


def preprocess_pipeline(
    parquet_dir: Path = None,
    output_dir: Path = None,
) -> None:
    """Filter forest_2020==1, split by tile, save as parquet."""
    if parquet_dir is None:
        parquet_dir = Path(__file__).parent / "data" / "parquet_with_baseline"
    if output_dir is None:
        output_dir = Path(__file__).parent / "data" / "processed"

    parquet_dir = Path(parquet_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    print(f"[INFO] Found {len(parquet_files)} parquet files")

    feature_cols = get_feature_columns()
    keep_cols = METADATA_COLUMNS + feature_cols + [TARGET_COLUMN, FILTER_COLUMN]

    counts = {"train": 0, "val": 0, "test": 0}
    positives = {"train": 0, "val": 0, "test": 0}

    for pf in tqdm(parquet_files, desc="Processing"):
        tile_id = pf.stem
        split = assign_split(tile_id)

        # Lazy load + filter
        df = (
            pl.scan_parquet(pf)
            .filter(pl.col(FILTER_COLUMN) == 1)
            .filter(pl.col(TARGET_COLUMN).is_not_nan())
            .select(keep_cols)
            .with_columns(pl.lit(split).alias("split"))
            .collect()
        )

        if df.height == 0:
            continue

        # Save to parquet
        output_path = output_dir / split / f"{tile_id}.parquet"
        df.write_parquet(output_path)

        counts[split] += df.height
        positives[split] += df.filter(pl.col(TARGET_COLUMN) == 1).height

        del df

    print(f"\n[INFO] Summary:")
    for split in ["train", "val", "test"]:
        if counts[split] > 0:
            pct = positives[split] / counts[split] * 100
            print(
                f"  {split}: {counts[split]:,} rows, {positives[split]:,} positive ({pct:.2f}%)"
            )

    # Compute class weights
    total_train = counts["train"]
    pos_train = positives["train"]
    neg_train = total_train - pos_train

    class_weights = {
        0: total_train / (2 * neg_train) if neg_train > 0 else 1.0,
        1: total_train / (2 * pos_train) if pos_train > 0 else 1.0,
    }

    # Save metadata as parquet
    meta_df = pl.DataFrame(
        {
            "split": ["train", "train", "val", "val", "test", "test"],
            "metric": ["count", "positive", "count", "positive", "count", "positive"],
            "value": [
                counts["train"],
                positives["train"],
                counts["val"],
                positives["val"],
                counts["test"],
                positives["test"],
            ],
        }
    )
    meta_df.write_parquet(output_dir / "metadata.parquet")

    # Save feature columns
    feature_df = pl.DataFrame({"feature": feature_cols})
    feature_df.write_parquet(output_dir / "features.parquet")

    print(f"[INFO] Class weights: {class_weights}")
    print(f"[INFO] Saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess parquet data")
    parser.add_argument("--parquet-dir", type=Path, help="Input parquet directory")
    parser.add_argument("--output-dir", type=Path, help="Output directory")

    args = parser.parse_args()

    preprocess_pipeline(parquet_dir=args.parquet_dir, output_dir=args.output_dir)
