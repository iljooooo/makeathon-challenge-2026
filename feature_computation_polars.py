"""
Feature computation pipeline using Polars DataFrame operations.
Optimized for speed with 64GB RAM available.
"""

import numpy as np
import polars as pl
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, List
from tqdm import tqdm
import re
import random

SEED = 42
SAMPLE_FRACTION = 0.05  # Set to 1.0 for full dataset
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]


def load_tile_to_dataframe(
    data_dir: Path, tile_id: str, split: str = "train"
) -> pl.DataFrame:
    """Load all data for a tile into a single long-format DataFrame."""
    data_dir = Path(data_dir)

    s2_dir = data_dir / "sentinel-2" / split
    s1_dir = data_dir / "sentinel-1" / split
    aef_dir = data_dir / "aef-embeddings" / split
    labels_base_dir = data_dir / "labels" / "train"

    s2_tile_dir = s2_dir / f"{tile_id}__s2_l2a"
    if not s2_tile_dir.exists():
        return pl.DataFrame()

    ref_transform, ref_crs, ref_shape = None, None, None
    s2_data = {}

    for f in sorted(s2_tile_dir.glob("*.tif")):
        parsed = parse_s2_filename(f.name)
        if parsed[0]:
            _, year, month = parsed
            with rasterio.open(f) as src:
                arr = src.read().astype(np.float32)
                if ref_transform is None:
                    ref_transform, ref_crs, ref_shape = (
                        src.transform,
                        src.crs,
                        src.shape,
                    )
                elif src.shape != ref_shape:
                    print(
                        f"[WARN] {tile_id} {year}-{month}: resampling S2 {src.shape} -> {ref_shape}"
                    )
                    arr = _resample_bands(arr, src, ref_transform, ref_crs, ref_shape)
                s2_data[(year, month)] = arr

    if not s2_data:
        return pl.DataFrame()

    height, width = ref_shape
    n_pixels = height * width
    py, px = np.indices((height, width), dtype=np.int32)
    pixel_y_flat, pixel_x_flat = py.flatten(), px.flatten()

    n_timesteps = len(s2_data)
    total_rows = n_timesteps * n_pixels

    tile_id_arr = np.full(total_rows, tile_id, dtype=object)
    split_arr = np.full(total_rows, split, dtype=object)
    year_arr = np.empty(total_rows, dtype=np.int32)
    month_arr = np.empty(total_rows, dtype=np.int32)
    px_arr = np.tile(pixel_x_flat, n_timesteps)
    py_arr = np.tile(pixel_y_flat, n_timesteps)

    band_arrays = {band: np.empty(total_rows, dtype=np.float32) for band in S2_BANDS}

    row_idx = 0
    for (year, month), arr in sorted(s2_data.items()):
        year_arr[row_idx : row_idx + n_pixels] = year
        month_arr[row_idx : row_idx + n_pixels] = month
        for i, band in enumerate(S2_BANDS, start=1):
            band_arrays[band][row_idx : row_idx + n_pixels] = arr[i].flatten()
        row_idx += n_pixels

    df_dict = {
        "tile_id": tile_id_arr,
        "split": split_arr,
        "year": year_arr,
        "month": month_arr,
        "pixel_x": px_arr,
        "pixel_y": py_arr,
        **band_arrays,
    }

    s1_tile_dir = s1_dir / f"{tile_id}__s1_rtc"
    s1_data = {}
    if s1_tile_dir.exists() and ref_transform is not None:
        for f in sorted(s1_tile_dir.glob("*.tif")):
            parsed = parse_s1_filename(f.name)
            if parsed[0]:
                _, year, month, orbit = parsed
                with rasterio.open(f) as src:
                    arr = src.read(1).astype(np.float32)
                    if src.shape != ref_shape:
                        upsampled = np.empty(ref_shape, dtype=np.float32)
                        reproject(
                            arr,
                            upsampled,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=ref_transform,
                            dst_crs=ref_crs,
                            resampling=Resampling.bilinear,
                            nodata=np.nan,
                        )
                        arr = upsampled
                    s1_data[(year, month, orbit)] = arr.flatten()

    vv_desc_arr = np.full(total_rows, np.nan, dtype=np.float32)
    vv_asc_arr = np.full(total_rows, np.nan, dtype=np.float32)
    if s1_data:
        row_idx = 0
        for year, month in sorted(s2_data.keys()):
            key_desc = (year, month, "descending")
            key_asc = (year, month, "ascending")
            if key_desc in s1_data:
                vv_desc_arr[row_idx : row_idx + n_pixels] = s1_data[key_desc]
            if key_asc in s1_data:
                vv_asc_arr[row_idx : row_idx + n_pixels] = s1_data[key_asc]
            row_idx += n_pixels

    df_dict["vv_desc"] = vv_desc_arr
    df_dict["vv_asc"] = vv_asc_arr

    aef_data = {}
    if ref_transform is not None:
        for year in range(2020, 2026):
            f = aef_dir / f"{tile_id}_{year}.tiff"
            if f.exists():
                with rasterio.open(f) as src:
                    arr = src.read().astype(np.float32)
                    if src.shape != ref_shape:
                        resampled = np.empty(
                            (arr.shape[0], height, width), dtype=np.float32
                        )
                        for i in range(arr.shape[0]):
                            reproject(
                                arr[i],
                                resampled[i],
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=ref_transform,
                                dst_crs=ref_crs,
                                resampling=Resampling.bilinear,
                                nodata=np.nan,
                            )
                        arr = resampled
                    aef_data[year] = arr

    if aef_data:
        n_aef_bands = next(iter(aef_data.values())).shape[0]
        aef_arrays = {
            f"aef_{i}": np.full(total_rows, np.nan, dtype=np.float32)
            for i in range(n_aef_bands)
        }

        year_to_rows = {}
        row_idx = 0
        for year, month in sorted(s2_data.keys()):
            if year not in year_to_rows:
                year_to_rows[year] = []
            year_to_rows[year].append((row_idx, row_idx + n_pixels))
            row_idx += n_pixels

        for year, arr in aef_data.items():
            if year in year_to_rows:
                for start, end in year_to_rows[year]:
                    for i in range(n_aef_bands):
                        aef_arrays[f"aef_{i}"][start:end] = arr[i].flatten()

        df_dict.update(aef_arrays)

    labels = {}
    if split == "train" and ref_transform is not None:
        gladl_dir = labels_base_dir / "gladl"
        for yy in range(21, 26):
            year = 2000 + yy
            f = gladl_dir / f"gladl_{tile_id}_alert{yy}.tif"
            if f.exists():
                with rasterio.open(f) as src:
                    arr = src.read(1).astype(np.float32)
                    if src.shape != ref_shape:
                        resampled = np.empty(ref_shape, dtype=np.float32)
                        reproject(
                            arr,
                            resampled,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=ref_transform,
                            dst_crs=ref_crs,
                            resampling=Resampling.nearest,
                        )
                        arr = resampled
                    labels[("gladl", year)] = arr.flatten()

        f = labels_base_dir / "glads2" / f"glads2_{tile_id}_alert.tif"
        if f.exists():
            with rasterio.open(f) as src:
                arr = src.read(1).astype(np.float32)
                if src.shape != ref_shape:
                    resampled = np.empty(ref_shape, dtype=np.float32)
                    reproject(
                        arr,
                        resampled,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.nearest,
                    )
                    arr = resampled
                for year in range(2021, 2026):
                    labels[("glads2", year)] = arr.flatten()

        f = labels_base_dir / "radd" / f"radd_{tile_id}_labels.tif"
        if f.exists():
            with rasterio.open(f) as src:
                arr = src.read(1).astype(np.float32)
                if src.shape != ref_shape:
                    resampled = np.empty(ref_shape, dtype=np.float32)
                    reproject(
                        arr,
                        resampled,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.nearest,
                    )
                    arr = resampled
                for year in range(2021, 2026):
                    labels[("radd", year)] = arr.flatten()

    label_arrays = {
        "gladl_alert": np.full(total_rows, np.nan, dtype=np.float32),
        "glads2_alert": np.full(total_rows, np.nan, dtype=np.float32),
        "radd_alert": np.full(total_rows, np.nan, dtype=np.float32),
    }

    if labels:
        row_idx = 0
        for year, month in sorted(s2_data.keys()):
            for system in ["gladl", "glads2", "radd"]:
                key = (system, year)
                if key in labels:
                    label_arrays[f"{system}_alert"][row_idx : row_idx + n_pixels] = (
                        labels[key]
                    )
            row_idx += n_pixels

    df_dict.update(label_arrays)

    return pl.DataFrame(df_dict).sort(["pixel_x", "pixel_y", "year", "month"])


def _resample_bands(arr, src, dst_transform, dst_crs, dst_shape):
    resampled = np.empty((arr.shape[0], *dst_shape), dtype=np.float32)
    for i in range(arr.shape[0]):
        reproject(
            arr[i],
            resampled[i],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            nodata=np.nan,
        )
    return resampled


def compute_all_features(
    df: pl.DataFrame, lookback: int = 6, min_obs: int = 3
) -> pl.DataFrame:
    """Compute all features using optimized Polars expressions."""

    df = df.lazy()

    df = df.with_columns(
        [
            (
                (2.5 * (pl.col("B08") - pl.col("B04")))
                / (pl.col("B08") + 6.0 * pl.col("B04") - 7.5 * pl.col("B02") + 1.0)
            )
            .fill_null(0.0)
            .alias("evi"),
            ((pl.col("B08") - pl.col("B11")) / (pl.col("B08") + pl.col("B11")))
            .fill_null(0.0)
            .alias("ndmi"),
            (
                ((pl.col("B11") + pl.col("B04")) - (pl.col("B08") + pl.col("B02")))
                / ((pl.col("B11") + pl.col("B04")) + (pl.col("B08") + pl.col("B02")))
            )
            .fill_null(0.0)
            .alias("bsi"),
            ((pl.col("B08") - pl.col("B05")) / (pl.col("B08") + pl.col("B05")))
            .fill_null(0.0)
            .alias("ndre"),
        ]
    )

    for idx in ["evi", "ndmi", "bsi", "ndre"]:
        df = _add_temporal_lazy(df, idx, lookback, min_obs)

    for orbit in ["desc", "asc"]:
        df = _add_vv_temporal_lazy(df, f"vv_{orbit}", lookback, min_obs)

    df = df.drop(
        [
            c
            for c in df.collect_schema()
            if c.startswith("B0") or c.startswith("B8") or c.startswith("B1")
        ]
    )

    df = _add_latent_features_lazy(df)

    return df.collect()


def _add_temporal_lazy(
    df: pl.LazyFrame, idx: str, lookback: int, min_obs: int
) -> pl.LazyFrame:
    """Add z-score and diff for optical index using rolling operations."""

    lag_exprs = []
    for lag in range(1, lookback + 1):
        lag_exprs.append(
            pl.col(idx)
            .shift(lag)
            .over(["pixel_x", "pixel_y"])
            .alias(f"_{idx}_lag{lag}")
        )

    df = df.with_columns(lag_exprs)

    lag_cols = [f"_{idx}_lag{i}" for i in range(1, lookback + 1)]

    hist_mean = pl.mean_horizontal(lag_cols)
    hist_std = pl.std_horizontal(lag_cols)
    valid_count = pl.sum_horizontal([pl.col(c).is_not_null() for c in lag_cols])

    df = df.with_columns(
        [
            pl.when(valid_count >= min_obs)
            .then((pl.col(idx) - hist_mean) / hist_std)
            .otherwise(None)
            .alias(f"{idx}_zscore_6mo"),
            (pl.col(idx) - pl.col(f"_{idx}_lag1")).alias(f"{idx}_diff_mom"),
        ]
    )

    return df.drop(lag_cols)


def _add_vv_temporal_lazy(
    df: pl.LazyFrame, vv_col: str, lookback: int, min_obs: int
) -> pl.LazyFrame:
    """Add z-score, roughness drop, and diff for VV radar."""

    lag_exprs = []
    for lag in range(1, lookback + 1):
        lag_exprs.append(
            pl.col(vv_col)
            .shift(lag)
            .over(["pixel_x", "pixel_y"])
            .alias(f"_{vv_col}_lag{lag}")
        )

    df = df.with_columns(lag_exprs)

    lag_cols = [f"_{vv_col}_lag{i}" for i in range(1, lookback + 1)]

    hist_mean = pl.mean_horizontal(lag_cols)
    hist_std = pl.std_horizontal(lag_cols)
    hist_median = pl.median_horizontal(lag_cols)
    valid_count = pl.sum_horizontal([pl.col(c).is_not_null() for c in lag_cols])

    df = df.with_columns(
        [
            pl.when(valid_count >= min_obs)
            .then((pl.col(vv_col) - hist_mean) / hist_std)
            .otherwise(None)
            .alias(f"{vv_col}_zscore_6mo"),
            pl.when(valid_count >= min_obs)
            .then((hist_median - pl.col(vv_col)).clip(lower_bound=0.0))
            .otherwise(None)
            .alias(f"{vv_col}_roughness_drop"),
            (pl.col(vv_col) - pl.col(f"_{vv_col}_lag1")).alias(f"{vv_col}_diff_mom"),
        ]
    )

    return df.drop(lag_cols)


def _add_latent_features_lazy(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add year-over-year latent space shifts."""

    aef_cols = [f"aef_{i}" for i in range(64)]

    def cosine_shift_struct(s):
        curr = np.array([s[f"aef_{i}"] for i in range(64)], dtype=np.float32)
        prev = s["_prev"]
        if prev is None:
            return np.float32(np.nan)

        prev_arr = np.array([prev.get(f"aef_{i}") for i in range(64)], dtype=np.float32)

        if np.any(np.isnan(curr)) or np.any(np.isnan(prev_arr)):
            return np.float32(np.nan)

        norm_curr, norm_prev = np.linalg.norm(curr), np.linalg.norm(prev_arr)
        if norm_curr == 0 or norm_prev == 0:
            return np.float32(0.0)

        return np.float32(1.0 - np.dot(curr, prev_arr) / (norm_curr * norm_prev))

    for years, suffix in [(1, "1y"), (2, "2y")]:
        lag_months = years * 12

        aef_lag_exprs = [
            pl.col(f"aef_{i}")
            .shift(lag_months)
            .over(["pixel_x", "pixel_y"])
            .alias(f"_aef_lag_{i}")
            for i in range(64)
        ]

        df = df.with_columns(aef_lag_exprs)

        struct_col = f"_aef_struct_{suffix}"
        df = df.with_columns(
            [
                pl.struct(
                    [
                        pl.struct(aef_cols).alias("_current"),
                        pl.struct([f"_aef_lag_{i}" for i in range(64)]).alias("_prev"),
                    ]
                ).alias(struct_col)
            ]
        )

        df = df.with_columns(
            [
                pl.col(struct_col)
                .map_elements(cosine_shift_struct, return_dtype=pl.Float32)
                .alias(f"latent_shift_yoy_{suffix}")
            ]
        )

        df = df.drop([struct_col] + [f"_aef_lag_{i}" for i in range(64)])

    df = df.drop(aef_cols)

    return df


def parse_s2_filename(
    filename: str,
) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    m = re.match(r"^([A-Z0-9]+_\d+_\d+)__s2_l2a_(\d{4})_(\d{1,2})\.tif$", filename)
    return (m.group(1), int(m.group(2)), int(m.group(3))) if m else (None, None, None)


def parse_s1_filename(
    filename: str,
) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[str]]:
    m = re.match(
        r"^([A-Z0-9]+_\d+_\d+)__s1_rtc_(\d{4})_(\d{1,2})_(ascending|descending)\.tif$",
        filename,
    )
    return (
        (m.group(1), int(m.group(2)), int(m.group(3)), m.group(4))
        if m
        else (None, None, None, None)
    )


def get_available_tiles(data_dir: Path, split: str = "train") -> List[str]:
    s2_dir = Path(data_dir) / "sentinel-2" / split
    if not s2_dir.exists():
        return []
    return [
        m.group(1)
        for d in sorted(s2_dir.iterdir())
        if d.is_dir() and (m := re.match(r"^([A-Z0-9]+_\d+_\d+)", d.name))
    ]


def compute_features_pipeline(
    data_dir: Path = None,
    output_path: Path = None,
    sample_fraction: float = SAMPLE_FRACTION,
    seed: int = SEED,
    lookback: int = 6,
    min_obs: int = 3,
) -> pl.DataFrame:
    """Main pipeline."""
    if data_dir is None:
        data_dir = Path(__file__).parent / "data" / "makeathon-challenge"
        print(f"Using default data directory: {data_dir}")
    if output_path is None:
        output_path = Path(__file__).parent / "features.parquet"
        print(f"Using default output path: {output_path}")

    data_dir, output_path = Path(data_dir), Path(output_path)

    all_tiles = [
        (t, s) for s in ["train", "test"] for t in get_available_tiles(data_dir, s)
    ]

    if sample_fraction < 1.0:
        rng = random.Random(seed)
        rng.shuffle(all_tiles)
        all_tiles = all_tiles[: max(1, int(len(all_tiles) * sample_fraction))]

    all_dfs = []
    for tile_id, split in tqdm(all_tiles, desc="Processing tiles"):
        print(f"\n{tile_id} ({split}): loading...")
        df = load_tile_to_dataframe(data_dir, tile_id, split)
        if df.height == 0:
            print(f"  Skipping: no data")
            continue

        print(f"  Computing features ({df.height:,} rows)...")
        df = compute_all_features(df, lookback, min_obs)
        all_dfs.append(df)
        print(f"  Done: {df.shape}")

    if all_dfs:
        final = pl.concat(all_dfs)
        output_cols = [
            "tile_id",
            "split",
            "year",
            "month",
            "pixel_x",
            "pixel_y",
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
            "vv_asc",
            "vv_desc_zscore_6mo",
            "vv_asc_zscore_6mo",
            "vv_desc_diff_mom",
            "vv_asc_diff_mom",
            "vv_desc_roughness_drop",
            "vv_asc_roughness_drop",
            "latent_shift_yoy_1y",
            "latent_shift_yoy_2y",
            "gladl_alert",
            "glads2_alert",
            "radd_alert",
        ]
        final = final.select([c for c in output_cols if c in final.columns])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nWriting to {output_path}...")
        final.write_parquet(output_path)
        return final

    return pl.DataFrame()


if __name__ == "__main__":
    df = compute_features_pipeline()
    print(f"\nGenerated {len(df):,} rows")
    print(f"Columns: {df.columns}")
    print(f"Shape: {df.shape}")
