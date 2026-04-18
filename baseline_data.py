"""
Hansen Global Forest Change 2024 baseline data integration.

Downloads Hansen GFC data via Google Earth Engine, reprojects to match
challenge tile grids, computes forest_2020 mask, and merges into features.parquet.

Target variable:
    deforestation = (lossyear >= 1) & (current_year >= (2000 + lossyear))

Forest 2020 filter:
    forest_2020 = (treecover2000 > threshold) & (lossyear == 0 | lossyear > 20)
"""

import numpy as np
import polars as pl
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import warnings

try:
    import ee

    EE_AVAILABLE = True
except ImportError:
    EE_AVAILABLE = False
    warnings.warn(
        "earthengine-api not installed. Install with: pip install earthengine-api"
    )

FOREST_THRESHOLD = 30
HANSEN_ASSET = "UMD/hansen/global_forest_change_2024_v1_12"


@dataclass
class HansenData:
    tile_id: str
    treecover2000: np.ndarray
    lossyear: np.ndarray
    forest_2020: np.ndarray
    shape: Tuple[int, int]
    transform: Any
    crs: Any


def initialize_earth_engine(project_id: Optional[str] = None) -> bool:
    if not EE_AVAILABLE:
        print("[ERROR] earthengine-api not installed")
        return False
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        print("[INFO] Earth Engine initialized successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Earth Engine initialization failed: {e}")
        return False


def load_challenge_tiles_geojson(geojson_path: Path) -> Dict[str, Dict]:
    tiles = {}
    with open(geojson_path) as f:
        data = json.load(f)
    for feature in data["features"]:
        tile_id = feature["properties"]["name"]
        tiles[tile_id] = {
            "geometry": feature["geometry"],
            "origin": feature["properties"].get("origin", ""),
        }
    return tiles


def get_tile_bounds_wgs84(geometry: Dict) -> Tuple[float, float, float, float]:
    coords = geometry["coordinates"]
    if geometry["type"] == "Polygon":
        flat_coords = [c for ring in coords for c in ring]
    elif geometry["type"] == "MultiPolygon":
        flat_coords = [c for poly in coords for ring in poly for c in ring]
    else:
        raise ValueError(f"Unknown geometry type: {geometry['type']}")

    lons = [c[0] for c in flat_coords]
    lats = [c[1] for c in flat_coords]
    return (min(lons), min(lats), max(lons), max(lats))


def get_reference_grid_from_s2(
    data_dir: Path, tile_id: str, split: str = "train"
) -> Tuple[Any, Any, Tuple[int, int]]:
    s2_dir = data_dir / "sentinel-2" / split / f"{tile_id}__s2_l2a"
    if not s2_dir.exists():
        raise FileNotFoundError(f"S2 directory not found: {s2_dir}")

    tif_files = sorted(s2_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No TIF files in: {s2_dir}")

    with rasterio.open(tif_files[0]) as src:
        return src.transform, src.crs, src.shape


def fetch_hansen_via_ee(
    bounds_wgs84: Tuple[float, float, float, float],
    ref_transform: Any,
    ref_crs: Any,
    ref_shape: Tuple[int, int],
    scale: float = 30.0,
    max_attempts: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    if not EE_AVAILABLE:
        raise RuntimeError("earthengine-api required for Hansen download")

    hansen = ee.Image(HANSEN_ASSET)
    region = ee.Geometry.Rectangle(bounds_wgs84, proj="EPSG:4326")

    treecover = hansen.select("treecover2000").clip(region)
    lossyear = hansen.select("lossyear").clip(region)

    def _download_band(image, band_name, ref_transform, ref_crs, ref_shape, scale):
        from pyproj import Transformer

        height, width = ref_shape
        bounds_pixel = [
            ref_transform * (0, 0),
            ref_transform * (width, 0),
            ref_transform * (width, height),
            ref_transform * (0, height),
        ]

        transformer = Transformer.from_crs(ref_crs, CRS.from_epsg(4326), always_xy=True)
        bounds_wgs = [transformer.transform(x, y) for x, y in bounds_pixel]
        lons = [b[0] for b in bounds_wgs]
        lats = [b[1] for b in bounds_wgs]

        region_ee = ee.Geometry.Rectangle(
            [min(lons), min(lats), max(lons), max(lats)],
            proj="EPSG:4326",
        )

        clipped = image.clip(region_ee)

        for attempt in range(max_attempts):
            try:
                data = clipped.sampleRectangle(
                    region=region_ee,
                    scale=scale,
                ).getInfo()

                arr = np.array(data["properties"][band_name], dtype=np.float32)
                return arr.T
            except Exception as e:
                if attempt < max_attempts - 1:
                    print(f"[WARN] Attempt {attempt + 1} failed for {band_name}: {e}")
                    import time

                    time.sleep(2)
                else:
                    raise

    tree_arr = _download_band(
        treecover, "treecover2000", ref_transform, ref_crs, ref_shape, scale
    )
    loss_arr = _download_band(
        lossyear, "lossyear", ref_transform, ref_crs, ref_shape, scale
    )

    target_shape = ref_shape
    tree_resampled = np.empty(target_shape, dtype=np.float32)
    loss_resampled = np.empty(target_shape, dtype=np.int32)

    from rasterio.transform import from_bounds

    hansen_transform = from_bounds(
        bounds_wgs84[0],
        bounds_wgs84[1],
        bounds_wgs84[2],
        bounds_wgs84[3],
        tree_arr.shape[1],
        tree_arr.shape[0],
    )

    reproject(
        source=tree_arr,
        destination=tree_resampled,
        src_transform=hansen_transform,
        src_crs=CRS.from_epsg(4326),
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.bilinear,
        src_nodata=0,
        dst_nodata=np.nan,
    )

    reproject(
        source=loss_arr,
        destination=loss_resampled,
        src_transform=hansen_transform,
        src_crs=CRS.from_epsg(4326),
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
    )

    return tree_resampled, loss_resampled


def compute_forest_2020_mask(
    treecover2000: np.ndarray,
    lossyear: np.ndarray,
    threshold: int = FOREST_THRESHOLD,
) -> np.ndarray:
    forest_2000 = treecover2000 > threshold
    no_loss_by_2020 = (lossyear == 0) | (lossyear > 20)
    return (forest_2000 & no_loss_by_2020).astype(np.uint8)


def save_hansen_data(
    output_dir: Path,
    tile_id: str,
    treecover2000: np.ndarray,
    lossyear: np.ndarray,
    forest_2020: np.ndarray,
    transform: Any,
    crs: Any,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": treecover2000.shape[1],
        "height": treecover2000.shape[0],
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
        "compress": "lzw",
    }

    with rasterio.open(
        output_dir / f"{tile_id}_treecover2000.tif", "w", **profile
    ) as dst:
        dst.write(treecover2000.astype(np.float32), 1)

    profile["dtype"] = "int16"
    profile["nodata"] = 0
    with rasterio.open(output_dir / f"{tile_id}_lossyear.tif", "w", **profile) as dst:
        dst.write(lossyear.astype(np.int16), 1)

    profile["dtype"] = "uint8"
    with rasterio.open(
        output_dir / f"{tile_id}_forest_2020.tif", "w", **profile
    ) as dst:
        dst.write(forest_2020, 1)


def process_tile_hansen(
    data_dir: Path,
    tile_id: str,
    split: str,
    output_dir: Path,
    use_ee: bool = True,
    hansen_cache_dir: Optional[Path] = None,
) -> Optional[HansenData]:
    try:
        ref_transform, ref_crs, ref_shape = get_reference_grid_from_s2(
            data_dir, tile_id, split
        )
    except FileNotFoundError as e:
        print(f"[WARN] {tile_id}: {e}")
        return None

    output_dir = Path(output_dir)
    tree_path = output_dir / f"{tile_id}_treecover2000.tif"
    loss_path = output_dir / f"{tile_id}_lossyear.tif"
    forest_path = output_dir / f"{tile_id}_forest_2020.tif"

    if tree_path.exists() and loss_path.exists() and forest_path.exists():
        with rasterio.open(tree_path) as src:
            treecover2000 = src.read(1).astype(np.float32)
        with rasterio.open(loss_path) as src:
            lossyear = src.read(1).astype(np.int32)
        with rasterio.open(forest_path) as src:
            forest_2020 = src.read(1).astype(np.uint8)

        return HansenData(
            tile_id=tile_id,
            treecover2000=treecover2000,
            lossyear=lossyear,
            forest_2020=forest_2020,
            shape=ref_shape,
            transform=ref_transform,
            crs=ref_crs,
        )

    geojson_path = data_dir / "metadata" / f"{split}_tiles.geojson"
    tiles_info = load_challenge_tiles_geojson(geojson_path)

    if tile_id not in tiles_info:
        print(f"[WARN] {tile_id} not found in {split}_tiles.geojson")
        return None

    bounds_wgs84 = get_tile_bounds_wgs84(tiles_info[tile_id]["geometry"])

    if use_ee:
        try:
            treecover2000, lossyear = fetch_hansen_via_ee(
                bounds_wgs84, ref_transform, ref_crs, ref_shape
            )
        except Exception as e:
            print(f"[ERROR] {tile_id}: EE fetch failed: {e}")
            return None
    else:
        print(f"[ERROR] {tile_id}: Non-EE mode not implemented yet")
        return None

    forest_2020 = compute_forest_2020_mask(treecover2000, lossyear, FOREST_THRESHOLD)

    save_hansen_data(
        output_dir,
        tile_id,
        treecover2000,
        lossyear,
        forest_2020,
        ref_transform,
        ref_crs,
    )

    return HansenData(
        tile_id=tile_id,
        treecover2000=treecover2000,
        lossyear=lossyear,
        forest_2020=forest_2020,
        shape=ref_shape,
        transform=ref_transform,
        crs=ref_crs,
    )


def compute_temporal_target(
    lossyear: np.ndarray,
    year: int,
    forest_2020: Optional[np.ndarray] = None,
) -> np.ndarray:
    loss_year_actual = 2000 + lossyear
    has_loss = lossyear >= 1
    target = (has_loss & (year >= loss_year_actual)).astype(np.float32)

    if forest_2020 is not None:
        target = np.where(forest_2020 == 1, target, np.nan)

    return target


def merge_hansen_into_features(
    features_parquet: Path,
    baseline_dir: Path,
    output_parquet: Path,
    batch_size: int = 1_000_000,
) -> pl.DataFrame:
    print(f"[INFO] Loading features from {features_parquet}")
    df = pl.read_parquet(features_parquet)

    tile_ids = df.select("tile_id").unique().to_series().to_list()

    hansen_cache = {}
    print(f"[INFO] Loading Hansen data for {len(tile_ids)} tiles")
    for tile_id in tile_ids:
        tree_path = baseline_dir / f"{tile_id}_treecover2000.tif"
        loss_path = baseline_dir / f"{tile_id}_lossyear.tif"
        forest_path = baseline_dir / f"{tile_id}_forest_2020.tif"

        if tree_path.exists() and loss_path.exists() and forest_path.exists():
            with rasterio.open(tree_path) as src:
                treecover2000 = src.read(1).astype(np.float32)
            with rasterio.open(loss_path) as src:
                lossyear = src.read(1).astype(np.int32)
            with rasterio.open(forest_path) as src:
                forest_2020 = src.read(1).astype(np.uint8)

            hansen_cache[tile_id] = {
                "treecover2000": treecover2000,
                "lossyear": lossyear,
                "forest_2020": forest_2020,
            }
        else:
            print(f"[WARN] Hansen data missing for {tile_id}")

    print(f"[INFO] Computing temporal targets and merging")

    result_dfs = []
    unique_tile_year_months = df.select(["tile_id", "year", "month"]).unique()

    for row in unique_tile_year_months.iter_rows():
        tile_id, year, month = row

        if tile_id not in hansen_cache:
            continue

        hansen = hansen_cache[tile_id]

        tile_month_df = df.filter(
            (pl.col("tile_id") == tile_id)
            & (pl.col("year") == year)
            & (pl.col("month") == month)
        )

        if tile_month_df.height == 0:
            continue

        treecover2000 = hansen["treecover2000"]
        lossyear = hansen["lossyear"]
        forest_2020 = hansen["forest_2020"]

        target = compute_temporal_target(lossyear, year, forest_2020)
        forest_2020_mask = forest_2020.astype(np.float32)
        treecover2000_norm = treecover2000 / 100.0

        height, width = target.shape
        n_pixels = tile_month_df.height

        pixel_x = tile_month_df.select("pixel_x").to_series().to_numpy()
        pixel_y = tile_month_df.select("pixel_y").to_series().to_numpy()

        target_flat = target[pixel_y, pixel_x]
        forest_flat = forest_2020_mask[pixel_y, pixel_x]
        treecover_flat = treecover2000_norm[pixel_y, pixel_x]
        lossyear_flat = lossyear[pixel_y, pixel_x].astype(np.float32)

        tile_month_df = tile_month_df.with_columns(
            [
                pl.Series("deforestation_target", target_flat, dtype=pl.Float32),
                pl.Series("forest_2020", forest_flat, dtype=pl.Float32),
                pl.Series("treecover2000_pct", treecover_flat, dtype=pl.Float32),
                pl.Series("lossyear_hansen", lossyear_flat, dtype=pl.Float32),
            ]
        )

        result_dfs.append(tile_month_df)

    if result_dfs:
        result_df = pl.concat(result_dfs)
        output_parquet = Path(output_parquet)
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        result_df.write_parquet(output_parquet)
        print(f"[INFO] Saved merged data to {output_parquet}")
        return result_df
    else:
        print("[WARN] No data to merge")
        return df


def process_all_tiles(
    data_dir: Path,
    output_dir: Path,
    project_id: Optional[str] = None,
) -> Dict[str, HansenData]:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    if not initialize_earth_engine(project_id):
        raise RuntimeError("Failed to initialize Earth Engine")

    results = {}

    for split in ["train", "test"]:
        geojson_path = data_dir / "metadata" / f"{split}_tiles.geojson"
        tiles_info = load_challenge_tiles_geojson(geojson_path)

        for tile_id in tiles_info.keys():
            print(f"[INFO] Processing {tile_id} ({split})")
            hansen_data = process_tile_hansen(
                data_dir=data_dir,
                tile_id=tile_id,
                split=split,
                output_dir=output_dir,
                use_ee=True,
            )
            if hansen_data:
                results[tile_id] = hansen_data
                print(
                    f"  treecover: min={hansen_data.treecover2000[~np.isnan(hansen_data.treecover2000)].min():.1f}, "
                    f"max={hansen_data.treecover2000[~np.isnan(hansen_data.treecover2000)].max():.1f}, "
                    f"forest_2020 pixels: {hansen_data.forest_2020.sum():,}"
                )

    return results


def create_baseline_pipeline(
    data_dir: Path = None,
    baseline_dir: Path = None,
    features_parquet: Path = None,
    output_parquet: Path = None,
    project_id: Optional[str] = None,
    skip_hansen_download: bool = False,
) -> pl.DataFrame:
    if data_dir is None:
        data_dir = Path(__file__).parent / "data" / "makeathon-challenge"
    if baseline_dir is None:
        baseline_dir = Path(__file__).parent / "data" / "baseline"
    if features_parquet is None:
        features_parquet = Path(__file__).parent / "features.parquet"
    if output_parquet is None:
        output_parquet = Path(__file__).parent / "features_with_baseline.parquet"

    data_dir = Path(data_dir)
    baseline_dir = Path(baseline_dir)
    features_parquet = Path(features_parquet)
    output_parquet = Path(output_parquet)

    if not skip_hansen_download:
        print("[INFO] Step 1: Downloading Hansen GFC data via Earth Engine")
        process_all_tiles(data_dir, baseline_dir, project_id)
    else:
        print("[INFO] Step 1: Skipping Hansen download (using cached data)")

    print("[INFO] Step 2: Merging Hansen data into features.parquet")
    result = merge_hansen_into_features(features_parquet, baseline_dir, output_parquet)

    print(f"\n[SUMMARY]")
    print(f"  Total rows: {result.height:,}")
    print(f"  Columns: {len(result.columns)}")
    print(f"  Output: {output_parquet}")

    target_counts = result.filter(pl.col("deforestation_target").is_not_nan()).select(
        [
            pl.col("deforestation_target").sum().alias("positive"),
            ((1 - pl.col("deforestation_target")) * pl.col("forest_2020"))
            .sum()
            .alias("negative"),
        ]
    )
    print(f"  Positives (deforested): {target_counts['positive'][0]:,.0f}")
    print(f"  Negatives (still forest): {target_counts['negative'][0]:,.0f}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hansen GFC baseline data integration")
    parser.add_argument("--data-dir", type=Path, help="Challenge data directory")
    parser.add_argument(
        "--baseline-dir", type=Path, help="Output directory for Hansen data"
    )
    parser.add_argument("--features", type=Path, help="Input features.parquet path")
    parser.add_argument("--output", type=Path, help="Output parquet path")
    parser.add_argument("--project-id", type=str, help="GEE project ID")
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip Hansen download"
    )

    args = parser.parse_args()

    df = create_baseline_pipeline(
        data_dir=args.data_dir,
        baseline_dir=args.baseline_dir,
        features_parquet=args.features,
        output_parquet=args.output,
        project_id=args.project_id,
        skip_hansen_download=args.skip_download,
    )
