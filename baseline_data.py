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

    import requests
    from rasterio.io import MemoryFile

    hansen = ee.Image(HANSEN_ASSET).unmask(0)

    region_geojson = {
        "type": "Polygon",
        "coordinates": [
            [
                [bounds_wgs84[0], bounds_wgs84[1]],
                [bounds_wgs84[2], bounds_wgs84[1]],
                [bounds_wgs84[2], bounds_wgs84[3]],
                [bounds_wgs84[0], bounds_wgs84[3]],
                [bounds_wgs84[0], bounds_wgs84[1]],
            ]
        ],
    }

    both_bands = hansen.select(["treecover2000", "lossyear"])

    for attempt in range(max_attempts):
        try:
            url = both_bands.getDownloadURL(
                params={
                    "region": region_geojson,
                    "scale": scale,
                    "crs": "EPSG:4326",
                    "format": "GEO_TIFF",
                }
            )
            response = requests.get(url, timeout=300)
            response.raise_for_status()

            with MemoryFile(response.content) as memfile:
                with memfile.open() as src:
                    tree_arr = src.read(1).astype(np.float32)
                    loss_arr = src.read(2).astype(np.int32)
                    src_transform = src.transform
                    src_crs = src.crs
            break
        except Exception as e:
            if attempt < max_attempts - 1:
                print(f"[WARN] Attempt {attempt + 1} failed: {e}")
                import time

                time.sleep(10)
            else:
                raise

    import gc

    gc.collect()

    target_shape = ref_shape
    tree_resampled = np.empty(target_shape, dtype=np.float32)
    loss_resampled = np.empty(target_shape, dtype=np.int32)

    reproject(
        source=tree_arr,
        destination=tree_resampled,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.bilinear,
        src_nodata=0,
        dst_nodata=np.nan,
    )

    reproject(
        source=loss_arr,
        destination=loss_resampled,
        src_transform=src_transform,
        src_crs=src_crs,
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


def merge_single_tile_parquet(
    tile_parquet_path: Path,
    baseline_dir: Path,
    output_dir: Path,
) -> Optional[Path]:
    tile_id = tile_parquet_path.stem

    tree_path = baseline_dir / f"{tile_id}_treecover2000.tif"
    loss_path = baseline_dir / f"{tile_id}_lossyear.tif"
    forest_path = baseline_dir / f"{tile_id}_forest_2020.tif"

    if not (tree_path.exists() and loss_path.exists() and forest_path.exists()):
        print(f"  [WARN] {tile_id}: Hansen baseline data not found")
        return None

    with rasterio.open(tree_path) as src:
        treecover2000 = src.read(1).astype(np.float32)
    with rasterio.open(loss_path) as src:
        lossyear = src.read(1).astype(np.int32)
    with rasterio.open(forest_path) as src:
        forest_2020 = src.read(1).astype(np.uint8)

    print(f"  {tile_id}: Loading parquet...")
    df = pl.read_parquet(tile_parquet_path)

    if df.height == 0:
        print(f"  [WARN] {tile_id}: empty parquet")
        return None

    unique_years = df.select("year").unique().to_series().to_list()

    result_dfs = []
    for year in unique_years:
        year_df = df.filter(pl.col("year") == year)
        if year_df.height == 0:
            continue

        target = compute_temporal_target(lossyear, year, forest_2020)
        forest_2020_mask = forest_2020.astype(np.float32)
        treecover2000_norm = treecover2000 / 100.0

        pixel_x = year_df.select("pixel_x").to_series().to_numpy()
        pixel_y = year_df.select("pixel_y").to_series().to_numpy()

        target_flat = target[pixel_y, pixel_x]
        forest_flat = forest_2020_mask[pixel_y, pixel_x]
        treecover_flat = treecover2000_norm[pixel_y, pixel_x]
        lossyear_flat = lossyear[pixel_y, pixel_x].astype(np.float32)

        year_df = year_df.with_columns(
            [
                pl.Series("deforestation_target", target_flat, dtype=pl.Float32),
                pl.Series("forest_2020", forest_flat, dtype=pl.Float32),
                pl.Series("treecover2000_pct", treecover_flat, dtype=pl.Float32),
                pl.Series("lossyear_hansen", lossyear_flat, dtype=pl.Float32),
            ]
        )
        result_dfs.append(year_df)

    if result_dfs:
        result_df = pl.concat(result_dfs)
        output_path = output_dir / f"{tile_id}.parquet"
        output_dir.mkdir(parents=True, exist_ok=True)
        result_df.write_parquet(output_path)
        print(f"  {tile_id}: {result_df.height:,} rows -> {output_path}")

        del df, result_df, result_dfs
        del treecover2000, lossyear, forest_2020
        import gc

        gc.collect()

        return output_path

    return None


def merge_all_tile_parquets(
    parquet_dir: Path,
    baseline_dir: Path,
    output_dir: Path,
    skip_existing: bool = True,
) -> Dict[str, Path]:
    parquet_dir = Path(parquet_dir)
    baseline_dir = Path(baseline_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_parquets = sorted(parquet_dir.glob("*.parquet"))
    print(f"[INFO] Found {len(tile_parquets)} tile parquet files")

    output_paths = {}
    for tile_parquet in tile_parquets:
        tile_id = tile_parquet.stem
        output_path = output_dir / f"{tile_id}.parquet"

        if skip_existing and output_path.exists():
            print(f"  [SKIP] {tile_id} already merged")
            output_paths[tile_id] = output_path
            continue

        result = merge_single_tile_parquet(tile_parquet, baseline_dir, output_dir)
        if result:
            output_paths[tile_id] = result

    return output_paths


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

        import gc

        gc.collect()

    return results


def create_baseline_pipeline(
    data_dir: Path = None,
    baseline_dir: Path = None,
    parquet_dir: Path = None,
    output_dir: Path = None,
    project_id: Optional[str] = None,
    skip_hansen_download: bool = False,
    skip_merge_existing: bool = True,
) -> Dict[str, Path]:
    if data_dir is None:
        data_dir = Path(__file__).parent / "data" / "makeathon-challenge"
    if baseline_dir is None:
        baseline_dir = Path(__file__).parent / "data" / "baseline"
    if parquet_dir is None:
        parquet_dir = Path(__file__).parent / "data" / "parquet"
    if output_dir is None:
        output_dir = Path(__file__).parent / "data" / "parquet_with_baseline"

    data_dir = Path(data_dir)
    baseline_dir = Path(baseline_dir)
    parquet_dir = Path(parquet_dir)
    output_dir = Path(output_dir)

    if not skip_hansen_download:
        print("[INFO] Step 1: Downloading Hansen GFC data via Earth Engine")
        process_all_tiles(data_dir, baseline_dir, project_id)
    else:
        print("[INFO] Step 1: Skipping Hansen download (using cached data)")

    print("[INFO] Step 2: Merging Hansen data into per-tile parquets")
    output_paths = merge_all_tile_parquets(
        parquet_dir, baseline_dir, output_dir, skip_existing=skip_merge_existing
    )

    print(f"\n[SUMMARY]")
    print(f" _tiles merged: {len(output_paths)}")
    print(f"  Output directory: {output_dir}")

    return output_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hansen GFC baseline data integration")
    parser.add_argument("--data-dir", type=Path, help="Challenge data directory")
    parser.add_argument(
        "--baseline-dir", type=Path, help="Output directory for Hansen data"
    )
    parser.add_argument("--parquet-dir", type=Path, help="Input parquet directory")
    parser.add_argument("--output-dir", type=Path, help="Output parquet directory")
    parser.add_argument(
        "-i", "--id", "--project-id", dest="project_id", type=str, help="GEE project ID"
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip Hansen download"
    )
    parser.add_argument(
        "--no-skip-existing", action="store_true", help="Re-merge existing tiles"
    )

    args = parser.parse_args()

    output_paths = create_baseline_pipeline(
        data_dir=args.data_dir,
        baseline_dir=args.baseline_dir,
        parquet_dir=args.parquet_dir,
        output_dir=args.output_dir,
        project_id=args.project_id,
        skip_hansen_download=args.skip_download,
        skip_merge_existing=not args.no_skip_existing,
    )

    print(f"\nGenerated {len(output_paths)} merged parquet files")
