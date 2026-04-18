import numpy as np
import polars as pl
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List
from dataclasses import dataclass, field
from tqdm import tqdm
import re
import random

SEED = 42
SAMPLE_FRACTION = 1
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]


def nan_array(shape: Tuple[int, int]) -> np.ndarray:
    return np.full(shape, np.nan, dtype=np.float32)


@dataclass
class TileData:
    tile_id: str
    split: str
    s2_data: Dict[Tuple[int, int], Dict[str, np.ndarray]] = field(default_factory=dict)
    s1_data: Dict[Tuple[int, int, str], np.ndarray] = field(default_factory=dict)
    aef_data: Dict[int, np.ndarray] = field(default_factory=dict)
    labels: Dict[str, Dict[int, np.ndarray]] = field(default_factory=dict)
    reference_shape: Tuple[int, int] = (0, 0)
    reference_transform: Any = None
    reference_crs: Any = None


@dataclass
class TileFeatures:
    tile_id: str
    year: int
    month: int
    split: str = "unknown"
    evi: np.ndarray = None
    ndmi: np.ndarray = None
    bsi: np.ndarray = None
    ndre: np.ndarray = None
    evi_zscore: np.ndarray = None
    ndmi_zscore: np.ndarray = None
    bsi_zscore: np.ndarray = None
    ndre_zscore: np.ndarray = None
    evi_diff: np.ndarray = None
    ndmi_diff: np.ndarray = None
    bsi_diff: np.ndarray = None
    ndre_diff: np.ndarray = None
    vv_desc: np.ndarray = None
    vv_asc: np.ndarray = None
    vv_desc_zscore: np.ndarray = None
    vv_asc_zscore: np.ndarray = None
    vv_desc_diff: np.ndarray = None
    vv_asc_diff: np.ndarray = None
    vv_roughness_drop_desc: np.ndarray = None
    vv_roughness_drop_asc: np.ndarray = None
    latent_yoy_1y: np.ndarray = None
    latent_yoy_2y: np.ndarray = None
    gladl_alert: np.ndarray = None
    glads2_alert: np.ndarray = None
    radd_alert: np.ndarray = None


def compute_evi(b02: np.ndarray, b04: np.ndarray, b08: np.ndarray) -> np.ndarray:
    numerator = b08 - b04
    denominator = b08 + 6.0 * b04 - 7.5 * b02 + 1.0
    return np.where(denominator != 0, 2.5 * numerator / denominator, 0.0)


def compute_ndmi(b08: np.ndarray, b11: np.ndarray) -> np.ndarray:
    denominator = b08 + b11
    return np.where(denominator != 0, (b08 - b11) / denominator, 0.0)


def compute_bsi(
    b02: np.ndarray, b04: np.ndarray, b08: np.ndarray, b11: np.ndarray
) -> np.ndarray:
    numerator = (b11 + b04) - (b08 + b02)
    denominator = (b11 + b04) + (b08 + b02)
    return np.where(denominator != 0, numerator / denominator, 0.0)


def compute_ndre(b05: np.ndarray, b08: np.ndarray) -> np.ndarray:
    denominator = b08 + b05
    return np.where(denominator != 0, (b08 - b05) / denominator, 0.0)


def compute_vv_roughness_drop(
    vv_t: np.ndarray, vv_historical: np.ndarray
) -> np.ndarray:
    historical_median = np.median(vv_historical, axis=0)
    return np.maximum(historical_median - vv_t, 0.0)


def compute_latent_space_shift(
    embedding_t: np.ndarray, embedding_t_minus_1: np.ndarray
) -> np.ndarray:
    dot = np.sum(embedding_t * embedding_t_minus_1, axis=0)
    norm_t = np.linalg.norm(embedding_t, axis=0)
    norm_t1 = np.linalg.norm(embedding_t_minus_1, axis=0)
    denom = norm_t * norm_t1
    cosine_sim = np.where(denom != 0, dot / denom, 0.0)
    return 1.0 - cosine_sim


def compute_zscore_anomaly(
    current_val: np.ndarray, historical_stack: np.ndarray
) -> np.ndarray:
    hist_mean = np.mean(historical_stack, axis=0)
    hist_std = np.std(historical_stack, axis=0)
    return np.where(hist_std != 0, (current_val - hist_mean) / hist_std, 0.0)


def compute_first_order_difference(
    current_val: np.ndarray, previous_val: np.ndarray
) -> np.ndarray:
    return current_val - previous_val


def upsample_and_align_radar(
    source_data: np.ndarray,
    source_transform: Any,
    source_crs: Any,
    reference_transform: Any,
    reference_crs: Any,
    reference_shape: Tuple[int, int],
) -> np.ndarray:
    destination = np.empty(reference_shape, dtype=np.float32)
    reproject(
        source=source_data,
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=reference_transform,
        dst_crs=reference_crs,
        resampling=Resampling.bilinear,
        nodata=np.nan,
    )
    return destination


def compute_optical_indices(s2_bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "evi": compute_evi(s2_bands["B02"], s2_bands["B04"], s2_bands["B08"]),
        "ndmi": compute_ndmi(s2_bands["B08"], s2_bands["B11"]),
        "bsi": compute_bsi(
            s2_bands["B02"], s2_bands["B04"], s2_bands["B08"], s2_bands["B11"]
        ),
        "ndre": compute_ndre(s2_bands["B05"], s2_bands["B08"]),
    }


def compute_optical_temporal_features(
    current_indices: Dict[str, np.ndarray],
    historical_indices_stack: Dict[str, np.ndarray],
    previous_indices: Optional[Dict[str, np.ndarray]] = None,
    min_observations: int = 3,
) -> Dict[str, np.ndarray]:
    features = {}
    for idx_name in ["evi", "ndmi", "bsi", "ndre"]:
        current = current_indices[idx_name]
        hist_stack = historical_indices_stack.get(idx_name)
        features[f"{idx_name}_zscore"] = (
            compute_zscore_anomaly(current, hist_stack)
            if hist_stack is not None and len(hist_stack) >= min_observations
            else nan_array(current.shape)
        )
        features[f"{idx_name}_diff"] = (
            compute_first_order_difference(current, previous_indices[idx_name])
            if previous_indices and idx_name in previous_indices
            else nan_array(current.shape)
        )
    return features


def compute_radar_features(
    current_vv: np.ndarray,
    historical_vv_stack: Optional[np.ndarray] = None,
    previous_vv: Optional[np.ndarray] = None,
    min_observations: int = 3,
) -> Dict[str, np.ndarray]:
    shape = current_vv.shape
    feats = {}
    if historical_vv_stack is not None and len(historical_vv_stack) >= min_observations:
        feats["vv_zscore"] = compute_zscore_anomaly(current_vv, historical_vv_stack)
        feats["vv_roughness_drop"] = compute_vv_roughness_drop(
            current_vv, historical_vv_stack
        )
    else:
        feats["vv_zscore"] = nan_array(shape)
        feats["vv_roughness_drop"] = nan_array(shape)
    feats["vv_diff"] = (
        compute_first_order_difference(current_vv, previous_vv)
        if previous_vv is not None
        else nan_array(shape)
    )
    return feats


def compute_latent_features(
    embedding_t: np.ndarray,
    embedding_t_minus_1: Optional[np.ndarray] = None,
    embedding_t_minus_2: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    shape = embedding_t.shape[1:]
    return {
        "latent_shift_1y": compute_latent_space_shift(embedding_t, embedding_t_minus_1)
        if embedding_t_minus_1 is not None
        else nan_array(shape),
        "latent_shift_2y": compute_latent_space_shift(embedding_t, embedding_t_minus_2)
        if embedding_t_minus_2 is not None
        else nan_array(shape),
    }


def parse_tile_id(folder_name: str) -> str:
    tile_id = re.match(r"^([A-Z0-9]+_\d+_\d+)", folder_name)
    return tile_id.group(1) if tile_id else folder_name


def parse_aef_filename(filename: str) -> Tuple[str, int]:
    match = re.match(r"^([A-Z0-9]+_\d+_\d+)_(\d{4})\.tiff?$", filename)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def parse_s2_filename(filename: str) -> Tuple[str, int, int]:
    match = re.match(r"^([A-Z0-9]+_\d+_\d+)__s2_l2a_(\d{4})_(\d{1,2})\.tif$", filename)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))
    return None, None, None


def parse_s1_filename(filename: str) -> Tuple[str, int, int, str]:
    match = re.match(
        r"^([A-Z0-9]+_\d+_\d+)__s1_rtc_(\d{4})_(\d{1,2})_(ascending|descending)\.tif$",
        filename,
    )
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3)), match.group(4)
    return None, None, None, None


def parse_label_filename(filename: str) -> Tuple[str, str, int]:
    match = re.match(
        r"^(gladl|glads2|radd)_([A-Z0-9]+_\d+_\d+)_(?:alert|labels|alertDate)(\d{2})?\.tif$",
        filename,
    )
    if match:
        year = int(match.group(3)) if match.group(3) else None
        return match.group(2), match.group(1), year
    return None, None, None


def load_tile(
    data_dir: Path,
    tile_id: str,
    split: str = "train",
) -> TileData:
    data_dir = Path(data_dir)

    s2_dir = data_dir / "sentinel-2" / split
    s1_dir = data_dir / "sentinel-1" / split
    aef_dir = data_dir / "aef-embeddings" / split
    labels_base_dir = data_dir / "labels" / "train"

    s2_tile_dir = s2_dir / f"{tile_id}__s2_l2a"
    if not s2_tile_dir.exists():
        return TileData(tile_id=tile_id, split=split)

    s2_data = {}
    reference_transform = None
    reference_crs = None
    reference_shape = None

    for f in sorted(s2_tile_dir.glob("*.tif")):
        parsed = parse_s2_filename(f.name)
        if parsed[0]:
            _, year, month = parsed
            with rasterio.open(f) as src:
                arr = src.read().astype(np.float32)
                bands = {name: arr[i] for i, name in enumerate(S2_BANDS, start=1)}
                bands.update(
                    {"transform": src.transform, "crs": src.crs, "shape": src.shape}
                )
                s2_data[(year, month)] = bands
                if reference_transform is None:
                    reference_transform = src.transform
                    reference_crs = src.crs
                    reference_shape = src.shape

    s1_data = {}
    s1_tile_dir = s1_dir / f"{tile_id}__s1_rtc"
    if s1_tile_dir.exists() and reference_transform is not None:
        for f in sorted(s1_tile_dir.glob("*.tif")):
            parsed = parse_s1_filename(f.name)
            if parsed[0]:
                _, year, month, orbit = parsed
                with rasterio.open(f) as src:
                    arr = src.read(1).astype(np.float32)
                    upsampled = upsample_and_align_radar(
                        source_data=arr,
                        source_transform=src.transform,
                        source_crs=src.crs,
                        reference_transform=reference_transform,
                        reference_crs=reference_crs,
                        reference_shape=reference_shape,
                    )
                    s1_data[(year, month, orbit)] = upsampled

    aef_data = {}
    aef_dir = data_dir / "aef-embeddings" / split
    if reference_transform is not None:
        for year in range(2020, 2026):
            f = aef_dir / f"{tile_id}_{year}.tiff"
            if f.exists():
                with rasterio.open(f) as src:
                    arr = src.read().astype(np.float32)
                    if src.shape != reference_shape:
                        resampled = np.empty(
                            (arr.shape[0], *reference_shape), dtype=np.float32
                        )
                        for band_idx in range(arr.shape[0]):
                            reproject(
                                source=arr[band_idx],
                                destination=resampled[band_idx],
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=reference_transform,
                                dst_crs=reference_crs,
                                resampling=Resampling.bilinear,
                                nodata=np.nan,
                            )
                        aef_data[year] = resampled
                    else:
                        aef_data[year] = arr

    labels = {}
    if split == "train" and reference_transform is not None:

        def _load_label(path, is_yearly=True):
            if not path.exists():
                return {}
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float32)
                if src.shape != reference_shape:
                    resampled = nan_array(reference_shape)
                    reproject(
                        source=arr,
                        destination=resampled,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=reference_transform,
                        dst_crs=reference_crs,
                        resampling=Resampling.nearest,
                    )
                    arr = resampled
                if is_yearly:
                    m = re.search(r"alert(\d{2})\.tif$", path.name)
                    if m:
                        year = 2000 + int(m.group(1))
                        return {year: arr}
                    return {}
                return {y: arr for y in range(2021, 2026)}

        label_dir = labels_base_dir / "gladl"
        label_data = {}
        for yy in range(21, 26):
            f = label_dir / f"gladl_{tile_id}_alert{yy}.tif"
            if f.exists():
                label_data.update(_load_label(f, is_yearly=True))
        labels["gladl"] = label_data

        f = labels_base_dir / "glads2" / f"glads2_{tile_id}_alert.tif"
        labels["glads2"] = _load_label(f, is_yearly=False)

        f = labels_base_dir / "radd" / f"radd_{tile_id}_labels.tif"
        labels["radd"] = _load_label(f, is_yearly=False)

    return TileData(
        tile_id=tile_id,
        split=split,
        s2_data=s2_data,
        s1_data=s1_data,
        aef_data=aef_data,
        labels=labels,
        reference_shape=reference_shape or (0, 0),
        reference_transform=reference_transform,
        reference_crs=reference_crs,
    )


def get_historical_timesteps(
    all_timesteps: List[Tuple[int, int]],
    target_year: int,
    target_month: int,
    lookback_months: int = 6,
) -> List[Tuple[int, int]]:
    historical = []
    for y, m in all_timesteps:
        if (y < target_year) or (y == target_year and m < target_month):
            historical.append((y, m))
    if len(historical) > lookback_months:
        historical = historical[-lookback_months:]
    return historical


def get_previous_timestep(
    all_timesteps: List[Tuple[int, int]],
    target_year: int,
    target_month: int,
) -> Optional[Tuple[int, int]]:
    if target_month > 1:
        prev = (target_year, target_month - 1)
        if prev in all_timesteps:
            return prev
    prev_year = target_year - 1
    if (prev_year, 12) in all_timesteps:
        return (prev_year, 12)
    return None


def _build_historical_indices_stack(
    tile_data: TileData,
    historical_months: List[Tuple[int, int]],
    index_names: List[str],
    min_obs: int,
) -> Dict[str, np.ndarray]:
    stack = {idx: [] for idx in index_names}
    for hy, hm in historical_months:
        if (hy, hm) in tile_data.s2_data:
            indices = compute_optical_indices(tile_data.s2_data[(hy, hm)])
            for idx_name in index_names:
                stack[idx_name].append(indices[idx_name])
    return {idx: np.stack(vals) for idx, vals in stack.items() if len(vals) >= min_obs}


def _build_radar_features_for_orbit(
    tile_data: TileData,
    year: int,
    month: int,
    orbit: str,
    historical_months: List[Tuple[int, int]],
    prev_month: Optional[Tuple[int, int]],
    min_obs: int,
    reference_shape: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    current_vv = tile_data.s1_data.get((year, month, orbit), nan_array(reference_shape))
    hist_stack = [
        tile_data.s1_data[(hy, hm, orbit)]
        for hy, hm in historical_months
        if (hy, hm, orbit) in tile_data.s1_data
    ]
    hist_stack = np.stack(hist_stack) if len(hist_stack) >= min_obs else None
    prev_vv = tile_data.s1_data.get((*prev_month, orbit)) if prev_month else None
    feats = compute_radar_features(current_vv, hist_stack, prev_vv, min_obs)
    feats["vv_current"] = current_vv
    return feats


def _build_label_features(
    tile_data: TileData, year: int, reference_shape: Tuple[int, int]
) -> Dict[str, np.ndarray]:
    labels = {}
    for system in ["gladl", "glads2", "radd"]:
        if system in tile_data.labels and year in tile_data.labels[system]:
            labels[f"{system}_alert"] = tile_data.labels[system][year]
        else:
            labels[f"{system}_alert"] = nan_array(reference_shape)
    return labels


def _build_latent_features(
    tile_data: TileData, year: int, reference_shape: Tuple[int, int]
) -> Dict[str, np.ndarray]:
    if year not in tile_data.aef_data:
        return {
            "latent_shift_1y": nan_array(reference_shape),
            "latent_shift_2y": nan_array(reference_shape),
        }
    feats = compute_latent_features(
        tile_data.aef_data[year],
        tile_data.aef_data.get(year - 1),
        tile_data.aef_data.get(year - 2),
    )
    return {
        "latent_shift_1y": feats["latent_shift_1y"],
        "latent_shift_2y": feats["latent_shift_2y"],
    }


def compute_timestep_features(
    tile_data: TileData,
    year: int,
    month: int,
    window_config: Optional[Dict[str, Any]] = None,
) -> TileFeatures:
    cfg = window_config or {"lookback_months": 6, "min_observations": 3}
    lookback, min_obs = cfg["lookback_months"], cfg["min_observations"]
    ref_shape = tile_data.reference_shape
    all_timesteps = sorted(tile_data.s2_data.keys())

    s2_month = tile_data.s2_data.get((year, month))
    if s2_month is None:
        return TileFeatures(
            tile_id=tile_data.tile_id, year=year, month=month, split=tile_data.split
        )

    current_indices = compute_optical_indices(s2_month)
    historical_months = get_historical_timesteps(all_timesteps, year, month, lookback)
    hist_stack = _build_historical_indices_stack(
        tile_data, historical_months, ["evi", "ndmi", "bsi", "ndre"], min_obs
    )

    prev_month = get_previous_timestep(all_timesteps, year, month)
    prev_indices = (
        compute_optical_indices(tile_data.s2_data[prev_month])
        if prev_month and prev_month in tile_data.s2_data
        else None
    )
    optical_temporal = compute_optical_temporal_features(
        current_indices, hist_stack, prev_indices, min_obs
    )

    vv_features = {}
    for orbit in ["descending", "ascending"]:
        vv_features[orbit] = _build_radar_features_for_orbit(
            tile_data,
            year,
            month,
            orbit,
            historical_months,
            prev_month,
            min_obs,
            ref_shape,
        )

    latent = _build_latent_features(tile_data, year, ref_shape)
    labels = (
        _build_label_features(tile_data, year, ref_shape)
        if tile_data.split == "train"
        else {f"{s}_alert": nan_array(ref_shape) for s in ["gladl", "glads2", "radd"]}
    )

    return TileFeatures(
        tile_id=tile_data.tile_id,
        year=year,
        month=month,
        split=tile_data.split,
        evi=current_indices["evi"],
        ndmi=current_indices["ndmi"],
        bsi=current_indices["bsi"],
        ndre=current_indices["ndre"],
        evi_zscore=optical_temporal["evi_zscore"],
        ndmi_zscore=optical_temporal["ndmi_zscore"],
        bsi_zscore=optical_temporal["bsi_zscore"],
        ndre_zscore=optical_temporal["ndre_zscore"],
        evi_diff=optical_temporal["evi_diff"],
        ndmi_diff=optical_temporal["ndmi_diff"],
        bsi_diff=optical_temporal["bsi_diff"],
        ndre_diff=optical_temporal["ndre_diff"],
        vv_desc=vv_features["descending"].get("vv_current", nan_array(ref_shape)),
        vv_asc=vv_features["ascending"].get("vv_current", nan_array(ref_shape)),
        vv_desc_zscore=vv_features["descending"].get("vv_zscore", nan_array(ref_shape)),
        vv_asc_zscore=vv_features["ascending"].get("vv_zscore", nan_array(ref_shape)),
        vv_desc_diff=vv_features["descending"].get("vv_diff", nan_array(ref_shape)),
        vv_asc_diff=vv_features["ascending"].get("vv_diff", nan_array(ref_shape)),
        vv_roughness_drop_desc=vv_features["descending"].get(
            "vv_roughness_drop", nan_array(ref_shape)
        ),
        vv_roughness_drop_asc=vv_features["ascending"].get(
            "vv_roughness_drop", nan_array(ref_shape)
        ),
        latent_yoy_1y=latent["latent_shift_1y"],
        latent_yoy_2y=latent["latent_shift_2y"],
        gladl_alert=labels["gladl_alert"],
        glads2_alert=labels["glads2_alert"],
        radd_alert=labels["radd_alert"],
    )


def tile_features_to_dataframe(features: TileFeatures) -> pl.DataFrame:
    evi = features.evi
    if evi is None:
        return pl.DataFrame()

    height, width = evi.shape
    pixel_y, pixel_x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    def safe_flatten(arr):
        if arr is None:
            return np.full(height * width, np.nan, dtype=np.float32)
        return arr.flatten().astype(np.float32)

    return pl.DataFrame(
        {
            "tile_id": [features.tile_id] * (height * width),
            "split": [features.split if hasattr(features, "split") else "unknown"]
            * (height * width),
            "year": [features.year] * (height * width),
            "month": [features.month] * (height * width),
            "pixel_x": pixel_x.flatten().astype(np.int32),
            "pixel_y": pixel_y.flatten().astype(np.int32),
            "evi": safe_flatten(features.evi),
            "ndmi": safe_flatten(features.ndmi),
            "bsi": safe_flatten(features.bsi),
            "ndre": safe_flatten(features.ndre),
            "evi_zscore_6mo": safe_flatten(features.evi_zscore),
            "ndmi_zscore_6mo": safe_flatten(features.ndmi_zscore),
            "bsi_zscore_6mo": safe_flatten(features.bsi_zscore),
            "ndre_zscore_6mo": safe_flatten(features.ndre_zscore),
            "evi_diff_mom": safe_flatten(features.evi_diff),
            "ndmi_diff_mom": safe_flatten(features.ndmi_diff),
            "bsi_diff_mom": safe_flatten(features.bsi_diff),
            "ndre_diff_mom": safe_flatten(features.ndre_diff),
            "vv_desc": safe_flatten(features.vv_desc),
            "vv_asc": safe_flatten(features.vv_asc),
            "vv_desc_zscore_6mo": safe_flatten(features.vv_desc_zscore),
            "vv_asc_zscore_6mo": safe_flatten(features.vv_asc_zscore),
            "vv_desc_diff_mom": safe_flatten(features.vv_desc_diff),
            "vv_asc_diff_mom": safe_flatten(features.vv_asc_diff),
            "vv_roughness_drop_desc": safe_flatten(features.vv_roughness_drop_desc),
            "vv_roughness_drop_asc": safe_flatten(features.vv_roughness_drop_asc),
            "latent_shift_yoy_1y": safe_flatten(features.latent_yoy_1y),
            "latent_shift_yoy_2y": safe_flatten(features.latent_yoy_2y),
            "gladl_alert": safe_flatten(features.gladl_alert),
            "glads2_alert": safe_flatten(features.glads2_alert),
            "radd_alert": safe_flatten(features.radd_alert),
        }
    )


def get_available_tiles(
    data_dir: Path,
    split: str = "train",
) -> List[str]:
    data_dir = Path(data_dir)
    s2_dir = data_dir / "sentinel-2" / split
    if not s2_dir.exists():
        return []
    tiles = []
    for tile_dir in sorted(s2_dir.iterdir()):
        if tile_dir.is_dir():
            tile_id = parse_tile_id(tile_dir.name)
            tiles.append(tile_id)
    return tiles


def compute_features_pipeline(
    data_dir: Union[str, Path] = None,
    output_path: Union[str, Path] = None,
    sample_fraction: float = SAMPLE_FRACTION,
    seed: int = SEED,
    window_config: Optional[Dict[str, Any]] = None,
) -> pl.DataFrame:
    if data_dir is None:
        data_dir = Path(__file__).parent / "data" / "makeathon-challenge"
        print(f"Using default data directory: {data_dir}")
    if output_path is None:
        output_path = Path(__file__).parent / "features.parquet"
        print(f"Using default output path: {output_path}")

    data_dir = Path(data_dir)
    output_path = Path(output_path)

    if window_config is None:
        window_config = {"lookback_months": 6, "min_observations": 3}

    all_tiles = []
    for split in ["train", "test"]:
        tiles = get_available_tiles(data_dir, split)
        for tile_id in tiles:
            all_tiles.append((tile_id, split))

    if sample_fraction < 1.0:
        rng = random.Random(seed)
        rng.shuffle(all_tiles)
        n_tiles = max(1, int(len(all_tiles) * sample_fraction))
        all_tiles = all_tiles[:n_tiles]

    all_rows = []

    for tile_id, split in tqdm(all_tiles, desc="Processing tiles"):
        tile_data = load_tile(data_dir, tile_id, split)
        if not tile_data.s2_data:
            continue

        timesteps = sorted(tile_data.s2_data.keys())
        years = sorted(set(y for y, m in timesteps))

        for year in years:
            months = sorted([m for y, m in timesteps if y == year])
            for month in months:
                if (year, month) not in tile_data.s2_data:
                    continue

                features = compute_timestep_features(
                    tile_data, year, month, window_config
                )
                tile_df = tile_features_to_dataframe(features)
                if tile_df.height > 0:
                    tile_df = tile_df.with_columns(pl.lit(split).alias("split"))
                    all_rows.append(tile_df)

    if all_rows:
        final_df = pl.concat(all_rows)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.write_parquet(output_path)
        return final_df
    else:
        return pl.DataFrame()


if __name__ == "__main__":
    feature_dataframe = compute_features_pipeline()
    print(f"Generated {len(feature_dataframe)} rows")
    print(f"Columns: {feature_dataframe.columns}")
    print(f"Shape: {feature_dataframe.shape}")
