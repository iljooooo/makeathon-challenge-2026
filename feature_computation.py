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
    dot = np.sum(embedding_t * embedding_t_minus_1, axis=-1)
    norm_t = np.linalg.norm(embedding_t, axis=-1)
    norm_t1 = np.linalg.norm(embedding_t_minus_1, axis=-1)
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

        if hist_stack is not None and len(hist_stack) >= min_observations:
            features[f"{idx_name}_zscore"] = compute_zscore_anomaly(current, hist_stack)
        else:
            features[f"{idx_name}_zscore"] = np.full_like(
                current, np.nan, dtype=np.float32
            )

        if previous_indices is not None and idx_name in previous_indices:
            features[f"{idx_name}_diff"] = compute_first_order_difference(
                current, previous_indices[idx_name]
            )
        else:
            features[f"{idx_name}_diff"] = np.full_like(
                current, np.nan, dtype=np.float32
            )

    return features


def compute_radar_features(
    current_vv: np.ndarray,
    historical_vv_stack: Optional[np.ndarray] = None,
    previous_vv: Optional[np.ndarray] = None,
    min_observations: int = 3,
) -> Dict[str, np.ndarray]:
    features = {}

    if historical_vv_stack is not None and len(historical_vv_stack) >= min_observations:
        features["vv_zscore"] = compute_zscore_anomaly(current_vv, historical_vv_stack)
        features["vv_roughness_drop"] = compute_vv_roughness_drop(
            current_vv, historical_vv_stack
        )
    else:
        features["vv_zscore"] = np.full_like(current_vv, np.nan, dtype=np.float32)
        features["vv_roughness_drop"] = np.full_like(
            current_vv, np.nan, dtype=np.float32
        )

    if previous_vv is not None:
        features["vv_diff"] = compute_first_order_difference(current_vv, previous_vv)
    else:
        features["vv_diff"] = np.full_like(current_vv, np.nan, dtype=np.float32)

    return features


def compute_latent_features(
    embedding_t: np.ndarray,
    embedding_t_minus_1: Optional[np.ndarray] = None,
    embedding_t_minus_2: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    features = {}

    if embedding_t_minus_1 is not None:
        features["latent_shift_1y"] = compute_latent_space_shift(
            embedding_t, embedding_t_minus_1
        )
    else:
        features["latent_shift_1y"] = np.full(
            embedding_t.shape[1:], np.nan, dtype=np.float32
        )

    if embedding_t_minus_2 is not None:
        features["latent_shift_2y"] = compute_latent_space_shift(
            embedding_t, embedding_t_minus_2
        )
    else:
        features["latent_shift_2y"] = np.full(
            embedding_t.shape[1:], np.nan, dtype=np.float32
        )

    return features


def parse_tile_id(folder_name: str) -> str:
    tile_id = re.match(r"^([A-Z0-9]+_\d+_\d+)", folder_name)
    return tile_id.group(1) if tile_id else folder_name


def parse_aef_filename(filename: str) -> Tuple[str, int]:
    match = re.match(r"^([A-Z0-9]+_\d+_\d+)_(\d{4})\.tiff?$", filename)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def parse_s2_filename(filename: str) -> Tuple[str, int, int]:
    match = re.match(r"^([A-Z0-9]+_\d+_\d+)_s2_l2a_(\d{4})_(\d{1,2})\.tif$", filename)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))
    return None, None, None


def parse_s1_filename(filename: str) -> Tuple[str, int, int, str]:
    match = re.match(
        r"^([A-Z0-9]+_\d+_\d+)_s1_rtc_(\d{4})_(\d{1,2})_(ascending|descending)\.tif$",
        filename,
    )
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3)), match.group(4)
    return None, None, None, None


def parse_label_filename(filename: str) -> Tuple[str, str, int]:
    match = re.match(
        r"^(gladl|glads2|radd)_([A-Z0-9]+_\d+_\d+)_alert(\d{2})\.tif$", filename
    )
    if match:
        return match.group(2), match.group(1), int(match.group(3))
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
                s2_data[(year, month)] = {
                    "B02": arr[1],
                    "B03": arr[2],
                    "B04": arr[3],
                    "B05": arr[4],
                    "B06": arr[5],
                    "B07": arr[6],
                    "B08": arr[7],
                    "B8A": arr[8],
                    "B09": arr[9],
                    "B11": arr[10],
                    "B12": arr[11],
                    "transform": src.transform,
                    "crs": src.crs,
                    "shape": src.shape,
                }
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
    for year in range(2020, 2026):
        f = aef_dir / f"{tile_id}_{year}.tiff"
        if f.exists():
            with rasterio.open(f) as src:
                arr = src.read().astype(np.float32)
                aef_data[year] = arr

    labels = {}
    if split == "train":
        for label_system in ["gladl", "glads2", "radd"]:
            label_dir = labels_base_dir / label_system
            label_data = {}
            for year in range(2021, 2026):
                f = label_dir / f"{label_system}_{tile_id}_alert{str(year)[-2:]}.tif"
                if f.exists():
                    with rasterio.open(f) as src:
                        label_data[year] = src.read(1).astype(np.float32)
            labels[label_system] = label_data

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


def compute_timestep_features(
    tile_data: TileData,
    year: int,
    month: int,
    window_config: Optional[Dict[str, Any]] = None,
) -> TileFeatures:
    if window_config is None:
        window_config = {"lookback_months": 6, "min_observations": 3}

    lookback = window_config["lookback_months"]
    min_obs = window_config["min_observations"]
    reference_shape = tile_data.reference_shape

    all_timesteps = sorted(tile_data.s2_data.keys())

    s2_month = tile_data.s2_data.get((year, month))
    if s2_month is None:
        return TileFeatures(
            tile_id=tile_data.tile_id, year=year, month=month, split=tile_data.split
        )

    current_indices = compute_optical_indices(s2_month)

    historical_months = get_historical_timesteps(all_timesteps, year, month, lookback)
    historical_indices_stack = {}
    for idx_name in ["evi", "ndmi", "bsi", "ndre"]:
        stack = []
        for hy, hm in historical_months:
            if (hy, hm) in tile_data.s2_data:
                hist_bands = tile_data.s2_data[(hy, hm)]
                hist_indices = compute_optical_indices(hist_bands)
                stack.append(hist_indices[idx_name])
        if stack:
            historical_indices_stack[idx_name] = np.stack(stack)

    prev_month = get_previous_timestep(all_timesteps, year, month)
    previous_indices = None
    if prev_month and prev_month in tile_data.s2_data:
        previous_indices = compute_optical_indices(tile_data.s2_data[prev_month])

    optical_temporal = compute_optical_temporal_features(
        current_indices,
        historical_indices_stack,
        previous_indices,
        min_observations=min_obs,
    )

    vv_desc = np.full(reference_shape, np.nan, dtype=np.float32)
    vv_asc = np.full(reference_shape, np.nan, dtype=np.float32)
    vv_desc_features = {}
    vv_asc_features = {}

    if tile_data.s1_data:
        if (year, month, "descending") in tile_data.s1_data:
            vv_desc = tile_data.s1_data[(year, month, "descending")]
        if (year, month, "ascending") in tile_data.s1_data:
            vv_asc = tile_data.s1_data[(year, month, "ascending")]

        s1_hist_desc = []
        s1_hist_asc = []
        for hy, hm in historical_months:
            if (hy, hm, "descending") in tile_data.s1_data:
                s1_hist_desc.append(tile_data.s1_data[(hy, hm, "descending")])
            if (hy, hm, "ascending") in tile_data.s1_data:
                s1_hist_asc.append(tile_data.s1_data[(hy, hm, "ascending")])

        s1_hist_desc_stack = np.stack(s1_hist_desc) if s1_hist_desc else None
        s1_hist_asc_stack = np.stack(s1_hist_asc) if s1_hist_asc else None

        prev_vv_desc = None
        prev_vv_asc = None
        if prev_month:
            prev_y, prev_m = prev_month
            if (prev_y, prev_m, "descending") in tile_data.s1_data:
                prev_vv_desc = tile_data.s1_data[(prev_y, prev_m, "descending")]
            if (prev_y, prev_m, "ascending") in tile_data.s1_data:
                prev_vv_asc = tile_data.s1_data[(prev_y, prev_m, "ascending")]

        vv_desc_features = compute_radar_features(
            vv_desc, s1_hist_desc_stack, prev_vv_desc, min_observations=min_obs
        )
        vv_asc_features = compute_radar_features(
            vv_asc, s1_hist_asc_stack, prev_vv_asc, min_observations=min_obs
        )

    latent_yoy_1y = np.full(reference_shape, np.nan, dtype=np.float32)
    latent_yoy_2y = np.full(reference_shape, np.nan, dtype=np.float32)

    if tile_data.aef_data:
        if year in tile_data.aef_data:
            embedding_t = tile_data.aef_data[year]
            embedding_t1 = tile_data.aef_data.get(year - 1)
            embedding_t2 = tile_data.aef_data.get(year - 2)
            latent_features = compute_latent_features(
                embedding_t, embedding_t1, embedding_t2
            )
            latent_yoy_1y = latent_features["latent_shift_1y"]
            latent_yoy_2y = latent_features["latent_shift_2y"]

    gladl_alert = np.full(reference_shape, np.nan, dtype=np.float32)
    glads2_alert = np.full(reference_shape, np.nan, dtype=np.float32)
    radd_alert = np.full(reference_shape, np.nan, dtype=np.float32)

    if tile_data.split == "train" and tile_data.labels:
        if "gladl" in tile_data.labels and year in tile_data.labels["gladl"]:
            gladl_alert = tile_data.labels["gladl"][year]
        if "glads2" in tile_data.labels and year in tile_data.labels["glads2"]:
            glads2_alert = tile_data.labels["glads2"][year]
        if "radd" in tile_data.labels and year in tile_data.labels["radd"]:
            radd_alert = tile_data.labels["radd"][year]

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
        vv_desc=vv_desc,
        vv_asc=vv_asc,
        vv_desc_zscore=vv_desc_features.get(
            "vv_zscore", np.full(reference_shape, np.nan, dtype=np.float32)
        ),
        vv_asc_zscore=vv_asc_features.get(
            "vv_zscore", np.full(reference_shape, np.nan, dtype=np.float32)
        ),
        vv_desc_diff=vv_desc_features.get(
            "vv_diff", np.full(reference_shape, np.nan, dtype=np.float32)
        ),
        vv_asc_diff=vv_asc_features.get(
            "vv_diff", np.full(reference_shape, np.nan, dtype=np.float32)
        ),
        vv_roughness_drop_desc=vv_desc_features.get(
            "vv_roughness_drop", np.full(reference_shape, np.nan, dtype=np.float32)
        ),
        vv_roughness_drop_asc=vv_asc_features.get(
            "vv_roughness_drop", np.full(reference_shape, np.nan, dtype=np.float32)
        ),
        latent_yoy_1y=latent_yoy_1y,
        latent_yoy_2y=latent_yoy_2y,
        gladl_alert=gladl_alert,
        glads2_alert=glads2_alert,
        radd_alert=radd_alert,
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
    data_dir: Union[str, Path],
    output_path: Union[str, Path],
    sample_fraction: float = 1.0,
    seed: int = 42,
    window_config: Optional[Dict[str, Any]] = None,
) -> pl.DataFrame:
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
