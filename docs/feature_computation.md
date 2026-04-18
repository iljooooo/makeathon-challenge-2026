# Feature Computation Design Documentation

## Overview

This document describes the design decisions, reasoning, and methodology for the feature computation pipeline used in the deforestation detection challenge. The pipeline processes multi-source satellite imagery (Sentinel-1, Sentinel-2, and AlphaEarth Foundation embeddings) to generate pixel-level features optimized for machine learning models.

## Data Sources

| Source | Type | Resolution | Temporal Resolution | Bands/Dimensions |
|--------|------|------------|---------------------|------------------|
| Sentinel-2 | Optical multispectral | ~10m | Monthly | 12 bands (B01-B12) |
| Sentinel-1 | SAR radar (VV polarization) | ~30m | Monthly (ascending/descending orbits) | 1 band |
| AlphaEarth Foundation | Learned embeddings | ~10m | Annual | 64-dimensional latent vectors |
| Labels | Alert systems | Variable | Annual (2021-2025) | Binary alerts (gladl, glads2, radd) |

## Feature Categories

### 1. Optical Indices (Sentinel-2)

Derived from spectral bands to capture vegetation health, moisture, and soil characteristics.

#### Enhanced Vegetation Index (EVI)
```
EVI = 2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1)
```
- **Purpose**: Vegetation vigor with improved sensitivity in high-biomass regions
- **Why not NDVI?**: EVI reduces atmospheric influence and soil background effects
- **Key bands**: B08 (NIR), B04 (Red), B02 (Blue)

#### Normalized Difference Moisture Index (NDMI)
```
NDMI = (B08 - B11) / (B08 + B11)
```
- **Purpose**: Vegetation water content and stress detection
- **Deforestation signal**: Dead/drying vegetation shows lower NDMI
- **Key bands**: B08 (NIR), B11 (SWIR)

#### Bare Soil Index (BSI)
```
BSI = ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02))
```
- **Purpose**: Exposed soil detection
- **Deforestation signal**: Cleared areas transition from negative (vegetation) to positive (bare soil)
- **Key bands**: B11 (SWIR), B04 (Red), B08 (NIR), B02 (Blue)

#### Normalized Difference Red Edge (NDRE)
```
NDRE = (B08 - B05) / (B08 + B05)
```
- **Purpose**: Chlorophyll content and leaf area estimation
- **Advantage**: Red-edge band (B05) is sensitive to vegetation stress before visible symptoms
- **Key bands**: B08 (NIR), B05 (Red Edge)

### 2. Temporal Features

#### Z-Score Anomaly (Rolling Window)
```
z_score = (current - hist_mean) / hist_std
```
- **Purpose**: Detect deviations from historical baseline
- **Window**: Configurable (default: 6 months lookback)
- **Minimum observations**: 3 (to ensure statistical reliability)
- **Why rolling window?**: Captures gradual degradation trends, not just sudden changes

#### Month-over-Month Difference
```
diff = current - previous
```
- **Purpose**: Capture sudden changes (logging events, fires)
- **Wraparound handling**: December to January transition across years
- **Why both z-score and diff?**: Z-scores capture sustained anomalies; diffs capture transient events

### 3. Radar Features (Sentinel-1)

#### VV Polarization Roughness Drop
```
roughness_drop = max(historical_median - current, 0)
```
- **Purpose**: Surface roughness change detection
- **Deforestation signal**: Deforested areas become smoother (less backscatter), showing positive roughness drop
- **Orbit handling**: Ascending and descending kept as separate columns (different viewing geometries)

#### Spatial Alignment
- **Method**: Bilinear interpolation resampling via `rasterio.warp.reproject`
- **Why bilinear?**: Gentle low-pass filter that smooths speckle noise while preserving structural edges
- **Master grid**: Sentinel-2 10m grid (1002x1002 pixels)

### 4. Latent Space Features (AlphaEarth Embeddings)

#### Year-over-Year Cosine Distance
```
latent_shift = 1 - cos_similarity(embedding_t, embedding_t_minus_1)
```
- **Purpose**: Detect semantic changes in the learned representation
- **Advantage**: Captures complex multi-spectral patterns not expressible as simple indices
- **Two-year lookback**: Separate features for 1-year and 2-year shifts

### 5. Labels (Training Data Only)

Three independent alert systems with yearly columns (2021-2025):
- **GLAD-L**: Landsat-based alerts
- **GLAD-S2**: Sentinel-2-based alerts  
- **RADD**: Radar-based deforestation alerts

## Data Model

### TileData Structure
```python
@dataclass
class TileData:
    tile_id: str           # e.g., "18NWG_6_6"
    split: str             # "train" or "test"
    s2_data: Dict          # {(year, month): {B02, B04, ...}}
    s1_data: Dict          # {(year, month, orbit): array}
    aef_data: Dict         # {year: 64-band array}
    labels: Dict           # {system: {year: array}}
    reference_shape: Tuple # (height, width)
```

### TileFeatures Structure
```python
@dataclass
class TileFeatures:
    tile_id: str
    year: int
    month: int
    split: str
    # Optical indices
    evi, ndmi, bsi, ndre: np.ndarray
    # Temporal features
    *_zscore, *_diff: np.ndarray
    # Radar features
    vv_desc, vv_asc, vv_*: np.ndarray
    # Latent features
    latent_yoy_1y, latent_yoy_2y: np.ndarray
    # Labels
    gladl_alert, glads2_alert, radd_alert: np.ndarray
```

## Processing Pipeline

### Main Function: `compute_features_pipeline()`

```python
compute_features_pipeline(
    data_dir: Path = "./data/makeathon-challenge",
    output_path: Path = "./features.parquet",
    sample_fraction: float = 1.0,
    seed: int = 42,
    window_config: dict = {"lookback_months": 6, "min_observations": 3}
) -> pl.DataFrame
```

### Sampling Strategy
- **Sample fraction**: If < 1.0, randomly sample a subset of tiles
- **Fixed seed**: Ensures reproducibility (default: 42)
- **Complete temporal domain**: When sampling, selected tiles are processed with all their timesteps

### Processing Flow

```
1. Discover tiles from sentinel-2/train and sentinel-2/test directories
2. Optionally shuffle and sample (if sample_fraction < 1.0)
3. For each tile:
   a. Load all data sources (S2, S1, AEF, labels)
   b. Upsample S1 to S2 grid using bilinear interpolation
   c. For each (year, month) timestep:
      - Compute optical indices
      - Compute temporal features (z-scores, diffs)
      - Compute radar features
      - Compute latent space shifts
      - Attach labels (train only)
      - Flatten pixels to rows
   d. Accumulate DataFrames
4. Concatenate all rows
5. Write to parquet
```

## Modular Functions for Interactive Use

The pipeline is decomposed into modular functions for Jupyter notebook exploration:

### Single-Tile Loading
```python
tile_data = load_tile(data_dir, tile_id="18NWG_6_6", split="train")
```

### Single-Timestep Feature Computation
```python
features = compute_timestep_features(
    tile_data, 
    year=2023, 
    month=6,
    window_config={"lookback_months": 6, "min_observations": 3}
)
```

### Visualization-Ready Output
```python
# Access 2D arrays directly for heatmaps
plt.imshow(features.evi)
plt.imshow(features.vv_roughness_drop_desc)
plt.imshow(features.latent_yoy_1y)
```

## Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `tile_id` | str | Tile identifier |
| `split` | str | "train" or "test" |
| `year` | int | Year (2020-2025) |
| `month` | int | Month (1-12) |
| `pixel_x` | int | Column index |
| `pixel_y` | int | Row index |
| `evi` | float32 | Enhanced Vegetation Index |
| `ndmi` | float32 | Normalized Moisture Index |
| `bsi` | float32 | Bare Soil Index |
| `ndre` | float32 | Red Edge Index |
| `evi_zscore_6mo` | float32 | 6-month rolling z-score |
| `evi_diff_mom` | float32 | Month-over-month difference |
| `vv_desc` | float32 | VV descending backscatter |
| `vv_asc` | float32 | VV ascending backscatter |
| `vv_*_zscore_6mo` | float32 | Radar z-scores |
| `vv_*_diff_mom` | float32 | radar differences |
| `vv_roughness_drop_*` | float32 | Roughness drop signal |
| `latent_shift_yoy_1y` | float32 | 1-year latent shift |
| `latent_shift_yoy_2y` | float32 | 2-year latent shift |
| `*_alert` | float32 | Alert labels (NaN for test) |

## Edge Cases and Handling

| Scenario | Handling |
|----------|----------|
| Missing S1 data | NaN in radar columns |
| Missing AEF year | NaN in latent columns |
| Insufficient historical observations | NaN in z-score columns |
| Missing previous month | NaN in diff columns |
| Test split | NaN in label columns |
| Band division by zero | 0.0 (safe division) |

## Performance Considerations

### Memory Management
- **Tile-by-tile processing**: Each tile processed independently to bound memory
- **Estimated output**: ~16M rows × 32 columns ≈ 2GB raw, ~500MB compressed parquet
- **Streaming option**: Future enhancement for incremental parquet writes

### Computation Optimization
- **Vectorized numpy**: All computations use array operations
- **Lazy historical stack**: Only computed when needed for z-scores
- **Polars DataFrame**: Efficient columnar storage and concatenation

### I/O Optimization
- **Parquet format**: Columnar compression, efficient reads for ML
- **Single output file**: Simplicity for downstream training scripts

## Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| Pixel-level granularity | Enable pixel-wise ML models; finer than tile aggregates |
| Separate orbit columns | Different viewing geometries; model can learn orbit-specific patterns |
| Rolling window + diff | Capture both sustained trends and transient events |
| Bilinear resampling | Smooths speckle noise while preserving edges |
| NaN for missing data | Explicit missingness; models can learn to ignore |
| Configurable window | Flexibility for hyperparameter tuning |
| 6-month lookback | Balance between signal strength and temporal relevance |
| Min 3 observations | Statistical reliability for z-score calculation |