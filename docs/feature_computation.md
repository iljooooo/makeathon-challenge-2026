# Feature Computation Design Documentation

## Overview

This document describes the design decisions, reasoning, and methodology for the feature computation pipeline used in the deforestation detection challenge. The pipeline processes multi-source satellite imagery (Sentinel-1, Sentinel-2, and AlphaEarth Foundation embeddings) to generate pixel-level features optimized for machine learning models.

**Key design goal**: Generalize from training data (Asia, South America) to unseen African test data. Training labels are fuzzy alarm-rate signals (RADD, GLAD-L, GLAD-S2), not direct radar readings or true-color photos. The model must handle mountain shadows, cloud contamination, and different vegetation phenologies across continents.

## Data Sources

| Source | Type | Resolution | Temporal Resolution | Bands/Dimensions |
|--------|------|------------|---------------------|------------------|
| Sentinel-2 | Optical multispectral | ~10m | Monthly | 12 bands (B01-B12) |
| Sentinel-1 | SAR radar (VV polarization) | ~30m | Monthly (ascending/descending orbits) | 1 band |
| AlphaEarth Foundation | Learned embeddings | ~10m | Annual | 64-dimensional latent vectors |
| Labels | Alert systems | Variable | Annual (2021-2025) | Binary alerts (gladl, glads2, radd) |

## Feature Categories

### 1. Optical Indices (Sentinel-2)

Each index was verified against USGS and peer-reviewed literature. Only indices with clear, distinct informational value are included.

#### Enhanced Vegetation Index (EVI)
```
EVI = 2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1)
```
- **Source**: Huete et al. 2002; USGS Landsat EVI product
- **Purpose**: Vegetation vigor with atmospheric and soil background correction
- **Key bands**: B08 (NIR), B04 (Red), B02 (Blue)

#### Normalized Difference Moisture Index (NDMI)
```
NDMI = (B08 - B11) / (B08 + B11)
```
- **Source**: Gao 1996; USGS Landsat NDMI product
- **Deforestation signal**: Dead/drying vegetation shows lower NDMI
- **Key bands**: B08 (NIR), B11 (SWIR1)

#### Bare Soil Index (BSI)
```
BSI = ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02))
```
- **Source**: Rikimaru et al. 2002
- **Deforestation signal**: Cleared areas transition from negative (vegetation) to positive (bare soil)
- **Key bands**: B11 (SWIR1), B04 (Red), B08 (NIR), B02 (Blue)

#### Normalized Difference Red Edge (NDRE)
```
NDRE = (B08 - B05) / (B08 + B05)
```
- **Source**: Gitelson et al. 2003
- **Purpose**: Chlorophyll content; red-edge is sensitive to pre-visible stress
- **Key bands**: B08 (NIR), B05 (Red Edge)

#### Normalized Difference Vegetation Index (NDVI)
```
NDVI = (B08 - B04) / (B08 + B04)
```
- **Source**: Rouse et al. 1974; USGS Landsat NDVI product
- **Why alongside EVI?**: NDVI and EVI saturate differently. NDVI is more sensitive in low-biomass (African savanna) regions; EVI in high-biomass (tropical forest). Complementary.
- **Key bands**: B08 (NIR), B04 (Red)

#### Normalized Burn Ratio (NBR)
```
NBR = (B08 - B12) / (B08 + B12)
```
- **Source**: Key & Benson 2006 (USGS FIREMON); USGS Landsat NBR product
- **Purpose**: Detect burn scars and fire-related deforestation (slash-and-burn, wildfire)
- **Deforestation signal**: Burned areas show strong NBR decrease
- **Key bands**: B08 (NIR), B12 (SWIR2)

#### Soil-Adjusted Vegetation Index (SAVI)
```
SAVI = 1.5 * (B08 - B04) / (B08 + B04 + 0.5)
```
- **Source**: Huete 1988; USGS Landsat SAVI product (L=0.5)
- **Purpose**: Vegetation index correcting for soil brightness — critical for Africa's savannas where soil background is prominent
- **Key bands**: B08 (NIR), B04 (Red)

#### Modified Normalized Difference Water Index (MNDWI)
```
MNDWI = (B03 - B11) / (B03 + B11)
```
- **Source**: Xu 2006
- **Purpose**: Water body detection; suppresses built-up noise. SWIR (B11) replaces NIR to reduce mountain-shadow false positives
- **Mountain/generalization advantage**: Cloud shadows have very low NIR but relatively higher SWIR, so MNDWI (using SWIR) is better than NDWI at distinguishing shadow from water
- **Key bands**: B03 (Green), B11 (SWIR1)

#### Cloud Shadow Ratio
```
cloud_shadow_ratio = B11 / B08
```
- **Purpose**: Proxy feature to help the model distinguish cloud/mountain shadows from true deforestation. Cloud shadows suppress B08 (NIR) much more than B11 (SWIR), producing high ratios. True vegetation has lower ratios.

### 2. Temporal Features

#### Z-Score Anomaly (Rolling Window)
```
z_score = (current - hist_mean) / hist_std
```
- **Window**: 6-month lookback, minimum 3 observations
- **Applied to**: evi, ndmi, bsi, ndre, ndvi, nbr, savi

#### Month-over-Month Difference
```
diff = current - previous_month
```
- **Applied to**: evi, ndmi, bsi, ndre, ndvi, nbr, savi

#### Year-over-Year Same-Month Difference
```
yoy_diff = current_month - same_month_last_year
```
- **Source**: Standard remote sensing change detection methodology
- **Purpose**: Season-cycle-corrected change — compares same phenological state, eliminating seasonal confounds
- **Africa generalization**: Critical across biomes with different dry/wet seasonality
- **Fallback**: If prior-year same month unavailable, tries 2-year lag
- **Applied to**: evi, ndmi, nbr, savi

### 3. Radar Features (Sentinel-1)

#### VV Polarization Roughness Drop
```
roughness_drop = max(historical_median - current, 0)
```
- S1 RTC data is terrain-corrected and cloud-penetrating; inherently robust for mountains

### 4. Latent Space Features (AlphaEarth Embeddings)

#### Year-over-Year Cosine Distance
```
latent_shift = 1 - cos_similarity(embedding_t, embedding_t_minus_k)
```

### 5. Labels (Training Data Only)

Three independent alert systems with yearly columns (2021-2025):
- **GLAD-L**: 0=no loss, 2=probable, 3=confirmed
- **GLAD-S2**: 0-4 confidence scale
- **RADD**: 2=low confidence, 3=high confidence + date

#### Alert Consensus
```
alert_consensus = mean(gladl_alert, glads2_alert, radd_alert)
```
- Soft voting across all three noisy label sources

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
| `ndvi` | float32 | Normalized Difference Vegetation Index |
| `nbr` | float32 | Normalized Burn Ratio |
| `savi` | float32 | Soil-Adjusted Vegetation Index |
| `mndwi` | float32 | Modified NDWI |
| `cloud_shadow_ratio` | float32 | B11/B08 — cloud/shadow proxy |
| `*_zscore_6mo` | float32 | 6-month rolling z-score (7 indices) |
| `*_diff_mom` | float32 | Month-over-month difference (7 indices) |
| `*_yoy_diff` | float32 | Year-over-year same-month diff (evi,ndmi,nbr,savi) |
| `vv_desc` | float32 | VV descending backscatter |
| `vv_asc` | float32 | VV ascending backscatter |
| `vv_*_zscore_6mo` | float32 | Radar z-scores |
| `vv_*_diff_mom` | float32 | Radar differences |
| `vv_roughness_drop_*` | float32 | Roughness drop signal |
| `latent_shift_yoy_1y` | float32 | 1-year latent shift |
| `latent_shift_yoy_2y` | float32 | 2-year latent shift |
| `*_alert` | float32 | Alert labels (NaN for test) |
| `alert_consensus` | float32 | Mean of 3 alert systems |

## Index Selection Rationale

| Index | Kept? | Rationale |
|-------|-------|-----------|
| EVI | Yes | Core vegetation index; corrects atmospheric + soil background |
| NDMI | Yes | Moisture stress — unique info (NIR vs SWIR1) |
| BSI | Yes | Only direct soil exposure signal |
| NDRE | Yes | Only red-edge index — early stress detection |
| NDVI | Yes | Complements EVI in low-biomass (savanna) environments |
| NBR | Yes | Only burn scar index — essential for fire-related deforestation |
| SAVI | Yes | Only soil-corrected vegetation index — key for African savannas |
| MNDWI | Yes | Water detection with mountain shadow suppression |
| cloud_shadow_ratio | Yes | Only cloud/shadow proxy feature |
| NBR2 | **Cut** | Near-redundant with NBR for deforestation |
| MSAVI2 | **Cut** | Redundant with SAVI; buggy impl |
| NDWI | **Cut** | MNDWI strictly supersedes it for shadow rejection |
| VARI | **Cut** | Marginal on L2A data (already atmospherically corrected) |
| brightness | **Cut** | Redundant with EVI (same bands) |
| swir_nir_ratio | **Cut** | Redundant with cloud_shadow_ratio (both SWIR/NIR) |

## Edge Cases and Handling

| Scenario | Handling |
|----------|----------|
| Missing S1 data | NaN in radar columns |
| Missing AEF year | NaN in latent columns |
| Insufficient historical observations | NaN in z-score columns |
| Missing previous month | NaN in diff columns |
| Missing YoY same-month | NaN in yoy_diff columns |
| Test split | NaN in label columns |
| Band division by zero | 0.0 (safe division) |

## Performance Considerations

- **Tile-by-tile processing**: Each tile processed independently to bound memory
- **Estimated output**: ~16M rows × 40 columns ≈ 2.5GB raw, ~600MB compressed parquet
- **Vectorized numpy/polars**: All computations use array operations