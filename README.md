# osapiens Challenge Makeathon 2026

## Detecting Deforestation from Space

![Deforestation event example](content/deforestation.png)

This repository contains the materials for the osapiens Makeathon 2026 challenge on deforestation detection from multimodal satellite data. The goal is to build a system that identifies deforestation events after 2020 using noisy, heterogeneous geospatial inputs and weak supervision signals.

## Start Here

If you are new to the repository, use these files in this order:

1. [osapiens-challenge-full-description.md](./osapiens-challenge-full-description.md) for the written challenge brief and context.
2. [challenge.ipynb](./challenge.ipynb) for the full walkthrough of the dataset structure, label encodings, visualizations, and submission example.
3. [download_data.py](./download_data.py) for the dataset download entrypoint used by the project.

## Repository Guide

- [challenge.ipynb](./challenge.ipynb): Main challenge notebook with data layout, modality descriptions, label definitions, examples, and submission guidance.
- [osapiens-challenge-full-description.md](./osapiens-challenge-full-description.md): Full challenge description.
- [download_data.py](./download_data.py): Downloads the challenge data from S3 into `./data`.
- [submission_utils.py](./submission_utils.py): Utility for converting prediction rasters into submission-ready GeoJSON.
- [feature_computation.py](./feature_computation.py): Generates per-tile feature parquets from S2/S1/AEF data.
- [baseline_data.py](./baseline_data.py): Downloads Hansen GFC data and merges ground truth into feature parquets.
- [train_model.py](./train_model.py): Simple training script for logistic regression and XGBoost models.
- [Makefile](./Makefile): Convenience targets for environment setup and data download.

## Setup

Create the virtual environment and install the dependencies:

```bash
make install
```

Download the dataset:

```bash
make download_data_from_s3
```

This uses [download_data.py](./download_data.py) and stores the files under:

```text
data/makeathon-challenge/
```

## Dataset Layout

After downloading, the notebook expects the data in the following structure:

```text
data/makeathon-challenge/
├── sentinel-1/
│   ├── train/{tile_id}__s1_rtc/{tile_id}__s1_rtc_{year}_{month}_{ascending|descending}.tif
│   └── test/...
├── sentinel-2/
│   ├── train/{tile_id}__s2_l2a/{tile_id}__s2_l2a_{year}_{month}.tif
│   └── test/...
├── aef-embeddings/
│   ├── train/{tile_id}_{year}.tiff
│   └── test/...
├── labels/train/
│   ├── gladl/
│   ├── glads2/
│   └── radd/
└── metadata/
    ├── train_tiles.geojson
    └── test_tiles.geojson
```

## Pipeline

### Complete Dataset Generation

The complete dataset requires running two scripts in sequence:

```bash
# Step 1: Generate per-tile features (~4GB per tile, ~84GB total for 21 tiles)
python feature_computation.py

# Step 2: Add Hansen GFC ground truth
python baseline_data.py -i YOUR_GEE_PROJECT_ID
```

**Output locations:**
```
data/parquet/                    # After step 1
├── {tile_id}.parquet           # Features + weak labels
└── ...

data/parquet_with_baseline/      # After step 2 (COMPLETE DATASET)
├── {tile_id}.parquet           # Features + weak labels + ground truth
└── ...

data/baseline/                   # Hansen GFC GeoTIFFs
├── {tile_id}_treecover2000.tif
├── {tile_id}_lossyear.tif
└── {tile_id}_forest_2020.tif
```

### Feature Columns

| Column | Type | Description |
|--------|------|-------------|
| `tile_id`, `split`, `year`, `month` | str/int | Metadata |
| `pixel_x`, `pixel_y` | int | Pixel coordinates |
| `evi`, `ndmi`, `bsi`, `ndre` | float32 | Optical spectral indices |
| `*_zscore_6mo` | float32 | 6-month rolling z-scores |
| `*_diff_mom` | float32 | Month-over-month differences |
| `vv_desc`, `vv_asc`, ... | float32 | Sentinel-1 radar features |
| `latent_shift_yoy_*` | float32 | AlphaEarth embedding shifts |
| `gladl_alert`, `glads2_alert`, `radd_alert` | float32 | Weak labels (not ground truth) |

### Ground Truth Columns (from Hansen GFC)

| Column | Type | Description |
|--------|------|-------------|
| `deforestation_target` | float32 [0,1,NaN] | Ground truth: 1 if lost by this year, NaN if not forest in 2020 |
| `forest_2020` | float32 [0,1] | Forest mask: 1 if forest in 2020 (filter for training/evaluation) |
| `treecover2000_pct` | float32 [0,1] | Canopy coverage % in 2000 |
| `lossyear_hansen` | float32 | Year of forest loss (0=no loss, 1-24=years 2001-2024) |

### Training

```bash
# Logistic regression
python train_model.py --model logistic --sample-ratio 10

# XGBoost
python train_model.py --model xgboost --sample-ratio 10

# With specific tiles
python train_model.py --model logistic --train-tiles 18NWG_6_6 18NWH_1_4 --test-tiles 18NXH_6_8
```

**Training data preparation:**
- Filters to `forest_2020 == 1` pixels only
- Balances positive/negative samples using `--sample-ratio`
- Excludes NaN features

### CLI Reference

**feature_computation.py:**
```bash
python feature_computation.py \
  --data-dir data/makeathon-challenge \
  --output-dir data/parquet \
  --sample-fraction 1.0 \
  --skip-existing  # Skip already processed tiles
```

**baseline_data.py:**
```bash
python baseline_data.py \
  --id gen-lang-client-0287110623 \
  --data-dir data/makeathon-challenge \
  --baseline-dir data/baseline \
  --parquet-dir data/parquet \
  --output-dir data/parquet_with_baseline \
  --skip-download  # Use cached Hansen data
```

**train_model.py:**
```bash
python train_model.py \
  --parquet-dir data/parquet_with_baseline \
  --model logistic \
  --train-tiles TILE1 TILE2 \
  --test-tiles TILE3 \
  --sample-ratio 10 \
  --output-dir models
```

## Explore the Notebook for the Full Challenge Walkthrough
