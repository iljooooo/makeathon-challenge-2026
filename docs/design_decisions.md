# Design Decisions

## Memory Management

### Dataset Size
- **Total**: ~84GB of processed parquet files
- **Train**: 12 tiles, ~540M rows
- **RAM constraint**: 64GB server

### Approach: Streaming + Online Learning

We cannot load all data into memory at once. Two strategies are implemented:

#### 1. SGDClassifier (Online Logistic Regression) - `train_model.py --online`

```bash
python train_model.py --model logistic --online --sample-frac 1.0 --epochs 3
```

- Uses `partial_fit()` to process one parquet file at a time
- Memory footprint: O(tile_size) instead of O(total_dataset)
- Fits imputer/scaler on first tile, applies to subsequent tiles
- Each epoch visits all tiles sequentially
- Suitable for full dataset training (100% sampling)

#### 2. MLP with Batch Streaming - `train_mlp.py`

```bash
python train_mlp.py --batch-size 8192 --epochs 10
```

- `DataStream` class yields batches without loading entire dataset
- One parquet file loaded at a time, then freed
- Batch size configurable (default 8192)
- StandardScaler and SimpleImputer fitted on first tile sample (500K rows)

---

## Feature Engineering

### Kept Features (~3% NaN)
- `vv_desc`, `vv_desc_zscore_6mo`, `vv_desc_diff_mom`, `vv_roughness_drop_desc` (Sentinel-1 descending orbit)
- All temporal features (`*_zscore_6mo`, `*_diff_mom`)
- `latent_shift_yoy_*` features

### Dropped Features (~43-55% NaN)
- `vv_asc*` columns (Sentinel-1 ascending orbit)

### Imputation Strategy
- **Median imputation** for remaining NaN values
- `keep_empty_features=True` for features with high NaN but still informative

### Hansen GFC Integration
- `lossyear_hansen`: Year of forest loss (0=no loss, 1-24=years 2001-2024)
- `treecover2000_pct`: Canopy cover percentage in year 2000
- Both included as features to help model understand forest history

---

## Target Variable: `deforestation_target`

**Definition**: Binary classification for deforestation occurring *strictly after 2020*.

```
forest_2020 = (treecover2000 > 30) & (lossyear == 0 | lossyear > 20)
deforestation_target = 1 if forest_2020 AND lossyear >= 21 AND current_year >= (2000 + lossyear)
```

- Forest in 2020: Trees existed and no loss recorded before 2021
- Deforestation: Loss detected in years 2021-2024 (lossyear 21-24)
- Non-forest pixels are filtered out (NaN in target)

---

## Train/Val/Test Split

Stratified by tile to prevent data leakage (adjacent pixels from same tile):

| Split | Tiles | Notes |
|-------|-------|-------|
| Train | 12 | `18NUG`, `18NVH`, `18NVK`, `18NWK`, `18NXL`, `18NWN`, `18NXG`, `18NWK`, `48NXA`, `48NWB`, `48NXB`, `47NKB` |
| Val | 4 | `18NWG_6_6`, `18NWH_1_4`, `18NXH_6_8`, `18NWM_9_4` |
| Test | 5 | `18NVJ_1_6`, `18NYH_2_1`, `33NTE_5_1`, `47QMA_6_2`, `48PWA_0_6` |

---

## Class Imbalance

- **Positive class (deforestation)**: ~11% of forest pixels
- **Handling**: 

### Logistic Regression / SGD
```python
class_weights = {0: total/(2*neg), 1: total/(2*pos)}
# or with BCEWithLogitsLoss:
pos_weight = total/pos ≈ 9
```

### MLP
```python
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(8.0))
```

---

## Model Architectures

### Logistic Regression (Baseline)
- Fast training, interpretable
- Good for establishing baseline performance
- SGD variant for memory efficiency

### MLP
- **Architecture**: `input_dim → [hidden_layers] → 1`
- **Default**: `(512, 256, 128)` - funnel architecture
- **Batch Normalization** after each linear layer
- **ReLU activation**, **Dropout (0.2)**
- **AdamW optimizer** with cosine annealing LR scheduler
- **Binary cross-entropy** with class weighting

### XGBoost (Optional)
- Use with sampling due to memory constraints
- `scale_pos_weight` for class imbalance

---

## Evaluation Metrics

- **Primary**: ROC-AUC (robust to class imbalance)
- **Secondary**: F1-Score @ threshold=0.5
- **Precision/Recall** tracked for business context

Threshold optimization via validation set precision-recall curve.

---

## Data Pipeline

```
Raw Data (S2/S1/AEF)
    ↓
feature_computation.py → data/parquet/*.parquet (per-tile)
    ↓
baseline_data.py → merges Hansen GFC targets → data/parquet_with_baseline/
    ↓
preprocessing.py → filters forest_2020, splits train/val/test → data/processed/
    ↓
train_*.py → models/
```

---

## Future Improvements

1. **Spatial cross-validation**: More robust evaluation accounting for spatial autocorrelation
2. **Temporal features**: Better capture seasonal patterns
3. **Ensemble**: Combine LogisticRegression + MLP + XGBoost predictions
4. **Segmentation models**: U-Net for pixel-perfect prediction (requires more GPU memory)
5. **Class weighting tuning**: Experiment with different `pos_weight` values