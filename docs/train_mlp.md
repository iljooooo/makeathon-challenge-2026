# MLP Training Documentation

## Overview

This document describes the training pipeline for a Multi-Layer Perceptron (MLP) neural network designed to detect deforestation from satellite imagery features.

## Task Definition

### Objective
Binary classification: Predict whether a pixel location experienced deforestation after 2020, given it was forest in 2020.

### Input
- **20 features** derived from Sentinel-2, Sentinel-1, and auxiliary data:
  - Spectral indices: EVI, NDMI, BSI, NDRE
  - Temporal features: 6-month z-scores, month-over-month differences
  - SAR features: VV descending orbit (backscatter, z-scores, roughness)
  - Latent features: Year-over-year shifts from autoencoder
  - Hansen GFC: Treecover 2000 percentage, loss year

### Output
- Binary probability (0-1) of deforestation
- Threshold: 0.5 default (optimizable via validation set)

### Class Imbalance
- Positive class (deforestation): ~11% of forest pixels
- Negative class (no deforestation): ~89% of forest pixels
- Handled via weighted loss (`pos_weight=8.0`)

---

## Model Architecture

### Network Structure
```
Input (20 features)
    ↓
Linear(20 → 256)
BatchNorm1d(256)
ReLU
Dropout(0.2)
    ↓
Linear(256 → 128)
BatchNorm1d(128)
ReLU
Dropout(0.2)
    ↓
Linear(128 → 64)
BatchNorm1d(64)
ReLU
Dropout(0.2)
    ↓
Linear(64 → 1)
    ↓
Sigmoid → Probability
```

### Hyperparameters (Default)

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `hidden_dims` | (256, 128, 64) | Funnel architecture |
| `learning_rate` | 1e-4 | AdamW optimizer LR (lowered for large batch) |
| `weight_decay` | 1e-5 | L2 regularization |
| `batch_size` | 8192 | Large batch for memory efficiency |
| `epochs` | 10 | Full passes through training data |
| `dropout` | 0.2 | Dropout rate after each hidden layer |
| `pos_weight` | 8.0 | Weight for positive class in BCE loss |

### Parameter Count
- Total: ~47,500 trainable parameters
- Small enough for fast iteration, large enough for learning feature interactions

---

## Training Pipeline

### Data Loading: Memory-Efficient Streaming

**Challenge**: ~510M training samples (~84GB) cannot fit in RAM (64GB server).

**Solution**: `DataStream` class streams data one parquet file at a time.

```python
class DataStream:
    def iter_batches(self, progress=None, task_id=None):
        # Shuffle tile order each epoch
        indices = list(range(len(self.files)))
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Load one tile at a time
        for idx in indices:
            df = pl.read_parquet(self.files[idx])
            X = preprocess(df)  # Impute + Scale
            y = df.select(TARGET_COLUMN).to_numpy()
            
            # Yield batches within tile
            for start in range(0, n, self.batch_size):
                yield X[batch_idx], y[batch_idx]
            
            # Free memory before next tile
            del X, y
```

**Benefits**:
- Constant memory footprint: O(tile_size) instead of O(dataset_size)
- Each epoch sees all data in randomized order
- Progress bar updates per batch for visibility

### Preprocessing

**Applied per-sample**:
1. **Median imputation** for NaN values (3% of VV descending features)
2. **StandardScaler** normalization (fitted on 500K row sample from first tile)

```python
# Fit once on training data
imputer = SimpleImputer(strategy="median", keep_empty_features=True)
scaler = StandardScaler()
imputer.fit(X_train_sample)
scaler.fit(imputer.transform(X_train_sample))

# Apply to all splits
X = scaler.transform(imputer.transform(X_raw))
```

### Training Loop

**Per Epoch**:
1. Shuffle tile order (12 tiles)
2. Load tile → preprocess → yield batches
3. Forward pass, compute loss, backward pass, optimizer step
4. Track EMA loss (α=0.9) for smoother visualization
5. Log progress every N batches (`--log-every 10`)

**Loss Computation**:
```python
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(hp.pos_weight, device=device)
)
# Equivalent to:
# loss = -(pos_weight * y * log(p) + (1-y) * log(1-p))
# for positive samples, weighted by pos_weight
```

**Why `pos_weight=8.0`?**
- Class imbalance ratio: ~89% negative, ~11% positive
- We want the model to pay more attention to positive samples
- `pos_weight = neg_count / pos_count ≈ 8`

**Optimizer**: AdamW
- Adaptive learning rates per parameter
- Weight decay (L2) for regularization
- Learning rate: `1e-4` (conservative for large batch)

**Scheduler**: CosineAnnealingLR
- Starts at `learning_rate`
- Decays to `learning_rate * 0.01` over `epochs`
- Smooth decay helps convergence

### Validation

**Per Epoch**:
1. Load validation tiles (4 tiles, ~195M samples)
2. Forward pass (no gradients)
3. Compute metrics:
   - **Accuracy**: (TP + TN) / N
   - **Recall**: TP / (TP + FN) — sensitivity
   - **Precision**: TP / (TP + FP) — positive predictive value
   - **F1**: 2 * P * R / (P + R) — harmonic mean
   - **AUC**: Area under ROC curve
   - **Val Loss**: Weighted BCE (same as training)

**Best Model Selection**:
- Track best validation AUC
- Save model checkpoint when AUC improves
- Load best model for final test evaluation

### Batch Processing

**Progress Tracking**:
```
Epoch 3/10 | Train Batch  432/2145 ━━━━━━━━━━━━╺━━━━━━━━━━  45% 0:02:34
```

- **Rich progress bars** for epoch and batch-level progress
- **Live updates** every N batches (configurable)
- **Time remaining** estimated based on elapsed time

### Loss Tracking

**Train Loss (Raw)**: Average BCE loss over all batches in epoch
```python
avg_loss = sum(batch_losses) / n_batches
```

**Train Loss (EMA)**: Exponential moving average for smoothing
```python
if ema_loss is None:
    ema_loss = batch_loss
else:
    ema_loss = 0.9 * ema_loss + 0.1 * batch_loss  # α = 0.9
```

**Purpose**: EMA shows trend without batch-level noise. Useful for early stopping detection.

**Val Loss**: Weighted BCE to match training loss
```python
def weighted_bce_loss(labels, probs, pos_weight):
    loss = -(pos_weight * labels * log(probs) + (1 - labels) * log(1 - probs))
    return mean(loss)
```

---

## Evaluation Metrics

### Metrics Legend

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Train Loss** | BCE (raw) | Binary cross entropy averaged per epoch |
| **Train EMA** | 0.9 * prev + 0.1 * current | Smoothed loss trend (α=0.9) |
| **Val Loss** | Weighted BCE | Matches training loss with pos_weight |
| **Accuracy** | (TP + TN) / N | Overall correctness |
| **Recall** | TP / (TP + FN) | Fraction of actual deforestation detected |
| **Precision** | TP / (TP + FP) | Fraction of predicted deforestation that's correct |
| **F1** | 2PR / (P + R) | Balance between precision and recall |
| **AUC** | Area under ROC curve | Model's ability to distinguish classes |

### Loss Color Coding

- **Green**: Loss decreased from previous epoch (improving)
- **Red**: Loss increased from previous epoch (worsening)
- **Cyan**: First epoch (no comparison)

### Output Format

```
╭────────────────────────────────────────── Training Metrics ───────────────────────────────────────────╮
│                                                                                                       │
│  ┏━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┓              │
│  ┃ Epoch ┃ Train Loss ┃ Train EMA ┃ Val Loss ┃  Acc ┃ Recall ┃ Prec ┃   F1 ┃  AUC ┃ Time ┃              │
│  ┡━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━┩              │
│  │   1/10 │    0.1189 │   0.1527 │   0.1928 │ 0.91 │  0.80 │ 0.76 │ 0.78 │ 0.99 │  8m  │              │
│  │   2/10 │    0.0930 │   0.0334 │   0.2060 │ 0.93 │  0.81 │ 0.79 │ 0.80 │ 0.99 │  8m  │              │
│  └─────────┴───────────┴──────────┴──────────┴──────┴────────┴──────┴──────┴──────┴──────┘              │
│  ⠇ Train Batch 50570 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━  81% 0:00:44      │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

---

## Implementation Details

### Weighted Binary Cross-Entropy

**Purpose**: Handle class imbalance by weighting positive samples more.

**Training**:
```python
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(8.0, device=device)
)
```

**Validation** (must match training):
```python
def weighted_bce_loss(labels, probs, pos_weight=8.0):
    """
    Compute weighted BCE to match training loss.
    
    Args:
        labels: Ground truth (0 or 1), shape (N,)
        probs: Predicted probabilities, shape (N,)
        pos_weight: Weight for positive class
    
    Returns:
        Average weighted BCE loss
    """
    eps = 1e-8  # Numerical stability
    loss = -(pos_weight * labels * np.log(probs + eps) 
             + (1 - labels) * np.log(1 - probs + eps))
    return float(np.mean(loss))
```

### Learning Rate Selection

**Why 1e-4?**
- Batch size 8192 is ~16× larger than typical 512
- Linear scaling rule suggests LR × 16, but instability risk
- Conservative choice: `1e-4` for stable training
- Can increase to `5e-4` for faster convergence if monitoring loss

### Batch Size Trade-offs

| Batch Size | Memory | Convergence | Noise | Speed |
|------------|--------|-------------|-------|-------|
| 4096 | Lower | More accurate gradients | More noise | Slower |
| 8192 | Medium | Moderate | Moderate | Balanced |
| 16384 | Higher | Less accurate | Less noise | Faster |

**Default 8192**: Good balance for 64GB RAM server with 510M samples.

### Tile-Based Splitting

**Why tiles?**
- Spatial autocorrelation: adjacent pixels are correlated
- Random split would leak information (nearby pixels in train/val)
- Tile-based split ensures independence

**Split**:
- Train: 12 tiles (~540M samples)
- Val: 4 tiles (~195M samples)
- Test: 5 tiles (~276M samples)

### Memory Management

**Peak Memory Usage**:
- Single tile load: ~2-4GB (depends on tile size)
- Model + gradients: ~500MB
- Batch tensors: ~batch_size × features × 4 bytes ≈ 650KB for batch_size=8192, features=20
- **Total**: <8GB per epoch

**Techniques**:
1. Stream tiles one at a time
2. Delete tile from memory after processing
3. PyTorch automatic gradient cleanup per batch
4. Use float32 (not float64) for features

### Rich Logging Implementation

**Why Rich?**
- Live progress updates
- Clean table formatting
- Color-coded metrics
- Cross-platform (works in terminal, Jupyter, SSH)

**Components**:
```python
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, SpinnerColumn
from rich.table import Table
```

**Live Display**:
```python
with Live(console=console, refresh_per_second=4) as live:
    # Creates Panel with:
    # - Metrics table (completed epochs)
    # - Progress bars (current training)
    live.update(make_panel(...))
```

**Color Coding**:
```python
def colorize_loss(value, prev_value):
    if prev_value is None:
        return f"[cyan]{value:.4f}[/]"      # First epoch
    if value < prev_value:
        return f"[green]{value:.4f}[/]"     # Improvement
    else:
        return f"[red]{value:.4f}[/]"        # Getting worse
```

---

## Usage

### Basic Training

```bash
python train_mlp.py
```

Uses default hyperparameters.

### Custom Configuration

```bash
python train_mlp.py \
    --epochs 30 \
    --batch-size 16384 \
    --learning-rate 5e-5 \
    --pos-weight 10.0 \
    --hidden-dims 512 256 128 \
    --log-every 20
```

### From Config File

```bash
# Create config
cat > config.json << EOF
{
    "hidden_dims": [512, 256, 128],
    "learning_rate": 0.00005,
    "batch_size": 16384,
    "epochs": 30,
    "dropout": 0.3,
    "pos_weight": 10.0
}
EOF

# Train
python train_mlp.py --config config.json
```

### Resume Training

Currently not supported. To resume:
1. Manually load checkpoint
2. Modify optimizer state
3. Continue training loop

---

## Output Files

### Model Checkpoint

**Location**: `models/mlp_best.pt`

```python
torch.save(model.state_dict(), output_dir / "mlp_best.pt")
```

**Contents**: Model weights only (no optimizer state)

### Preprocessors

**Location**: `models/preprocessors.pkl`

```python
{
    "imputer": SimpleImputer,    # Median imputation
    "scaler": StandardScaler,    # Feature normalization
    "feature_cols": FEATURE_COLUMNS  # Column names
}
```

**Usage**:
```python
import pickle

with open("models/preprocessors.pkl", "rb") as f:
    prep = pickle.load(f)

X_imputed = prep["imputer"].transform(X_raw)
X_scaled = prep["scaler"].transform(X_imputed)
```

### Hyperparameters

**Location**: `models/mlp_config.json`

```json
{
    "input_dim": 20,
    "output_dim": 1,
    "hidden_dims": [256, 128, 64],
    "learning_rate": 0.0001,
    "weight_decay": 1e-05,
    "batch_size": 8192,
    "epochs": 10,
    "dropout": 0.2,
    "pos_weight": 8.0
}
```

---

## Troubleshooting

### Out of Memory (OOM)

**Symptoms**: Training crashes with CUDA OOM or system freeze.

**Solutions**:
1. Reduce batch size: `--batch-size 4096`
2. Reduce model size: `--hidden-dims 128 64`
3. Use CPU: `--device cpu` (slower but unlimited memory)

### Loss Oscillates

**Symptoms**: Loss jumps up and down between epochs.

**Causes**:
- Learning rate too high
- Small batch size with high noise

**Solutions**:
1. Reduce LR: `--learning-rate 5e-5`
2. Increase batch size: `--batch-size 16384`
3. Use larger EMA smoothing (modify in code: `ema_loss = 0.95 * ema + 0.05 * loss`)

### Low Recall

**Symptoms**: High accuracy (>95%) but low recall (<50%).

**Causes**:
- Model biased toward majority class (no deforestation)
- `pos_weight` too low

**Solutions**:
1. Increase `pos_weight`: `--pos-weight 15.0`
2. Use focal loss (requires code modification)
3. Adjust threshold (use `find_optimal_threshold()` in `train_model.py`)

### Slow Training

**Symptoms**: Each epoch takes >20 minutes.

**Causes**:
- Large dataset (510M samples)
- CPU bottleneck (data loading)

**Solutions**:
1. Use GPU: ensure `--device cuda`
2. Increase batch size (fewer I/O calls)
3. Pre-load tiles (if RAM allows): currently not implemented

---

## Performance Benchmarks

### Expected Metrics (Default Config)

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | ~93% | ~92% | ~91% |
| Recall | ~80% | ~78% | ~77% |
| Precision | ~76% | ~75% | ~73% |
| F1 | ~78% | ~76% | ~75% |
| AUC | >0.99 | >0.99 | >0.99 |
| Time/Epoch | — | — | ~8-10 min |

### Training Time

- **Per epoch**: 8-10 minutes (510M samples, batch_size=8192)
- **Full training** (10 epochs): ~90 minutes
- **GPU**: NVIDIA (CUDA required)

---

## Future Improvements

1. **Early Stopping**: Stop when val loss doesn't improve for N epochs
2. **Learning Rate Warmup**: Gradually increase LR in first few epochs
3. **Gradient Clipping**: Prevent exploding gradients
4. **Mixed Precision (AMP)**: Faster training with float16
5. **Tile-level Ensemble**: Train separate models per tile, ensemble predictions
6. **Feature Importance**: SHAP or permutation importance for interpretability
7. **Segmentation Model**: U-Net for pixel-perfect prediction (requires more GPU memory)

---

## References

- **Rich documentation**: https://rich.readthedocs.io/
- **PyTorch BCEWithLogitsLoss**: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
- **AdamW optimizer**: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
- **CosineAnnealingLR**: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html