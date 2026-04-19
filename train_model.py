"""
Training script for deforestation detection - memory-efficient version.

Loads data in chunks, samples for memory efficiency.

Usage:
    python train_model.py --model logistic --sample-frac 0.1
    python train_model.py --model xgboost --sample-frac 0.1 --n-estimators 200
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, Optional, List
import json
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
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
    "vv_desc_zscore_6mo",
    "vv_desc_diff_mom",
    "vv_roughness_drop_desc",
    "latent_shift_yoy_1y",
    "latent_shift_yoy_2y",
    "treecover2000_pct",
    "lossyear_hansen",
]

TARGET_COLUMN = "deforestation_target"

VAL_TILES = ["18NWG_6_6", "18NWH_1_4", "18NXH_6_8", "18NWM_9_4"]
TEST_TILES = ["18NVJ_1_6", "18NYH_2_1", "33NTE_5_1", "47QMA_6_2", "48PWA_0_6"]


def load_split_data(
    processed_dir: Path,
    split: str,
    sample_frac: float = 1.0,
    seed: int = 42,
    max_rows: int = None,
) -> pl.DataFrame:
    """Load split data with memory-efficient sampling."""
    split_dir = processed_dir / split
    parquet_files = sorted(split_dir.glob("*.parquet"))

    if not parquet_files:
        return pl.DataFrame()

    # For small samples, use row limits instead of fraction
    if sample_frac < 0.05 and max_rows is None:
        # Estimate ~10M rows per tile, sample accordingly
        n_tiles = len(parquet_files)
        max_rows = int(100_000_000 * sample_frac / n_tiles)  # per tile

    dfs = []
    for pf in parquet_files:
        if max_rows:
            # Read only first N rows (fast)
            df = pl.read_parquet(pf, n_rows=max_rows)
        else:
            df = pl.read_parquet(pf)

        if sample_frac < 1.0:
            df = df.sample(fraction=sample_frac, seed=seed)
        dfs.append(df)

    return pl.concat(dfs) if len(dfs) > 1 else dfs[0]


def prepare_numpy(
    df: pl.DataFrame,
    imputer: Optional[SimpleImputer] = None,
    scaler: Optional[StandardScaler] = None,
    fit: bool = False,
) -> tuple:
    """Convert to numpy, impute, scale."""
    X = df.select(FEATURE_COLUMNS).to_numpy()
    y = df.select(TARGET_COLUMN).to_numpy().ravel()

    if fit:
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        X = imputer.fit_transform(X)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = imputer.transform(X)
        X = scaler.transform(X)

    return X.astype(np.float32), y.astype(np.float32), imputer, scaler


def train_logistic_regression_batch(X, y, class_weights=None, max_iter=1000):
    """Standard batch training - requires all data in memory."""
    from sklearn.linear_model import LogisticRegression

    print(f"[INFO] Training Logistic Regression on {X.shape[0]:,} samples...")

    model = LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weights,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X, y)

    print(f"[INFO] Done. Coefficients shape: {model.coef_.shape}")
    return model


def train_sgd_online(processed_dir, sample_frac=1.0, max_epochs=10, class_weights=None):
    """True online learning with SGDClassifier - processes data in batches."""
    from sklearn.linear_model import SGDClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    print(f"[INFO] Training SGDClassifier online (sample_frac={sample_frac})...")

    train_dir = Path(processed_dir) / "train"
    parquet_files = sorted(train_dir.glob("*.parquet"))

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()

    classes = np.array([0, 1])
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        learning_rate="optimal",
        eta0=0.01,
        random_state=42,
        warm_start=True,
    )

    sample_weight_val = None
    if class_weights:
        sample_weight_val = {0: class_weights.get(0, 1.0), 1: class_weights.get(1, 1.0)}

    first_pass = True
    for epoch in range(max_epochs):
        print(f"[INFO] Epoch {epoch + 1}/{max_epochs}")

        for pf in parquet_files:
            df = pl.read_parquet(pf)
            if sample_frac < 1.0:
                df = df.sample(fraction=sample_frac, seed=42 + epoch)

            X = df.select(FEATURE_COLUMNS).to_numpy()
            y = df.select(TARGET_COLUMN).to_numpy().ravel()
            del df

            if first_pass:
                imputer.fit(X)
                scaler.fit(imputer.transform(X))
                first_pass = False

            X_imp = imputer.transform(X)
            X_scaled = scaler.transform(X_imp)

            weights = None
            if sample_weight_val:
                weights = np.where(y == 1, sample_weight_val[1], sample_weight_val[0])

            if epoch == 0:
                model.partial_fit(
                    X_scaled.astype(np.float32),
                    y.astype(np.int32),
                    classes=classes,
                    sample_weight=weights,
                )
            else:
                model.partial_fit(
                    X_scaled.astype(np.float32),
                    y.astype(np.int32),
                    sample_weight=weights,
                )

            del X, y, X_imp, X_scaled

        import gc

        gc.collect()

    print(f"[INFO] Done. Coefficients shape: {model.coef_.shape}")
    return model, imputer, scaler


def train_xgboost(
    X, y, class_weights=None, n_estimators=100, max_depth=6, learning_rate=0.1
):
    from xgboost import XGBClassifier

    print(f"[INFO] Training XGBoost on {X.shape[0]:,} samples...")

    scale_pos_weight = None
    if class_weights:
        scale_pos_weight = class_weights[1] / class_weights[0]

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X, y)

    print(f"[INFO] Done. Trees: {n_estimators}")
    return model


def evaluate_model(model, X, y, threshold=0.5):
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
        confusion_matrix,
    )

    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_proba)) if len(np.unique(y)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(y, y_proba))
        if len(np.unique(y)) > 1
        else 0.0,
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }


def find_optimal_threshold(model, X, y):
    from sklearn.metrics import precision_recall_curve

    y_proba = model.predict_proba(X)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]

    return best_threshold, float(best_f1)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="logistic", choices=["logistic", "xgboost"]
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.1,
        help="Fraction of data to use (for memory)",
    )
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--online",
        action="store_true",
        help="Use online SGD training (memory efficient)",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Epochs for online training"
    )

    args = parser.parse_args()

    if args.model == "logistic" and args.online:
        train_tiles = sorted((args.processed_dir / "train").glob("*.parquet"))
        n_tiles = len(train_tiles)
        total_est = n_tiles * 50_000_000
        pos_frac = 0.11
        total_pos = int(total_est * pos_frac * args.sample_frac)
        total_samples = int(total_est * args.sample_frac)
        class_weights = {
            0: total_samples / (2 * (total_samples - total_pos)),
            1: total_samples / (2 * total_pos),
        }
        print(f"[INFO] Estimated class weights: {class_weights}")

        model, imputer, scaler = train_sgd_online(
            args.processed_dir,
            sample_frac=args.sample_frac,
            max_epochs=args.epochs,
            class_weights=class_weights,
        )

        print("[INFO] Loading val data...")
        val_df = load_split_data(
            args.processed_dir, "val", sample_frac=min(0.3, args.sample_frac)
        )
        X_val = imputer.transform(val_df.select(FEATURE_COLUMNS).to_numpy())
        X_val = scaler.transform(X_val).astype(np.float32)
        y_val = val_df.select(TARGET_COLUMN).to_numpy().ravel().astype(np.int32)
        del val_df

        print("[INFO] Loading test data...")
        test_df = load_split_data(
            args.processed_dir, "test", sample_frac=min(0.3, args.sample_frac)
        )
        X_test = imputer.transform(test_df.select(FEATURE_COLUMNS).to_numpy())
        X_test = scaler.transform(X_test).astype(np.float32)
        y_test = test_df.select(TARGET_COLUMN).to_numpy().ravel().astype(np.int32)
        del test_df

        if args.threshold is None:
            threshold, best_f1 = find_optimal_threshold(model, X_val, y_val)
            print(f"[INFO] Optimal threshold: {threshold:.4f} (F1: {best_f1:.4f})")
        else:
            threshold = args.threshold

        val_metrics = evaluate_model(model, X_val, y_val, threshold)
        print(
            f"[INFO] Val: Acc={val_metrics['accuracy']:.4f} F1={val_metrics['f1']:.4f} AUC={val_metrics['roc_auc']:.4f}"
        )

        test_metrics = evaluate_model(model, X_test, y_test, threshold)
        print(
            f"[INFO] Test: Acc={test_metrics['accuracy']:.4f} F1={test_metrics['f1']:.4f} AUC={test_metrics['roc_auc']:.4f}"
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(args.output_dir / "sgd_model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(args.output_dir / "preprocessors.pkl", "wb") as f:
            pickle.dump(
                {"imputer": imputer, "scaler": scaler, "feature_cols": FEATURE_COLUMNS},
                f,
            )
        print(f"[INFO] Saved to {args.output_dir}")
        return

    print(f"[INFO] Loading train data (sample_frac={args.sample_frac})...")
    train_df = load_split_data(
        args.processed_dir, "train", sample_frac=args.sample_frac
    )
    print(f"[INFO] Train: {train_df.height:,} rows")

    # Fit imputer/scaler on train
    X_train, y_train, imputer, scaler = prepare_numpy(train_df, fit=True)
    del train_df

    pos = y_train.sum()
    total = len(y_train)
    print(
        f"[INFO] Train: {total:,} samples, {pos:,.0f} positive ({pos / total * 100:.2f}%)"
    )

    # Class weights
    class_weights = {0: total / (2 * (total - pos)), 1: total / (2 * pos)}
    print(f"[INFO] Class weights: {class_weights}")

    # Load val
    print("[INFO] Loading val data...")
    val_df = load_split_data(args.processed_dir, "val", sample_frac=args.sample_frac)
    X_val, y_val, _, _ = prepare_numpy(val_df, imputer, scaler)
    del val_df

    # Load test
    print("[INFO] Loading test data...")
    test_df = load_split_data(args.processed_dir, "test", sample_frac=args.sample_frac)
    X_test, y_test, _, _ = prepare_numpy(test_df, imputer, scaler)
    del test_df

    # Train
    if args.model == "logistic":
        model = train_logistic_regression_batch(
            X_train, y_train, class_weights, args.max_iter
        )
    else:
        model = train_xgboost(
            X_train,
            y_train,
            class_weights,
            args.n_estimators,
            args.max_depth,
            args.learning_rate,
        )

    # Threshold
    if args.threshold is None:
        threshold, best_f1 = find_optimal_threshold(model, X_val, y_val)
        print(f"[INFO] Optimal threshold: {threshold:.4f} (F1: {best_f1:.4f})")
    else:
        threshold = args.threshold

    # Eval
    val_metrics = evaluate_model(model, X_val, y_val, threshold)
    print(
        f"[INFO] Val: Acc={val_metrics['accuracy']:.4f} P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} F1={val_metrics['f1']:.4f} AUC={val_metrics['roc_auc']:.4f}"
    )

    test_metrics = evaluate_model(model, X_test, y_test, threshold)
    print(
        f"[INFO] Test: Acc={test_metrics['accuracy']:.4f} P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f} F1={test_metrics['f1']:.4f} AUC={test_metrics['roc_auc']:.4f}"
    )

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.output_dir / f"{args.model}_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(args.output_dir / "preprocessors.pkl", "wb") as f:
        pickle.dump(
            {"imputer": imputer, "scaler": scaler, "feature_cols": FEATURE_COLUMNS}, f
        )

    with open(args.output_dir / f"{args.model}_metrics.json", "w") as f:
        json.dump(
            {"val": val_metrics, "test": test_metrics, "threshold": float(threshold)},
            f,
            indent=2,
            default=lambda x: float(x) if hasattr(x, "item") else str(x),
        )

    print(f"[INFO] Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
