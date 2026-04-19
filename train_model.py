"""
Training script for deforestation detection.

Supports Logistic Regression and XGBoost with proper class weighting
for imbalanced data.

Usage:
    python train_model.py --model logistic
    python train_model.py --model xgboost --n-estimators 200
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
import pickle

from preprocessing import (
    preprocess_pipeline,
    get_feature_columns,
    compute_class_weights,
)
from dataset import ProcessedDataModule, get_class_weights_numpy


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weights: Optional[Dict[int, float]] = None,
    max_iter: int = 1000,
    C: float = 1.0,
):
    """Train Logistic Regression with class weighting."""
    from sklearn.linear_model import LogisticRegression

    print(f"[INFO] Training Logistic Regression on {X_train.shape[0]:,} samples...")
    print(f"[INFO] Feature dimension: {X_train.shape[1]}")

    model = LogisticRegression(
        max_iter=max_iter,
        C=C,
        class_weight=class_weights,
        n_jobs=-1,
        solver="lbfgs",
    )

    model.fit(X_train, y_train)

    print(f"[INFO] Training complete. Coefficients shape: {model.coef_.shape}")

    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weights: Optional[Dict[int, float]] = None,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    scale_pos_weight: Optional[float] = None,
):
    """Train XGBoost with class weighting."""
    from xgboost import XGBClassifier

    print(f"[INFO] Training XGBoost on {X_train.shape[0]:,} samples...")
    print(
        f"[INFO] n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}"
    )

    if scale_pos_weight is None and class_weights is not None:
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

    model.fit(X_train, y_train)

    print(f"[INFO] Training complete. Number of trees: {n_estimators}")

    return model


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """Evaluate model and return metrics."""
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
        confusion_matrix,
    )

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba))
        if len(np.unique(y_test)) > 1
        else 0.0,
        "pr_auc": float(average_precision_score(y_test, y_proba))
        if len(np.unique(y_test)) > 1
        else 0.0,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    return metrics


def find_optimal_threshold(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, Optional[float]]:
    """Find optimal classification threshold on validation set."""
    from sklearn.metrics import precision_recall_curve

    y_proba = model.predict_proba(X_val)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    if metric == "f1":
        best_idx = np.argmax(f1_scores)
    elif metric == "recall":
        best_idx = np.argmax(recalls)
    elif metric == "precision":
        best_idx = np.argmax(precisions)
    else:
        best_idx = np.argmax(f1_scores)

    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_score = f1_scores[best_idx] if metric == "f1" else None

    return best_threshold, float(best_score) if best_score else None


def save_model(
    model,
    model_name: str,
    output_dir: Path,
    metrics: Dict,
    feature_cols: list,
    threshold: float = 0.5,
) -> None:
    """Save model and metrics to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{model_name}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[INFO] Saved model to {model_path}")

    metrics_path = output_dir / f"{model_name}_metrics.json"
    save_metrics = {
        "threshold": threshold,
        "feature_cols": feature_cols,
        **metrics,
    }
    with open(metrics_path, "w") as f:
        json.dump(save_metrics, f, indent=2, default=str)
    print(f"[INFO] Saved metrics to {metrics_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train deforestation models")
    parser.add_argument(
        "--model", type=str, default="logistic", choices=["logistic", "xgboost"]
    )
    parser.add_argument(
        "--parquet-dir", type=Path, help="Input parquet directory (runs preprocessing)"
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Processed data directory",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models"), help="Model output directory"
    )

    parser.add_argument(
        "--max-iter", type=int, default=1000, help="Max iterations (LogisticRegression)"
    )
    parser.add_argument(
        "--n-estimators", type=int, default=100, help="Number of trees (XGBoost)"
    )
    parser.add_argument("--max-depth", type=int, default=6, help="Max depth (XGBoost)")
    parser.add_argument(
        "--learning-rate", type=float, default=0.1, help="Learning rate (XGBoost)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold (auto if None)",
    )

    args = parser.parse_args()

    if args.parquet_dir:
        print(f"[INFO] Running preprocessing pipeline on {args.parquet_dir}")
        data = preprocess_pipeline(
            parquet_dir=args.parquet_dir,
            output_dir=args.processed_dir,
        )
        X_train, y_train = data["train"]
        X_val, y_val = data["val"]
        X_test, y_test = data["test"]
        class_weights = data["class_weights"]
        feature_cols = get_feature_columns()
    else:
        print(f"[INFO] Loading preprocessed data from {args.processed_dir}")
        dm = ProcessedDataModule(processed_dir=args.processed_dir)
        X_train, y_train = dm.get_numpy("train")
        X_val, y_val = dm.get_numpy("val")
        X_test, y_test = dm.get_numpy("test")
        class_weights = dm.get_class_weights()
        feature_cols = dm.feature_cols

    print(
        f"[INFO] Train: {X_train.shape[0]:,} samples, positive: {y_train.sum():,} ({y_train.mean() * 100:.2f}%)"
    )
    print(
        f"[INFO] Val: {X_val.shape[0]:,} samples, positive: {y_val.sum():,} ({y_val.mean() * 100:.2f}%)"
    )
    print(
        f"[INFO] Test: {X_test.shape[0]:,} samples, positive: {y_test.sum():,} ({y_test.mean() * 100:.2f}%)"
    )
    print(f"[INFO] Class weights: {class_weights}")

    if args.model == "logistic":
        model = train_logistic_regression(
            X_train,
            y_train,
            class_weights=class_weights,
            max_iter=args.max_iter,
        )
    elif args.model == "xgboost":
        model = train_xgboost(
            X_train,
            y_train,
            class_weights=class_weights,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print("[INFO] Evaluating on validation set...")
    if args.threshold is None:
        threshold, best_score = find_optimal_threshold(model, X_val, y_val, metric="f1")
        print(f"[INFO] Optimal threshold: {threshold:.4f} (F1: {best_score:.4f})")
    else:
        threshold = args.threshold

    val_metrics = evaluate_model(model, X_val, y_val, threshold=threshold)
    print(f"[INFO] Validation metrics:")
    print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall:    {val_metrics['recall']:.4f}")
    print(f"  F1:        {val_metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {val_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {val_metrics['pr_auc']:.4f}")

    print("[INFO] Evaluating on test set...")
    test_metrics = evaluate_model(model, X_test, y_test, threshold=threshold)
    print(f"[INFO] Test metrics:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {test_metrics['pr_auc']:.4f}")

    metrics = {
        "val": val_metrics,
        "test": test_metrics,
    }

    save_model(model, args.model, args.output_dir, metrics, feature_cols, threshold)

    print(f"[INFO] Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
