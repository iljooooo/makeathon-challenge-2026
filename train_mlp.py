"""
MLP training for deforestation detection - batch processing, memory efficient.
"""

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeColumn
from rich.table import Table
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

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


@dataclass
class HyperParams:
    input_dim: int = len(FEATURE_COLUMNS)
    output_dim: int = 1
    hidden_dims: tuple = (256, 128, 64)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 8192
    epochs: int = 10
    dropout: float = 0.2
    pos_weight: float = 8.0

    def to_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dims": list(self.hidden_dims),
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "dropout": self.dropout,
            "pos_weight": self.pos_weight,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HyperParams":
        return cls(
            input_dim=int(data.get("input_dim", len(FEATURE_COLUMNS))),
            output_dim=int(data.get("output_dim", 1)),
            hidden_dims=tuple(data.get("hidden_dims", (256, 128, 64))),
            learning_rate=float(data.get("learning_rate", 1e-3)),
            weight_decay=float(data.get("weight_decay", 1e-5)),
            batch_size=int(data.get("batch_size", 8192)),
            epochs=int(data.get("epochs", 10)),
            dropout=float(data.get("dropout", 0.2)),
            pos_weight=float(data.get("pos_weight", 8.0)),
        )

    def to_json(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_json(cls, path: str) -> "HyperParams":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)


class MLP(nn.Module):
    def __init__(self, hp: HyperParams):
        super().__init__()
        layers = []
        prev_dim = hp.input_dim
        for h in hp.hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(hp.dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, hp.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DataStream:
    def __init__(
        self, processed_dir: Path, split: str, batch_size: int, shuffle: bool = True
    ):
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.files = sorted((self.processed_dir / split).glob("*.parquet"))
        self.imputer = None
        self.scaler = None

    def fit_preprocessors(self) -> tuple:
        sample_df = pl.read_parquet(self.files[0], n_rows=500_000)
        X = sample_df.select(FEATURE_COLUMNS).to_numpy()
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
        self.scaler = StandardScaler().fit(self.imputer.transform(X))
        return self.imputer, self.scaler

    def set_preprocessors(self, imputer, scaler):
        self.imputer = imputer
        self.scaler = scaler

    def iter_batches(self, progress=None, task_id=None):
        indices = list(range(len(self.files)))
        if self.shuffle:
            np.random.shuffle(indices)

        for idx in indices:
            df = pl.read_parquet(self.files[idx])
            X = df.select(FEATURE_COLUMNS).to_numpy()
            y = df.select(TARGET_COLUMN).to_numpy().ravel()
            n_samples = len(X)
            del df

            X = self.scaler.transform(self.imputer.transform(X)).astype(np.float32)
            y = y.astype(np.float32)

            n = len(X)
            batch_indices = np.arange(n)
            if self.shuffle:
                np.random.shuffle(batch_indices)

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch_idx = batch_indices[start:end]
                if progress is not None and task_id is not None:
                    progress.update(task_id, advance=end - start)
                yield X[batch_idx], y[batch_idx]

            del X, n_samples

    def estimate_samples(self) -> int:
        total = 0
        for pf in self.files[:2]:
            try:
                df = pl.scan_parquet(pf)
                total += int(df.select(pl.count()).collect().item())
            except:
                total += 50_000_000
        return total * len(self.files) // min(2, len(self.files))


def train_epoch(
    model, stream, optimizer, criterion, device, hp, progress=None, task_id=None
):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in stream.iter_batches(progress, task_id):
        X_tensor = torch.tensor(X_batch, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y_batch, dtype=torch.float32, device=device)

        optimizer.zero_grad()
        logits = model(X_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, stream, device, hp, progress=None, task_id=None):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in stream.iter_batches(progress, task_id):
            X_tensor = torch.tensor(X_batch, dtype=torch.float32, device=device)
            logits = model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(y_batch)
            all_preds.extend((probs >= 0.5).astype(int))

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
        "auc": float(roc_auc_score(all_labels, all_probs))
        if len(np.unique(all_labels)) > 1
        else 0.0,
        "loss": float(
            -np.mean(
                all_labels * np.log(all_probs + 1e-8)
                - (1 - all_labels) * np.log(1 - all_probs + 1e-8)
            )
        ),
    }


def create_metrics_table() -> Table:
    table = Table(title="Training Metrics")
    table.add_column("Epoch", justify="right", style="cyan", width=6)
    table.add_column("Train Loss", justify="right", style="green", width=10)
    table.add_column("Val Loss", justify="right", style="yellow", width=10)
    table.add_column("Val Acc", justify="right", style="magenta", width=10)
    table.add_column("Val F1", justify="right", style="blue", width=10)
    table.add_column("Val AUC", justify="right", style="red", width=10)
    table.add_column("Duration", justify="right", style="white", width=10)
    return table


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    console = Console()

    hp = HyperParams()
    if args.config:
        hp = HyperParams.from_json(args.config)
    if args.batch_size:
        hp.batch_size = args.batch_size
    if args.epochs:
        hp.epochs = args.epochs
    if args.learning_rate:
        hp.learning_rate = args.learning_rate
    if args.pos_weight:
        hp.pos_weight = args.pos_weight

    console.print(f"[bold cyan]Config:[/] {hp.to_dict()}")

    device = torch.device(args.device)
    console.print(f"[bold cyan]Device:[/] {device}")

    train_stream = DataStream(args.processed_dir, "train", hp.batch_size, shuffle=True)
    val_stream = DataStream(args.processed_dir, "val", hp.batch_size, shuffle=False)
    test_stream = DataStream(args.processed_dir, "test", hp.batch_size, shuffle=False)

    console.print("[bold yellow]Fitting preprocessors on train sample...")
    imputer, scaler = train_stream.fit_preprocessors()
    val_stream.set_preprocessors(imputer, scaler)
    test_stream.set_preprocessors(imputer, scaler)

    model = MLP(hp).to(device)
    optimizer = AdamW(
        model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=hp.epochs, eta_min=hp.learning_rate * 0.01
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(hp.pos_weight, device=device)
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[bold cyan]Model parameters:[/] {n_params:,}")
    console.print(f"[bold cyan]Architecture:[/] {hp.hidden_dims}")

    console.print("\n[bold yellow]Estimating dataset sizes...")
    n_train_est = train_stream.estimate_samples()
    n_val_est = val_stream.estimate_samples()
    n_test_est = test_stream.estimate_samples()
    console.print(f"  Train: ~{n_train_est:,} samples")
    console.print(f"  Val: ~{n_val_est:,} samples")
    console.print(f"  Test: ~{n_test_est:,} samples")

    metrics_table = create_metrics_table()
    best_val_auc = 0.0
    epoch_durations = []

    console.print(f"\n[bold green]Starting training for {hp.epochs} epochs...[/]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeColumn(),
        console=console,
    ) as progress:
        epoch_task = progress.add_task("[cyan]Epochs", total=hp.epochs)

        for epoch in range(hp.epochs):
            epoch_start = time.time()

            progress.update(
                epoch_task, description=f"[cyan]Epoch {epoch + 1}/{hp.epochs}"
            )

            train_task = progress.add_task(f"[green]  Train", total=n_train_est)
            train_loss = train_epoch(
                model,
                train_stream,
                optimizer,
                criterion,
                device,
                hp,
                progress,
                train_task,
            )
            progress.remove_task(train_task)

            val_task = progress.add_task(f"[yellow]  Val", total=n_val_est)
            val_metrics = evaluate(model, val_stream, device, hp, progress, val_task)
            progress.remove_task(val_task)

            epoch_duration = time.time() - epoch_start
            epoch_durations.append(epoch_duration)

            scheduler.step()

            metrics_table.add_row(
                str(epoch + 1),
                f"{train_loss:.4f}",
                f"{val_metrics['loss']:.4f}",
                f"{val_metrics['accuracy']:.4f}",
                f"{val_metrics['f1']:.4f}",
                f"{val_metrics['auc']:.4f}",
                f"{epoch_duration:.1f}s",
            )

            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                args.output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), args.output_dir / "mlp_best.pt")

            progress.advance(epoch_task)

    console.print("\n")
    console.print(metrics_table)

    avg_duration = sum(epoch_durations) / len(epoch_durations)
    console.print(f"\n[bold cyan]Average epoch duration:[/] {avg_duration:.1f}s")

    console.print("\n[bold yellow]Loading best model and evaluating on test...")
    model.load_state_dict(torch.load(args.output_dir / "mlp_best.pt"))
    test_metrics = evaluate(model, test_stream, device, hp)

    test_table = Table(title="Test Results")
    test_table.add_column("Metric", style="cyan", width=15)
    test_table.add_column("Value", justify="right", style="green", width=10)
    test_table.add_row("Accuracy", f"{test_metrics['accuracy']:.4f}")
    test_table.add_row("F1 Score", f"{test_metrics['f1']:.4f}")
    test_table.add_row("AUC", f"{test_metrics['auc']:.4f}")
    console.print(test_table)

    hp.to_json(args.output_dir / "mlp_config.json")
    with open(args.output_dir / "preprocessors.pkl", "wb") as f:
        pickle.dump(
            {"imputer": imputer, "scaler": scaler, "feature_cols": FEATURE_COLUMNS}, f
        )

    console.print(f"\n[bold green]Saved model to {args.output_dir}[/]")


if __name__ == "__main__":
    main()
