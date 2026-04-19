"""
MLP training for deforestation detection - batch processing, memory efficient.
"""

import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from rich.box import HEAVY_HEAD, SIMPLE
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

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
    max_lr: float = 5e-5  # Peak learning rate after warmup
    min_lr: float = 5e-7  # Minimum LR at end of decay (100x smaller than max)
    weight_decay: float = 1e-5
    batch_size: int = 8192
    epochs: int = 10
    dropout: float = 0.2
    pos_weight: float = 8.0
    warmup_proportion: float = 0.2  # Fraction of epochs for warmup

    def to_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dims": list(self.hidden_dims),
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "dropout": self.dropout,
            "pos_weight": self.pos_weight,
            "warmup_proportion": self.warmup_proportion,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HyperParams":
        max_lr = float(
            data.get("max_lr", data.get("learning_rate", 5e-5))
        )  # Backward compat
        return cls(
            input_dim=int(data.get("input_dim", len(FEATURE_COLUMNS))),
            output_dim=int(data.get("output_dim", 1)),
            hidden_dims=tuple(data.get("hidden_dims", (256, 128, 64))),
            max_lr=max_lr,
            min_lr=float(data.get("min_lr", max_lr * 0.01)),
            weight_decay=float(data.get("weight_decay", 1e-5)),
            batch_size=int(data.get("batch_size", 8192)),
            epochs=int(data.get("epochs", 10)),
            dropout=float(data.get("dropout", 0.2)),
            pos_weight=float(data.get("pos_weight", 8.0)),
            warmup_proportion=float(data.get("warmup_proportion", 0.2)),
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


def get_warmup_cosine_schedule(
    optimizer, warmup_epochs: int, total_epochs: int, max_lr: float, min_lr: float
):
    """
    Create a learning rate schedule with linear warmup and cosine decay.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs for warmup
        total_epochs: Total number of training epochs
        max_lr: Peak learning rate after warmup
        min_lr: Minimum learning rate at end of training

    Returns:
        LambdaLR scheduler
    """

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            # Linear warmup: LR goes from max_lr/warmup_epochs to max_lr
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine decay: LR goes from max_lr to min_lr
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            cos_factor = 0.5 * (1 + math.cos(math.pi * progress))
            # Scale: min_lr + (max_lr - min_lr) * cos_factor, then divide by max_lr for lambda
            return min_lr / max_lr + (1 - min_lr / max_lr) * cos_factor

    return LambdaLR(optimizer, lr_lambda)


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

    def _iter_tiles(self):
        """Loads and preprocesses tiles one by one."""
        indices = list(range(len(self.files)))
        if self.shuffle:
            np.random.shuffle(indices)

        for idx in indices:
            df = pl.read_parquet(self.files[idx])
            X = df.select(FEATURE_COLUMNS).to_numpy()
            y = df.select(TARGET_COLUMN).to_numpy().ravel()
            del df

            X = self.scaler.transform(self.imputer.transform(X)).astype(np.float32)
            y = y.astype(np.float32)
            yield X, y

    def _background_iter_tiles(self):
        """Prefetches the next tile in a background thread to prevent GPU starvation."""
        import threading
        import queue

        q = queue.Queue(maxsize=1)

        def worker():
            try:
                for X, y in self._iter_tiles():
                    q.put((True, (X, y)))
                q.put((False, None))
            except Exception as e:
                q.put((False, e))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        while True:
            valid, item = q.get()
            if not valid:
                if item is not None:
                    raise item
                break
            yield item

    def iter_batches(self, progress=None, task_id=None):
        # Prefetching tiles keeps the next tile's IO & preprocessing completely overlapped with GPU training.
        # Yields numpy arrays directly, so switching to JAX/Flax requires no changes here.
        for X, y in self._background_iter_tiles():
            n = len(X)
            batch_indices = np.arange(n)
            if self.shuffle:
                np.random.shuffle(batch_indices)

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch_idx = batch_indices[start:end]
                if progress is not None and task_id is not None:
                    progress.update(task_id, advance=1)
                yield X[batch_idx], y[batch_idx]

            del X, y

    def estimate_samples(self) -> int:
        total = 0
        for pf in self.files[:2]:
            try:
                df = pl.scan_parquet(pf)
                total += int(df.select(pl.len()).collect().item())
            except Exception:
                total += 50_000_000
        avg_per_file = total // max(1, len(self.files[:2]))
        return avg_per_file * len(self.files)

    def estimate_batches(self) -> int:
        n_samples = self.estimate_samples()
        return n_samples // self.batch_size + 1


def train_epoch(
    model,
    stream,
    optimizer,
    criterion,
    device,
    progress,
    train_task,
    log_every=10,
):
    model.train()
    total_loss = 0.0
    n_batches = 0
    ema_loss = None

    for X_batch, y_batch in stream.iter_batches(progress, train_task):
        # torch.from_numpy avoids extra CPU copies before transfer; non_blocking=True pipelines to GPU
        X_tensor = torch.from_numpy(X_batch).to(device, non_blocking=True)
        y_tensor = torch.from_numpy(y_batch).to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(X_tensor)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        n_batches += 1

        if ema_loss is None:
            ema_loss = batch_loss
        else:
            ema_loss = 0.9 * ema_loss + 0.1 * batch_loss

        if n_batches % log_every == 0:
            progress.update(
                train_task,
                description=f"[cyan] Train Batch {n_batches}",
            )

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, ema_loss


def weighted_bce_loss(labels, probs, pos_weight):
    """Compute weighted BCE loss matching training loss."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    eps = 1e-8
    loss = -(
        pos_weight * labels * np.log(probs + eps)
        + (1 - labels) * np.log(1 - probs + eps)
    )
    return float(np.mean(loss))


def evaluate(model, stream, device, progress=None, task_id=None, pos_weight=8.0):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in stream.iter_batches(progress, task_id):
            X_tensor = torch.from_numpy(X_batch).to(device, non_blocking=True)
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
        "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
        "auc": float(roc_auc_score(all_labels, all_probs))
        if len(np.unique(all_labels)) > 1
        else 0.0,
        "loss": weighted_bce_loss(all_labels, all_probs, pos_weight),
    }


def print_metrics_legend(console):
    console.print("\n[bold cyan]Metrics Legend:[/]")
    console.print(
        "  Train Loss = Binary Cross Entropy (BCE) - Raw average loss per epoch"
    )
    console.print("  Train EMA  = Exponential Moving Average - Smoothed loss (α=0.9)")
    console.print(
        "  Val Loss   = Weighted BCE              - Matches training with pos_weight"
    )
    console.print("  Acc        = (TP+TN)/N                 - Overall accuracy")
    console.print(
        "  Recall     = TP/(TP+FN)                - True positive rate (sensitivity)"
    )
    console.print(
        "  Precision  = TP/(TP+FP)                - Positive predictive value"
    )
    console.print(
        "  F1         = 2PR/(P+R)                 - Harmonic mean of precision and recall"
    )
    console.print(
        "  AUC        = Area under ROC curve      - Classifier discrimination ability"
    )
    console.print()


def create_metrics_table():
    table = Table(
        title="",
        box=HEAVY_HEAD,
        show_header=True,
        header_style="bold cyan",
        padding=(0, 1),
    )
    table.add_column("Epoch", justify="right", style="cyan", width=8)
    table.add_column("Train Loss", justify="right", style="", width=11)
    table.add_column("Train EMA", justify="right", style="", width=10)
    table.add_column("Val Loss", justify="right", style="", width=10)
    table.add_column("Acc", justify="right", style="cyan", width=8)
    table.add_column("Recall", justify="right", style="cyan", width=8)
    table.add_column("Prec", justify="right", style="cyan", width=8)
    table.add_column("F1", justify="right", style="cyan", width=8)
    table.add_column("AUC", justify="right", style="cyan", width=8)
    table.add_column("Time", justify="right", style="cyan", width=8)
    return table


def colorize_loss(value, prev_value, is_ema=False):
    if prev_value is None:
        return f"[cyan]{value:.4f}[/]"
    if value < prev_value:
        return f"[green]{value:.4f}[/]"
    else:
        return f"[red]{value:.4f}[/]"


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}m{secs}s"
    else:
        hrs, remainder = divmod(int(seconds), 3600)
        mins, secs = divmod(remainder, 60)
        return f"{hrs}h{mins}m"


def format_samples(n):
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--max-lr", type=float, default=None, help="Peak learning rate after warmup"
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=None,
        help="Minimum learning rate at end of training",
    )
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=10, help="Log every N batches")
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
    if args.max_lr:
        hp.max_lr = args.max_lr
    if args.min_lr:
        hp.min_lr = args.min_lr
    if args.pos_weight:
        hp.pos_weight = args.pos_weight

    config_str = json.dumps(hp.to_dict(), indent=4)
    console.print(f"\n[bold cyan]Neural Network Config:[/] {config_str}")

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

    # Optimizer initialized with max_lr (will be adjusted by scheduler)
    optimizer = AdamW(model.parameters(), lr=hp.max_lr, weight_decay=hp.weight_decay)

    warmup_epochs = max(1, int(hp.epochs * hp.warmup_proportion))
    decay_epochs = hp.epochs - warmup_epochs
    scheduler = get_warmup_cosine_schedule(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=hp.epochs,
        max_lr=hp.max_lr,
        min_lr=hp.min_lr,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(hp.pos_weight, device=device)
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[bold cyan]Model parameters:[/] {n_params:,}")
    console.print(f"[bold cyan]Architecture:[/] {hp.hidden_dims}")
    console.print(
        f"[bold cyan]Learning rate schedule:[/] warmup={warmup_epochs} epochs, decay={decay_epochs} epochs (max_lr={hp.max_lr:.2e} → min_lr={hp.min_lr:.2e})"
    )

    console.print("\n[bold yellow]Estimating dataset sizes...")
    n_train_est = train_stream.estimate_samples()
    n_val_est = val_stream.estimate_samples()
    n_test_est = test_stream.estimate_samples()
    n_train_batches = train_stream.estimate_batches()
    n_val_batches = val_stream.estimate_batches()
    console.print(f"  Train: ~{n_train_est:,} samples (~{n_train_batches:,} batches)")
    console.print(f"  Val: ~{n_val_est:,} samples (~{n_val_batches:,} batches)")
    console.print(f"  Test: ~{n_test_est:,} samples")

    print_metrics_legend(console)

    epoch_data = []
    best_val_auc = 0.0

    progress = Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[cyan][progress.description]{task.description}[/]", justify="left"),
        BarColumn(bar_width=None, complete_style="cyan", finished_style="cyan"),
        TextColumn("[cyan][progress.percentage]{task.percentage:>3.0f}%[/]"),
        TimeRemainingColumn(),
        console=console,
        expand=True,
    )

    def make_panel(
        current_epoch=None, n_batches_done=0, current_loss=None, current_ema=None
    ):
        table = create_metrics_table()
        prev_train_loss = None
        prev_train_ema = None
        prev_val_loss = None

        for i, (tl, tlema, vl, va, vr, vp, vf, vau, t) in enumerate(epoch_data):
            train_loss_str = colorize_loss(tl, prev_train_loss)
            train_ema_str = colorize_loss(tlema, prev_train_ema, is_ema=True)
            val_loss_str = colorize_loss(vl, prev_val_loss)

            table.add_row(
                f"[cyan]{i + 1}/{hp.epochs}[/]",
                train_loss_str,
                train_ema_str,
                val_loss_str,
                f"[cyan]{va:.4f}[/]",
                f"[cyan]{vr:.4f}[/]",
                f"[cyan]{vp:.4f}[/]",
                f"[cyan]{vf:.4f}[/]",
                f"[cyan]{vau:.4f}[/]",
                f"[cyan]{format_time(t)}[/]",
            )

            prev_train_loss = tl
            prev_train_ema = tlema
            prev_val_loss = vl

        if current_epoch is not None and current_loss is not None:
            cur_train_str = colorize_loss(current_loss, prev_train_loss)
            cur_ema_str = (
                colorize_loss(current_ema, prev_train_ema, is_ema=True)
                if current_ema
                else "--"
            )
            table.add_row(
                f"[bold cyan]{current_epoch}/{hp.epochs}[/]",
                f"[bold]{cur_train_str}[/]",
                f"[bold]{cur_ema_str}[/]",
                "[dim]--[/]",
                "[dim]--[/]",
                "[dim]--[/]",
                "[dim]--[/]",
                "[dim]--[/]",
                "[dim]--[/]",
                "[dim]--[/]",
            )

        return Panel(
            Group(table, progress),
            title="[bold blue]Training Metrics[/]",
            border_style="blue",
            padding=(1, 2),
        )

    with Live(console=console, refresh_per_second=4) as live:
        for epoch in range(1, hp.epochs + 1):
            epoch_start = time.time()

            train_task = progress.add_task(
                f"[cyan] Train Batch 0/{n_train_batches}",
                total=n_train_batches,
            )

            live.update(make_panel(epoch, 0, None, None))

            avg_train_loss, ema_train_loss = train_epoch(
                model,
                train_stream,
                optimizer,
                criterion,
                device,
                progress,
                train_task,
                log_every=args.log_every,
            )
            progress.remove_task(train_task)

            val_task = progress.add_task(
                f"[cyan] Val Batch 0/{n_val_batches}",
                total=n_val_batches,
            )
            live.update(
                make_panel(epoch, n_train_batches, avg_train_loss, ema_train_loss)
            )

            val_metrics = evaluate(
                model, val_stream, device, progress, val_task, pos_weight=hp.pos_weight
            )
            progress.remove_task(val_task)

            epoch_duration = time.time() - epoch_start
            epoch_data.append(
                (
                    avg_train_loss,
                    ema_train_loss,
                    val_metrics["loss"],
                    val_metrics["accuracy"],
                    val_metrics["recall"],
                    val_metrics["precision"],
                    val_metrics["f1"],
                    val_metrics["auc"],
                    epoch_duration,
                )
            )

            scheduler.step()

            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                args.output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), args.output_dir / "mlp_best.pt")

            live.update(make_panel())

    if epoch_data:
        avg_duration = sum(e[8] for e in epoch_data) / len(epoch_data)
        console.print(
            f"\n[bold cyan]Average epoch duration:[/] {format_time(avg_duration)}"
        )

    console.print("\n[bold yellow]Loading best model and evaluating on test...")
    model.load_state_dict(torch.load(args.output_dir / "mlp_best.pt"))
    test_metrics = evaluate(model, test_stream, device, pos_weight=hp.pos_weight)

    test_table = Table(title="Test Results", box=SIMPLE, padding=(0, 2))
    test_table.add_column("Metric", style="cyan", width=15)
    test_table.add_column("Value", justify="right", style="cyan", width=10)
    test_table.add_row("Accuracy", f"{test_metrics['accuracy']:.4f}")
    test_table.add_row("Recall", f"{test_metrics['recall']:.4f}")
    test_table.add_row("Precision", f"{test_metrics['precision']:.4f}")
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
