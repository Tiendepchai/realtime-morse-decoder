from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # noqa: BLE001
    plt = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # noqa: BLE001
    SummaryWriter = None


def _setup_plot_style() -> None:
    if plt is None:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
        }
    )


def _nice_upper_bound(value: float, minimum: float) -> float:
    value = float(max(value, 0.0))
    if value <= 0.0:
        return float(minimum)

    candidate = value * 1.12
    magnitude = 10 ** np.floor(np.log10(candidate))
    residual = candidate / magnitude
    if residual <= 1:
        nice = 1
    elif residual <= 2:
        nice = 2
    elif residual <= 5:
        nice = 5
    else:
        nice = 10
    return float(max(minimum, nice * magnitude))


def _annotate_axis(ax, summary: str) -> None:
    ax.text(
        0.02,
        0.98,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#444444",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#ffffff", "edgecolor": "#dddddd", "alpha": 0.95},
    )


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_cer: float
    learning_rate: float
    epoch_time_sec: float
    augment_enabled: bool
    blank_logit_bias: float
    grad_accum_steps: int
    amp_enabled: bool


def metrics_to_dataframe(history: Iterable[EpochMetrics]) -> pd.DataFrame:
    rows = [asdict(metric) for metric in history]
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return dataframe
    dataframe["augment_enabled"] = dataframe["augment_enabled"].astype(int)
    dataframe["amp_enabled"] = dataframe["amp_enabled"].astype(int)
    dataframe["val_cer_percent"] = dataframe["val_cer"] * 100.0
    return dataframe


def write_metrics_csv(history: Iterable[EpochMetrics], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_to_dataframe(history).to_csv(output_path, index=False)
    return output_path


def write_metrics_json(history: Iterable[EpochMetrics], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(metric) for metric in history]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def save_training_plots(history: Iterable[EpochMetrics], output_dir: str | Path) -> list[Path]:
    if plt is None:
        return []

    _setup_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataframe = metrics_to_dataframe(history)
    if dataframe.empty:
        return []

    saved_paths: list[Path] = []
    epochs = dataframe["epoch"].tolist()
    loss_upper = _nice_upper_bound(float(max(dataframe["train_loss"].max(), dataframe["val_loss"].max())), minimum=1.0)
    cer_upper = _nice_upper_bound(float(dataframe["val_cer_percent"].max()), minimum=1.0)
    epoch_time_upper = _nice_upper_bound(float(dataframe["epoch_time_sec"].max()), minimum=1.0)

    loss_fig, loss_ax = plt.subplots(figsize=(10, 6))
    loss_ax.plot(epochs, dataframe["train_loss"], label="Train Loss", linewidth=2.5, color="#0f766e")
    loss_ax.plot(epochs, dataframe["val_loss"], label="Val Loss", linewidth=2.5, color="#b91c1c")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("CTC Loss")
    loss_ax.set_title("Training And Validation Loss")
    loss_ax.set_ylim(0, loss_upper)
    loss_ax.grid(True, alpha=0.25)
    loss_ax.legend()
    _annotate_axis(loss_ax, "Lower is better")
    loss_fig.tight_layout()
    loss_path = output_dir / "loss_curve.png"
    loss_fig.savefig(loss_path, dpi=220, bbox_inches="tight")
    plt.close(loss_fig)
    saved_paths.append(loss_path)

    cer_fig, cer_ax = plt.subplots(figsize=(10, 6))
    cer_ax.plot(epochs, dataframe["val_cer_percent"], label="Val CER (%)", linewidth=2.5, color="#1d4ed8")
    cer_ax.set_xlabel("Epoch")
    cer_ax.set_ylabel("CER (%)")
    cer_ax.set_title("Validation CER")
    cer_ax.set_ylim(0, cer_upper)
    cer_ax.grid(True, alpha=0.25)
    cer_ax.legend()
    _annotate_axis(cer_ax, "Lower is better")
    cer_fig.tight_layout()
    cer_path = output_dir / "cer_curve.png"
    cer_fig.savefig(cer_path, dpi=220, bbox_inches="tight")
    plt.close(cer_fig)
    saved_paths.append(cer_path)

    combo_fig, axes = plt.subplots(3, 1, figsize=(10, 13), sharex=True)
    axes[0].plot(epochs, dataframe["train_loss"], label="Train Loss", linewidth=2.2, color="#0f766e")
    axes[0].plot(epochs, dataframe["val_loss"], label="Val Loss", linewidth=2.2, color="#b91c1c")
    axes[0].set_ylabel("CTC Loss")
    axes[0].set_title("Epoch Metrics")
    axes[0].set_ylim(0, loss_upper)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    _annotate_axis(axes[0], "Lower is better")
    axes[1].plot(epochs, dataframe["val_cer_percent"], label="Val CER (%)", linewidth=2.2, color="#1d4ed8")
    axes[1].set_ylabel("CER (%)")
    axes[1].set_ylim(0, cer_upper)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    _annotate_axis(axes[1], "Lower is better")
    axes[2].plot(epochs, dataframe["epoch_time_sec"], label="Epoch Time (s)", linewidth=2.2, color="#7c3aed")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Time (s)")
    axes[2].set_ylim(0, epoch_time_upper)
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()
    _annotate_axis(axes[2], "Lower is better")
    combo_fig.tight_layout()
    combo_path = output_dir / "training_dashboard.png"
    combo_fig.savefig(combo_path, dpi=220, bbox_inches="tight")
    plt.close(combo_fig)
    saved_paths.append(combo_path)
    return saved_paths


def create_tensorboard_writer(log_dir: str | Path):
    if SummaryWriter is None:
        return None
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def log_epoch_to_tensorboard(writer, metrics: EpochMetrics) -> None:
    if writer is None:
        return
    writer.add_scalar("loss/train", metrics.train_loss, metrics.epoch)
    writer.add_scalar("loss/val", metrics.val_loss, metrics.epoch)
    writer.add_scalar("cer/val", metrics.val_cer, metrics.epoch)
    writer.add_scalar("cer_percent/val", metrics.val_cer * 100.0, metrics.epoch)
    writer.add_scalar("lr", metrics.learning_rate, metrics.epoch)
    writer.add_scalar("timing/epoch_time_sec", metrics.epoch_time_sec, metrics.epoch)
    writer.add_scalar("config/blank_logit_bias", metrics.blank_logit_bias, metrics.epoch)
    writer.add_scalar("config/grad_accum_steps", metrics.grad_accum_steps, metrics.epoch)
    writer.add_scalar("config/augment_enabled", int(metrics.augment_enabled), metrics.epoch)
    writer.add_scalar("config/amp_enabled", int(metrics.amp_enabled), metrics.epoch)
    writer.flush()


def log_run_config_to_tensorboard(writer, config: dict[str, Any]) -> None:
    if writer is None:
        return
    writer.add_text("run/config", json.dumps(config, indent=2, sort_keys=True), global_step=0)
    writer.flush()
