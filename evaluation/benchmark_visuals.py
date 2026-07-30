from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _setup_style() -> None:
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

    candidate = value * 1.15
    magnitude = 10 ** np.floor(np.log10(candidate))
    residual = candidate / magnitude
    for nice in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if residual <= nice:
            break
    return float(max(minimum, nice * magnitude))


def _nice_lower_bound(value: float, step: float = 5.0) -> float:
    return float(max(0.0, step * np.floor(value / step)))


def _annotate_axis(ax: plt.Axes, summary: str) -> None:
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


def _format_value(value: float) -> str:
    value = float(value)
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _add_bar_labels(ax: plt.Axes, containers, values: np.ndarray, y_limit: float) -> None:
    y_low, y_high = ax.get_ylim()
    span = y_high - y_low
    offset = 0.015 * span
    for container, value in zip(containers, values):
        y_position = container.get_height() + offset
        va = "bottom"
        if y_position > y_high - 0.03 * span:
            y_position = max(y_low + offset, container.get_height() - offset)
            va = "top"
        ax.text(
            container.get_x() + container.get_width() / 2,
            y_position,
            _format_value(value),
            ha="center",
            va=va,
            fontsize=9,
            color="#333333",
            rotation=0,
        )


def plot_benchmark_dashboard(summary_df: pd.DataFrame, output_dir: str | os.PathLike[str]) -> None:
    if summary_df.empty:
        return

    _setup_style()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_df = summary_df.sort_values("cer_percent_mean", ascending=True).reset_index(drop=True)
    methods = summary_df["method"].tolist()
    x = np.arange(len(methods))
    width = 0.24

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    error_upper = float(
        max(
            (summary_df["cer_percent_mean"] + summary_df["cer_percent_std"]).max(),
            (summary_df["wer_percent_mean"] + summary_df["wer_percent_std"]).max(),
            (summary_df["decode_failure_percent_mean"] + summary_df["decode_failure_percent_std"]).max(),
        )
    )
    cer_container = axes[0, 0].bar(
        x - width,
        summary_df["cer_percent_mean"],
        width=width,
        yerr=summary_df["cer_percent_std"],
        color="#1f77b4",
        capsize=4,
        label="CER (%)",
    )
    wer_container = axes[0, 0].bar(
        x,
        summary_df["wer_percent_mean"],
        width=width,
        yerr=summary_df["wer_percent_std"],
        color="#ff7f0e",
        capsize=4,
        label="WER (%)",
    )
    failure_container = axes[0, 0].bar(
        x + width,
        summary_df["decode_failure_percent_mean"],
        width=width,
        yerr=summary_df["decode_failure_percent_std"],
        color="#d62728",
        capsize=4,
        label="Failure (%)",
    )
    axes[0, 0].set_title("Error Rates Across Repeated Runs")
    axes[0, 0].set_ylabel("Error Rate (%)")
    error_ylim = _nice_upper_bound(error_upper, minimum=1.0)
    axes[0, 0].set_ylim(0, error_ylim)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(methods, rotation=20)
    axes[0, 0].legend()
    _annotate_axis(axes[0, 0], "Lower is better")
    _add_bar_labels(axes[0, 0], cer_container, summary_df["cer_percent_mean"].to_numpy(), error_ylim)
    _add_bar_labels(axes[0, 0], wer_container, summary_df["wer_percent_mean"].to_numpy(), error_ylim)
    _add_bar_labels(axes[0, 0], failure_container, summary_df["decode_failure_percent_mean"].to_numpy(), error_ylim)

    exact_lower = float((summary_df["exact_match_percent_mean"] - summary_df["exact_match_percent_std"]).min())
    exact_container = axes[0, 1].bar(
        x,
        summary_df["exact_match_percent_mean"],
        yerr=summary_df["exact_match_percent_std"],
        color="#2ca02c",
        capsize=4,
    )
    axes[0, 1].set_title("Sequence Exact Match")
    axes[0, 1].set_ylabel("Exact Match (%)")
    exact_bottom = _nice_lower_bound(exact_lower - 5.0)
    axes[0, 1].set_ylim(exact_bottom, 100.0)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(methods, rotation=20)
    _annotate_axis(axes[0, 1], "Higher is better")
    _add_bar_labels(axes[0, 1], exact_container, summary_df["exact_match_percent_mean"].to_numpy(), 100.0 - exact_bottom)

    latency_upper = float(
        max(
            (summary_df["latency_mean_ms_mean"] + summary_df["latency_mean_ms_std"]).max(),
            (summary_df["latency_p90_ms_mean"] + summary_df["latency_p90_ms_std"]).max(),
        )
    )
    latency_mean_container = axes[1, 0].bar(
        x - width / 2,
        summary_df["latency_mean_ms_mean"],
        width=width,
        yerr=summary_df["latency_mean_ms_std"],
        color="#9467bd",
        capsize=4,
        label="Mean Latency (ms)",
    )
    latency_p90_container = axes[1, 0].bar(
        x + width / 2,
        summary_df["latency_p90_ms_mean"],
        width=width,
        yerr=summary_df["latency_p90_ms_std"],
        color="#8c564b",
        capsize=4,
        label="P90 Latency (ms)",
    )
    axes[1, 0].set_title("Latency Distribution Summary")
    axes[1, 0].set_ylabel("Latency (ms)")
    latency_ylim = _nice_upper_bound(latency_upper, minimum=1.0)
    axes[1, 0].set_ylim(0, latency_ylim)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(methods, rotation=20)
    axes[1, 0].legend()
    _annotate_axis(axes[1, 0], "Lower is better")
    _add_bar_labels(axes[1, 0], latency_mean_container, summary_df["latency_mean_ms_mean"].to_numpy(), latency_ylim)
    _add_bar_labels(axes[1, 0], latency_p90_container, summary_df["latency_p90_ms_mean"].to_numpy(), latency_ylim)

    rtf_upper = float((summary_df["rtf_cpu_mean"] + summary_df["rtf_cpu_std"]).max())
    if rtf_upper < 0.1:
        rtf_scale = 1000.0
        rtf_label = "RTF (x1e-3)"
        rtf_note = "Lower is better; 1.0 real-time is off-scale"
    else:
        rtf_scale = 1.0
        rtf_label = "RTF"
        rtf_note = "Lower is better; below 1.0 means faster than real-time"
    rtf_container = axes[1, 1].bar(
        x,
        summary_df["rtf_cpu_mean"] * rtf_scale,
        yerr=summary_df["rtf_cpu_std"] * rtf_scale,
        color="#17becf",
        capsize=4,
    )
    axes[1, 1].set_title("Real-Time Factor")
    axes[1, 1].set_ylabel(rtf_label)
    rtf_ylim = _nice_upper_bound(rtf_upper * rtf_scale, minimum=0.5 if rtf_scale != 1.0 else 0.05)
    axes[1, 1].set_ylim(0, rtf_ylim)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(methods, rotation=20)
    if rtf_scale == 1.0 and rtf_upper <= 1.2:
        axes[1, 1].axhline(1.0, color="#555555", linestyle="--", linewidth=1.5)
    _annotate_axis(axes[1, 1], rtf_note)
    _add_bar_labels(axes[1, 1], rtf_container, (summary_df["rtf_cpu_mean"] * rtf_scale).to_numpy(), rtf_ylim)

    fig.suptitle("Morse Benchmark Suite: Mean ± Std Over Repeated Runs", fontsize=18, y=0.98)
    fig.tight_layout()
    fig.savefig(output_path / "benchmark_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_run_scatter(run_df: pd.DataFrame, output_dir: str | os.PathLike[str]) -> None:
    if run_df.empty:
        return

    _setup_style()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    methods = sorted(run_df["method"].unique().tolist())
    method_to_x = {method: index for index, method in enumerate(methods)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for seed, group in run_df.groupby("seed"):
        x = np.array([method_to_x[method] for method in group["method"]], dtype=np.float64)
        jitter = np.linspace(-0.08, 0.08, max(1, group.shape[0]))
        axes[0].scatter(x + jitter[: group.shape[0]], group["cer_percent"], s=70, alpha=0.85, label=f"seed={seed}")
        axes[1].scatter(x + jitter[: group.shape[0]], group["latency_mean_ms"], s=70, alpha=0.85, label=f"seed={seed}")

    axes[0].set_title("CER by Run")
    axes[0].set_ylabel("CER (%)")
    cer_upper = float((run_df["cer_percent"]).max())
    axes[0].set_ylim(0, _nice_upper_bound(cer_upper, minimum=1.0))
    _annotate_axis(axes[0], "Lower is better")
    axes[1].set_title("Mean Latency by Run")
    axes[1].set_ylabel("Latency (ms)")
    latency_upper = float(run_df["latency_mean_ms"].max())
    axes[1].set_ylim(0, _nice_upper_bound(latency_upper, minimum=1.0))
    _annotate_axis(axes[1], "Lower is better")
    for axis in axes:
        axis.set_xticks(range(len(methods)))
        axis.set_xticklabels(methods, rotation=20)
        axis.legend()

    fig.tight_layout()
    fig.savefig(output_path / "benchmark_runs.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
