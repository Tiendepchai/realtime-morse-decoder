import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd

def setup_style():
    """Apply an academic, highly legible seaborn style."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']
    })


def _nice_upper_bound(value, minimum=1.0):
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


def _annotate_axis(ax, text):
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=10,
        color='#444444',
        bbox={'boxstyle': 'round,pad=0.25', 'facecolor': '#ffffff', 'edgecolor': '#dddddd', 'alpha': 0.95},
    )

def plot_bar_comparison(metrics_list, output_dir):
    """Plot CER and WER side-by-side, and RTF in a separate plot."""
    setup_style()
    models = [m['model'].upper() for m in metrics_list]
    cers = [m['cer_percent'] for m in metrics_list]
    wers = [m['wer_percent'] for m in metrics_list]
    rtfs = [m['rtf_cpu'] for m in metrics_list]

    x = np.arange(len(models))
    width = 0.3

    # Plot 1: CER & WER
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, cers, width, label='CER (%)', color='#2F4F4F', edgecolor='black')
    ax.bar(x + width/2, wers, width, label='WER (%)', color='#4682B4', edgecolor='black')
    
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('CER and WER Comparison')
    ax.set_ylim(0, _nice_upper_bound(max(cers + wers), minimum=1.0))
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    _annotate_axis(ax, 'Lower is better')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "cer_wer_comparison.png"), dpi=300)
    plt.close()

    # Plot 2: RTF
    fig, ax = plt.subplots(figsize=(6, 5))
    rtf_max = max(rtfs) if rtfs else 0.0
    if rtf_max < 0.1:
        scale = 1000.0
        scaled_rtfs = [v * scale for v in rtfs]
        ylabel = 'RTF (x1e-3)'
        note = 'Lower is better; 1.0 real-time is off-scale'
    else:
        scale = 1.0
        scaled_rtfs = rtfs
        ylabel = 'RTF'
        note = 'Lower is better; below 1.0 means faster than real-time'
    ax.bar(x, scaled_rtfs, 0.4, label='RTF', color='#808080', edgecolor='black')
    ax.set_ylabel(ylabel)
    ax.set_title('RTF Comparison')
    ax.set_ylim(0, _nice_upper_bound(max(scaled_rtfs) if scaled_rtfs else 0.0, minimum=0.5 if scale != 1.0 else 0.05))
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    if scale == 1.0 and rtf_max <= 1.2:
        ax.axhline(1.0, color='#555555', linestyle='--', linewidth=1.5)
    _annotate_axis(ax, note)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "rtf_comparison.png"), dpi=300)
    plt.close()

def plot_latency_distribution(metrics, output_dir):
    """Plot latency histogram overlaid with P50 and P90."""
    setup_style()
    latencies = metrics['raw_latencies']
    model_name = metrics['model'].upper()
    p50 = metrics['latency_p50_ms']
    p90 = metrics['latency_p90_ms']

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(latencies, bins=30, kde=True, color='#4682B4', ax=ax, edgecolor='black', alpha=0.6)
    
    ax.axvline(p50, color='red', linestyle='--', linewidth=2, label=f'P50: {p50:.1f} ms')
    ax.axvline(p90, color='green', linestyle='-.', linewidth=2, label=f'P90: {p90:.1f} ms')
    
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Latency Distribution - {model_name}')
    ax.legend()
    _annotate_axis(ax, 'Lower is better')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"latency_dist_{metrics['model']}.png"), dpi=300)
    plt.close()

def plot_latency_boxplot(metrics_list, output_dir):
    """Plot side-by-side boxplots for latency over the test set."""
    setup_style()
    data = []
    labels = []
    for m in metrics_list:
        data.extend(m['raw_latencies'])
        labels.extend([m['model'].upper()] * len(m['raw_latencies']))
        
    df = pd.DataFrame({'Latency (ms)': data, 'Model': labels})
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='Model', y='Latency (ms)', data=df, ax=ax, width=0.5, palette='pastel')
    ax.set_title('Latency Comparison (C-RNN vs Conformer)')
    ax.set_ylabel('Latency (ms)')
    _annotate_axis(ax, 'Lower is better')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_boxplot.png"), dpi=300)
    plt.close()
