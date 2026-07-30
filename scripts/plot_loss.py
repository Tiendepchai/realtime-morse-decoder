
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def setup_style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'axes.facecolor': '#fafafa',
        'figure.facecolor': 'white',
    })


def annotate_axis(ax, text):
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=10, color='#444444',
        bbox={'boxstyle': 'round,pad=0.25', 'facecolor': '#ffffff', 'edgecolor': '#dddddd', 'alpha': 0.95},
    )


def nice_upper_bound(value, minimum=1.0):
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

def parse_logs(log_file):
    epochs = []
    train_losses = []
    val_losses = []
    val_cers = []

    epoch_pattern = re.compile(r"Epoch (\d+): Train Loss=([\d.]+), Val Loss=([\d.]+), Val CER=([\d.]+)")

    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return None

    with open(log_file, "r") as f:
        for line in f:
            match = epoch_pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                val_losses.append(float(match.group(3)))
                val_cers.append(float(match.group(4)))

    return epochs, train_losses, val_losses, val_cers

def plot_fancy_metrics(epochs, train_losses, val_losses, val_cers, output_dir):
    setup_style()
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Loss Curve - Premium Aesthetic
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Draw curves with shadows/glow effect using multiple lines
    ax.plot(epochs, train_losses, color='#3b82f6', label='Training Loss', 
            linewidth=3, alpha=0.9, marker='o', markersize=6, markeredgewidth=1.5, markeredgecolor='white')
    ax.plot(epochs, val_losses, color='#ef4444', label='Validation Loss', 
            linewidth=3, alpha=0.9, marker='s', markersize=6, markeredgewidth=1.5, markeredgecolor='white')

    # Fill area under curves for visual depth
    ax.fill_between(epochs, train_losses, color='#3b82f6', alpha=0.05)
    ax.fill_between(epochs, val_losses, color='#ef4444', alpha=0.05)

    # Title & Labels
    ax.set_title('Neural Network Convergence Profile', fontsize=22, fontweight='bold', pad=25, color='#1f2937')
    ax.set_xlabel('Epoch', fontsize=14, fontweight='medium', color='#4b5563')
    ax.set_ylabel('CTC Loss', fontsize=14, fontweight='medium', color='#4b5563')
    ax.set_ylim(0, nice_upper_bound(max(max(train_losses), max(val_losses)), minimum=1.0))
    
    # Customizing Grid & Spine
    ax.grid(True, linestyle='--', alpha=0.4, which='both')
    sns.despine(left=True, bottom=True)
    
    # Legend
    legend = ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=12)
    legend.get_frame().set_linewidth(0)
    annotate_axis(ax, 'Lower is better')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve_premium.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. CER Curve - Gradient Effect
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Gradient line effect
    color = '#10b981' # Emerald 500
    ax.plot(epochs, val_cers, color=color, label='Recognition Error (CER)', 
            linewidth=4, marker='D', markersize=7, markeredgewidth=2, markeredgecolor='white')
    
    # Annotation for best epoch
    best_idx = np.argmin(val_cers)
    ax.annotate(f'Best: {val_cers[best_idx]:.3f}', 
                xy=(epochs[best_idx], val_cers[best_idx]), 
                xytext=(epochs[best_idx]+1, val_cers[best_idx]+0.1),
                arrowprops=dict(facecolor='#111827', shrink=0.05, width=1, headwidth=6),
                fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#f3f4f6", ec="#d1d5db", lw=1))

    ax.set_title('Character Error Rate (CER) Trend', fontsize=22, fontweight='bold', pad=25, color='#1f2937')
    ax.set_xlabel('Epoch', fontsize=14, fontweight='medium', color='#4b5563')
    ax.set_ylabel('CER (%)', fontsize=14, fontweight='medium', color='#4b5563')
    
    ax.set_ylim(0, nice_upper_bound(max(val_cers), minimum=1.0))
    ax.grid(True, linestyle='--', alpha=0.4)
    sns.despine(left=True, bottom=True)
    annotate_axis(ax, 'Lower is better')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cer_curve_premium.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Premium plots saved to {output_dir}")

if __name__ == "__main__":
    log_file = "nohup.out"
    output_dir = "reports/figures"
    
    data = parse_logs(log_file)
    if data and data[0]:
        epochs, train_losses, val_losses, val_cers = data
        plot_fancy_metrics(epochs, train_losses, val_losses, val_cers, output_dir)
    else:
        print("No training data found in logs.")
