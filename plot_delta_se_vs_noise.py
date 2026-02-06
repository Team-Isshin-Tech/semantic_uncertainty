"""
Plot delta SE vs mean noise (mu) and std noise (sigma).
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

input_file = Path('corrected_results/se_after_noise_corrected.jsonl')
plot_dir = Path('corrected_results/relative_change_plots')
plot_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading data from {input_file}...")
data = []
with open(input_file, 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

print(f"Loaded {len(df)} rows")

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

# Plot 1: delta_se vs mu
print("Creating plot: delta_se vs mu...")
fig, ax = plt.subplots(figsize=(12, 7))
mu_values = sorted(df['mu'].unique())
positions = np.arange(len(mu_values))

box_data = [df[df['mu'] == mu]['delta_se'].values for mu in mu_values]

# Scatter points without x jitter
for i, mu in enumerate(mu_values):
    y_data = df[df['mu'] == mu]['delta_se'].values
    y_jitter = y_data + np.random.normal(0, 0.005, size=len(y_data))
    x_positions = np.full(len(y_jitter), i)
    ax.scatter(x_positions, y_jitter, alpha=0.25, s=18, color='#1f77b4',
               edgecolors='black', linewidths=0.2, zorder=3)

# Mean and std per discrete mu
mu_stats = df.groupby('mu')['delta_se'].agg(['mean', 'std']).reset_index()
ax.errorbar(
    np.arange(len(mu_values)),
    mu_stats['mean'],
    yerr=mu_stats['std'],
    fmt='D',
    color='black',
    ecolor='black',
    elinewidth=2,
    capsize=6,
    markersize=8,
    label='Mean ± Std'
)

ax.set_xticks(positions)
ax.set_xticklabels([f'{mu:.1f}' for mu in mu_values])
ax.set_xlim(-0.5, len(mu_values) - 0.5)
ax.minorticks_off()
ax.set_xlabel('Mean Noise (mu)', fontsize=14, fontweight='bold')
ax.set_ylabel('Delta SE', fontsize=14, fontweight='bold')
ax.set_title('Delta SE vs Mean Noise (mu)', fontsize=15, fontweight='bold', pad=15)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.grid(True, alpha=0.3, axis='y')
ax.legend(loc='best', fontsize=11)

plt.tight_layout()
plot1_path = plot_dir / 'delta_se_vs_mu.png'
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {plot1_path}")

# Plot 2: delta_se vs sigma
print("Creating plot: delta_se vs sigma...")
fig, ax = plt.subplots(figsize=(12, 7))
sigma_values = sorted(df['sigma'].unique())
positions = np.arange(len(sigma_values))

box_data = [df[df['sigma'] == sig]['delta_se'].values for sig in sigma_values]

# Scatter points without x jitter
for i, sig in enumerate(sigma_values):
    y_data = df[df['sigma'] == sig]['delta_se'].values
    y_jitter = y_data + np.random.normal(0, 0.005, size=len(y_data))
    x_positions = np.full(len(y_jitter), i)
    ax.scatter(x_positions, y_jitter, alpha=0.25, s=18, color='#2ca02c',
               edgecolors='black', linewidths=0.2, zorder=3)

# Mean and std per discrete sigma
sigma_stats = df.groupby('sigma')['delta_se'].agg(['mean', 'std']).reset_index()
ax.errorbar(
    np.arange(len(sigma_values)),
    sigma_stats['mean'],
    yerr=sigma_stats['std'],
    fmt='D',
    color='black',
    ecolor='black',
    elinewidth=2,
    capsize=6,
    markersize=8,
    label='Mean ± Std'
)

ax.set_xticks(positions)
ax.set_xticklabels([f'{sig:.1f}' for sig in sigma_values])
ax.set_xlim(-0.5, len(sigma_values) - 0.5)
ax.minorticks_off()
ax.set_xlabel('Std Noise (sigma)', fontsize=14, fontweight='bold')
ax.set_ylabel('Delta SE', fontsize=14, fontweight='bold')
ax.set_title('Delta SE vs Std Noise (sigma)', fontsize=15, fontweight='bold', pad=15)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.grid(True, alpha=0.3, axis='y')
ax.legend(loc='best', fontsize=11)

plt.tight_layout()
plot2_path = plot_dir / 'delta_se_vs_sigma.png'
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {plot2_path}")

print("Done.")
