"""
Generate clear, publication-quality plots for relative SE change.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Configuration
input_file = Path('corrected_results/se_after_noise_corrected.jsonl')
plot_dir = Path('corrected_results/relative_change_plots')

print(f"Loading data from {input_file}...")
data = []
with open(input_file, 'r') as f:
    for line in f:
        entry = json.loads(line)
        data.append(entry)

df = pd.DataFrame(data)
df['relative_change'] = df['delta_se'] / df['se_before']

print(f"✓ Loaded {len(df)} entries")
print(f"Unique mu values: {sorted(df['mu'].unique())}")
print(f"Unique sigma values: {sorted(df['sigma'].unique())}")

# Set publication-quality style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Color palettes
mu_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
sigma_colors = ['#9467bd', '#8c564b', '#e377c2']

# ============================================================================
# Plot 1: Box plots with individual points - Mean Noise (mu)
# ============================================================================
print("\nCreating Plot 1: Box plot vs Mean Noise (mu)...")
fig, ax = plt.subplots(figsize=(12, 7))

mu_values = sorted(df['mu'].unique())
positions = np.arange(len(mu_values))

# Create box plots
box_data = [df[df['mu'] == mu]['relative_change'].values for mu in mu_values]
bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                showfliers=False, notch=True,
                boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=1.5),
                medianprops=dict(color='red', linewidth=2.5),
                whiskerprops=dict(color='black', linewidth=1.5),
                capprops=dict(color='black', linewidth=1.5))

# Overlay scatter points without x-axis jitter (only vertical spread)
for i, mu in enumerate(mu_values):
    y_data = df[df['mu'] == mu]['relative_change'].values
    # Vertical jitter only to show overlapping points
    y_jitter = y_data + np.random.normal(0, 0.02, size=len(y_data))
    x_positions = np.full(len(y_jitter), i)
    ax.scatter(x_positions, y_jitter, alpha=0.3, s=20, color=mu_colors[i % len(mu_colors)],
               edgecolors='black', linewidths=0.3, zorder=3)

# Add mean markers
for i, mu in enumerate(mu_values):
    mean_val = df[df['mu'] == mu]['relative_change'].mean()
    ax.scatter(i, mean_val, marker='D', s=150, color='gold', 
               edgecolors='black', linewidths=2, zorder=5, label='Mean' if i == 0 else '')

ax.set_xticks(positions)
ax.set_xticklabels([f'{mu:.1f}' for mu in mu_values])
ax.set_xlim(-0.5, len(mu_values) - 0.5)
# Remove any minor ticks
ax.minorticks_off()
# Disable automatic tick generation
ax.tick_params(which='both', top=False, bottom=True)
ax.set_xlabel('Mean Noise (μ)', fontsize=14, fontweight='bold')
ax.set_ylabel('Relative SE Change (ΔSE / SE_before)', fontsize=14, fontweight='bold')
ax.set_title('Semantic Entropy Relative Change vs Mean Noise', fontsize=16, fontweight='bold', pad=20)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='No Change')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(loc='best', fontsize=11)

plt.tight_layout()
plot1_path = plot_dir / 'relative_change_vs_mu_clear.png'
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plot1_path}")
plt.close()

# ============================================================================
# Plot 2: Box plots with individual points - Std Noise (sigma)
# ============================================================================
print("\nCreating Plot 2: Box plot vs Std Noise (sigma)...")
fig, ax = plt.subplots(figsize=(12, 7))

sigma_values = sorted(df['sigma'].unique())
positions = np.arange(len(sigma_values))

# Create box plots
box_data = [df[df['sigma'] == sig]['relative_change'].values for sig in sigma_values]
bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                showfliers=False, notch=True,
                boxprops=dict(facecolor='lightgreen', edgecolor='black', linewidth=1.5),
                medianprops=dict(color='darkred', linewidth=2.5),
                whiskerprops=dict(color='black', linewidth=1.5),
                capprops=dict(color='black', linewidth=1.5))

# Overlay scatter points without x-axis jitter (only vertical spread)
for i, sig in enumerate(sigma_values):
    y_data = df[df['sigma'] == sig]['relative_change'].values
    # Vertical jitter only to show overlapping points
    y_jitter = y_data + np.random.normal(0, 0.02, size=len(y_data))
    x_positions = np.full(len(y_jitter), i)
    ax.scatter(x_positions, y_jitter, alpha=0.3, s=20, color=sigma_colors[i % len(sigma_colors)],
               edgecolors='black', linewidths=0.3, zorder=3)

# Add mean markers
for i, sig in enumerate(sigma_values):
    mean_val = df[df['sigma'] == sig]['relative_change'].mean()
    ax.scatter(i, mean_val, marker='D', s=150, color='gold', 
               edgecolors='black', linewidths=2, zorder=5, label='Mean' if i == 0 else '')

ax.set_xticks(positions)
ax.set_xticklabels([f'{sig:.1f}' for sig in sigma_values])
ax.set_xlim(-0.5, len(sigma_values) - 0.5)
# Remove any minor ticks
ax.minorticks_off()
# Disable automatic tick generation
ax.tick_params(which='both', top=False, bottom=True)
ax.set_xlabel('Std Noise (σ)', fontsize=14, fontweight='bold')
ax.set_ylabel('Relative SE Change (ΔSE / SE_before)', fontsize=14, fontweight='bold')
ax.set_title('Semantic Entropy Relative Change vs Std Noise', fontsize=16, fontweight='bold', pad=20)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='No Change')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(loc='best', fontsize=11)

plt.tight_layout()
plot2_path = plot_dir / 'relative_change_vs_sigma_clear.png'
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plot2_path}")
plt.close()

# ============================================================================
# Plot 3: Line plots with error bars (cleaner version)
# ============================================================================
print("\nCreating Plot 3: Line plots with confidence intervals...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left: vs mu
mu_stats = df.groupby('mu')['relative_change'].agg(['mean', 'std', 'sem']).reset_index()
mu_values_plot = sorted(df['mu'].unique())
ax1.errorbar(mu_stats['mu'], mu_stats['mean'], yerr=mu_stats['sem']*1.96,  # 95% CI
            marker='o', markersize=12, capsize=8, capthick=3, linewidth=3,
            color='#1f77b4', ecolor='#1f77b4', label='Mean ± 95% CI')
ax1.fill_between(mu_stats['mu'], 
                 mu_stats['mean'] - mu_stats['sem']*1.96,
                 mu_stats['mean'] + mu_stats['sem']*1.96,
                 alpha=0.2, color='#1f77b4')
ax1.set_xticks(mu_values_plot)
ax1.set_xticklabels([f'{mu:.1f}' for mu in mu_values_plot])
ax1.set_xlim(min(mu_values_plot) - 0.3, max(mu_values_plot) + 0.3)
ax1.minorticks_off()
ax1.set_xlabel('Mean Noise (μ)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Relative SE Change (ΔSE / SE_before)', fontsize=14, fontweight='bold')
ax1.set_title('(a) Relative Change vs Mean Noise', fontsize=14, fontweight='bold')
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# Right: vs sigma
sigma_stats = df.groupby('sigma')['relative_change'].agg(['mean', 'std', 'sem']).reset_index()
sigma_values_plot = sorted(df['sigma'].unique())
ax2.errorbar(sigma_stats['sigma'], sigma_stats['mean'], yerr=sigma_stats['sem']*1.96,
            marker='s', markersize=12, capsize=8, capthick=3, linewidth=3,
            color='darkgreen', ecolor='darkgreen', label='Mean ± 95% CI')
ax2.fill_between(sigma_stats['sigma'],
                 sigma_stats['mean'] - sigma_stats['sem']*1.96,
                 sigma_stats['mean'] + sigma_stats['sem']*1.96,
                 alpha=0.2, color='darkgreen')
ax2.set_xticks(sigma_values_plot)
ax2.set_xticklabels([f'{sig:.1f}' for sig in sigma_values_plot])
ax2.set_xlim(min(sigma_values_plot) - 0.3, max(sigma_values_plot) + 0.3)
ax2.minorticks_off()
ax2.set_xlabel('Std Noise (σ)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Relative SE Change (ΔSE / SE_before)', fontsize=14, fontweight='bold')
ax2.set_title('(b) Relative Change vs Std Noise', fontsize=14, fontweight='bold')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

plt.suptitle('Semantic Entropy Relative Change Under Noise Perturbations', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plot3_path = plot_dir / 'relative_change_line_plots.png'
plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plot3_path}")
plt.close()

# ============================================================================
# Plot 4: Violin plots (show full distribution)
# ============================================================================
print("\nCreating Plot 4: Violin plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left: vs mu
df_mu = df.copy()
df_mu['mu_str'] = df_mu['mu'].astype(str)
mu_order = [str(x) for x in sorted(df['mu'].unique())]
sns.violinplot(data=df_mu, x='mu_str', y='relative_change', ax=ax1, 
               palette='Set2', inner='quartile', linewidth=1.5, order=mu_order)
ax1.set_xlabel('Mean Noise (μ)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Relative SE Change (ΔSE / SE_before)', fontsize=14, fontweight='bold')
ax1.set_title('(a) Distribution vs Mean Noise', fontsize=14, fontweight='bold')
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax1.grid(True, alpha=0.3, axis='y')

# Right: vs sigma
df_sigma = df.copy()
df_sigma['sigma_str'] = df_sigma['sigma'].astype(str)
sigma_order = [str(x) for x in sorted(df['sigma'].unique())]
sns.violinplot(data=df_sigma, x='sigma_str', y='relative_change', ax=ax2,
               palette='Set3', inner='quartile', linewidth=1.5, order=sigma_order)
ax2.set_xlabel('Std Noise (σ)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Relative SE Change (ΔSE / SE_before)', fontsize=14, fontweight='bold')
ax2.set_title('(b) Distribution vs Std Noise', fontsize=14, fontweight='bold')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle('Semantic Entropy Relative Change Distribution', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plot4_path = plot_dir / 'relative_change_violin_plots.png'
plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plot4_path}")
plt.close()

print(f"\n{'='*80}")
print(f"COMPLETED - Generated 4 clear plot types:")
print(f"{'='*80}")
print(f"1. relative_change_vs_mu_clear.png       - Box plot with points (μ)")
print(f"2. relative_change_vs_sigma_clear.png    - Box plot with points (σ)")
print(f"3. relative_change_line_plots.png        - Line plots with confidence intervals")
print(f"4. relative_change_violin_plots.png      - Violin plots showing distributions")
print(f"\n✓ All plots saved to: {plot_dir}")
