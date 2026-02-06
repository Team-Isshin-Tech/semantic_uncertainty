"""
Plot SE_after / SE_before vs SE_before to understand how SE changes 
relative to different initial SE values.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Load corrected data
input_file = Path('corrected_results/se_after_noise_corrected.jsonl')
plot_dir = Path('corrected_results/relative_change_plots')

print(f"Loading data from {input_file}...")
data = []
with open(input_file, 'r') as f:
    for line in f:
        entry = json.loads(line)
        data.append(entry)

df = pd.DataFrame(data)

# Calculate SE_after / SE_before ratio
df['se_ratio'] = df['se_mean'] / df['se_before']

print(f"✓ Loaded {len(df)} entries")
print(f"SE_ratio statistics:")
print(f"  Mean: {df['se_ratio'].mean():.4f}")
print(f"  Std:  {df['se_ratio'].std():.4f}")
print(f"  Min:  {df['se_ratio'].min():.4f}")
print(f"  Max:  {df['se_ratio'].max():.4f}")

# Set publication-quality style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

# ============================================================================
# Plot 1: Scatter plot with color by mu
# ============================================================================
print("\nCreating Plot 1: SE_after/SE_before vs SE_before (colored by μ)...")
fig, ax = plt.subplots(figsize=(14, 8))

mu_values = sorted(df['mu'].unique())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, mu in enumerate(mu_values):
    data_mu = df[df['mu'] == mu]
    ax.scatter(data_mu['se_before'], data_mu['se_ratio'], 
              alpha=0.4, s=30, label=f'μ = {mu:.1f}', 
              color=colors[i % len(colors)], edgecolors='none')

ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2.5, 
          alpha=0.7, label='No change (ratio=1.0)', zorder=5)
ax.set_xlabel('Initial SE (SE_before)', fontsize=14, fontweight='bold')
ax.set_ylabel('SE Ratio (SE_mean / SE_before)', fontsize=14, fontweight='bold')
ax.set_title('SE Ratio vs Initial SE - Colored by Mean Noise (μ)', 
            fontsize=15, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='best')
ax.set_xlim(left=0)

plt.tight_layout()
plot1_path = plot_dir / 'se_ratio_vs_se_before_by_mu.png'
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plot1_path}")
plt.close()

# ============================================================================
# Plot 2: Scatter plot with color by sigma
# ============================================================================
print("\nCreating Plot 2: SE_after/SE_before vs SE_before (colored by σ)...")
fig, ax = plt.subplots(figsize=(14, 8))

sigma_values = sorted(df['sigma'].unique())
colors_sigma = ['#9467bd', '#8c564b', '#e377c2']

for i, sigma in enumerate(sigma_values):
    data_sigma = df[df['sigma'] == sigma]
    ax.scatter(data_sigma['se_before'], data_sigma['se_ratio'],
              alpha=0.4, s=30, label=f'σ = {sigma:.1f}',
              color=colors_sigma[i % len(colors_sigma)], edgecolors='none')

ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2.5,
          alpha=0.7, label='No change (ratio=1.0)', zorder=5)
ax.set_xlabel('Initial SE (SE_before)', fontsize=14, fontweight='bold')
ax.set_ylabel('SE Ratio (SE_mean / SE_before)', fontsize=14, fontweight='bold')
ax.set_title('SE Ratio vs Initial SE - Colored by Std Noise (σ)',
            fontsize=15, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='best')
ax.set_xlim(left=0)

plt.tight_layout()
plot2_path = plot_dir / 'se_ratio_vs_se_before_by_sigma.png'
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plot2_path}")
plt.close()

# ============================================================================
# Plot 3: 2D Hexbin plot (shows density)
# ============================================================================
print("\nCreating Plot 3: Density plot (hexbin) SE_ratio vs SE_before...")
fig, ax = plt.subplots(figsize=(12, 8))

hexbin = ax.hexbin(df['se_before'], df['se_ratio'], gridsize=30, 
                   cmap='YlOrRd', mincnt=1, edgecolors='black', linewidths=0.2)

ax.axhline(y=1.0, color='blue', linestyle='--', linewidth=2.5,
          alpha=0.7, label='No change (ratio=1.0)', zorder=5)

ax.set_xlabel('Initial SE (SE_before)', fontsize=14, fontweight='bold')
ax.set_ylabel('SE Ratio (SE_mean / SE_before)', fontsize=14, fontweight='bold')
ax.set_title('Density: SE Ratio vs Initial SE (Hexbin)', fontsize=15, fontweight='bold', pad=15)
ax.set_xlim(left=0)

# Add colorbar
cbar = plt.colorbar(hexbin, ax=ax)
cbar.set_label('Count', fontsize=12, fontweight='bold')

plt.tight_layout()
plot3_path = plot_dir / 'se_ratio_vs_se_before_hexbin.png'
plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plot3_path}")
plt.close()

# ============================================================================
# Plot 4: Binned analysis - average ratio vs SE_before bins
# ============================================================================
print("\nCreating Plot 4: Average ratio by SE_before bins...")
fig, ax = plt.subplots(figsize=(14, 8))

# Create bins for SE_before
se_bins = np.linspace(df['se_before'].min(), df['se_before'].max(), 15)
df['se_bin'] = pd.cut(df['se_before'], bins=se_bins)
bin_stats = df.groupby('se_bin').agg({
    'se_ratio': ['mean', 'std', 'sem', 'count'],
    'se_before': 'mean'
}).reset_index()

bin_stats.columns = ['se_bin', 'ratio_mean', 'ratio_std', 'ratio_sem', 'count', 'se_before_mean']
bin_stats = bin_stats[bin_stats['count'] >= 5]  # Only bins with >= 5 samples

# Plot with error bars
ax.errorbar(bin_stats['se_before_mean'], bin_stats['ratio_mean'], 
           yerr=bin_stats['ratio_sem']*1.96,  # 95% CI
           marker='o', markersize=12, capsize=8, capthick=3, linewidth=2.5,
           color='darkblue', ecolor='darkblue', label='Mean ± 95% CI')

ax.fill_between(bin_stats['se_before_mean'],
               bin_stats['ratio_mean'] - bin_stats['ratio_sem']*1.96,
               bin_stats['ratio_mean'] + bin_stats['ratio_sem']*1.96,
               alpha=0.2, color='darkblue')

ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2.5,
          alpha=0.7, label='No change (ratio=1.0)', zorder=5)

ax.set_xlabel('Initial SE (SE_before)', fontsize=14, fontweight='bold')
ax.set_ylabel('Average SE Ratio (SE_mean / SE_before)', fontsize=14, fontweight='bold')
ax.set_title('Average SE Ratio by Initial SE Level (with 95% CI)', 
            fontsize=15, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='best')
ax.set_xlim(left=0)

plt.tight_layout()
plot4_path = plot_dir / 'se_ratio_vs_se_before_binned.png'
plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plot4_path}")
plt.close()

# ============================================================================
# Analysis: Correlation and trends
# ============================================================================
print("\n" + "="*80)
print("STATISTICAL ANALYSIS")
print("="*80)

correlation = df['se_before'].corr(df['se_ratio'])
print(f"\nCorrelation between SE_before and SE_ratio: {correlation:.4f}")

# Divide into high and low SE groups
median_se = df['se_before'].median()
low_se = df[df['se_before'] < median_se]
high_se = df[df['se_before'] >= median_se]

print(f"\nLow SE group (SE < {median_se:.4f}):")
print(f"  Mean ratio: {low_se['se_ratio'].mean():.4f}")
print(f"  Std ratio:  {low_se['se_ratio'].std():.4f}")
print(f"  Count:      {len(low_se)}")

print(f"\nHigh SE group (SE >= {median_se:.4f}):")
print(f"  Mean ratio: {high_se['se_ratio'].mean():.4f}")
print(f"  Std ratio:  {high_se['se_ratio'].std():.4f}")
print(f"  Count:      {len(high_se)}")

# By mu
print(f"\n" + "-"*80)
print("By Mean Noise (μ):")
print("-"*80)
for mu in sorted(df['mu'].unique()):
    data_mu = df[df['mu'] == mu]
    print(f"\nμ = {mu:.1f}:")
    print(f"  Mean ratio: {data_mu['se_ratio'].mean():.4f}")
    print(f"  Std ratio:  {data_mu['se_ratio'].std():.4f}")

# By sigma
print(f"\n" + "-"*80)
print("By Std Noise (σ):")
print("-"*80)
for sigma in sorted(df['sigma'].unique()):
    data_sigma = df[df['sigma'] == sigma]
    print(f"\nσ = {sigma:.1f}:")
    print(f"  Mean ratio: {data_sigma['se_ratio'].mean():.4f}")
    print(f"  Std ratio:  {data_sigma['se_ratio'].std():.4f}")

print(f"\n" + "="*80)
print(f"✓ All plots saved to: {plot_dir}")
print(f"="*80)
