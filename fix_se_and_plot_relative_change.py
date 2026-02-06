"""
Fix SE values below threshold and plot relative SE change vs noise parameters.
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Configuration
THRESHOLD = 1e-16
input_file = Path('results/se_after_noise_complete.jsonl')
output_dir = Path('corrected_results')
output_dir.mkdir(exist_ok=True)

output_file = output_dir / 'se_after_noise_corrected.jsonl'
plot_dir = output_dir / 'relative_change_plots'
plot_dir.mkdir(exist_ok=True)

print(f"Loading data from {input_file}...")
data = []
with open(input_file, 'r') as f:
    for line in f:
        entry = json.loads(line)
        data.append(entry)

print(f"✓ Loaded {len(data)} entries")

# Fix SE values below threshold
print(f"\nFixing SE values below {THRESHOLD}...")
fixed_count_before = 0
fixed_count_mean = 0

for entry in data:
    if entry['se_before'] < THRESHOLD:
        entry['se_before'] = THRESHOLD
        fixed_count_before += 1
    
    if entry['se_mean'] < THRESHOLD:
        entry['se_mean'] = THRESHOLD
        fixed_count_mean += 1

print(f"✓ Fixed {fixed_count_before} se_before values")
print(f"✓ Fixed {fixed_count_mean} se_mean values")

# Save corrected data
print(f"\nSaving corrected data to {output_file}...")
with open(output_file, 'w') as f:
    for entry in data:
        f.write(json.dumps(entry) + '\n')

print(f"✓ Saved corrected JSONL file")

# Convert to DataFrame for plotting
df = pd.DataFrame(data)

# Calculate relative change: delta_se / se_before
df['relative_change'] = df['delta_se'] / df['se_before']

print(f"\nDataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Statistics
print(f"\nRelative Change Statistics:")
print(f"  Mean: {df['relative_change'].mean():.6f}")
print(f"  Std:  {df['relative_change'].std():.6f}")
print(f"  Min:  {df['relative_change'].min():.6f}")
print(f"  Max:  {df['relative_change'].max():.6f}")

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Plot 1: Relative Change vs Mean Noise (mu)
print(f"\nCreating Plot 1: Relative Change vs Mean Noise (mu)...")
fig, ax = plt.subplots(figsize=(10, 6))

# Group by mu and calculate statistics
mu_stats = df.groupby('mu')['relative_change'].agg(['mean', 'std', 'count']).reset_index()
mu_values = sorted(df['mu'].unique())

# Plot with error bars
ax.errorbar(mu_stats['mu'], mu_stats['mean'], yerr=mu_stats['std'], 
            marker='o', markersize=8, capsize=5, capthick=2, linewidth=2,
            label='Mean ± Std Dev')

# Also show individual points with transparency
for mu in mu_values:
    data_at_mu = df[df['mu'] == mu]['relative_change']
    x_jitter = np.random.normal(mu, 0.02, size=len(data_at_mu))
    ax.scatter(x_jitter, data_at_mu, alpha=0.1, s=10, color='gray')

ax.set_xlabel('Mean Noise (μ)', fontsize=12, fontweight='bold')
ax.set_ylabel('Relative SE Change (ΔSE / SE_before)', fontsize=12, fontweight='bold')
ax.set_title('Semantic Entropy Relative Change vs Mean Noise', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plot1_path = plot_dir / 'relative_change_vs_mu.png'
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plot1_path}")
plt.close()

# Plot 2: Relative Change vs Std Noise (sigma)
print(f"\nCreating Plot 2: Relative Change vs Std Noise (sigma)...")
fig, ax = plt.subplots(figsize=(10, 6))

# Group by sigma and calculate statistics
sigma_stats = df.groupby('sigma')['relative_change'].agg(['mean', 'std', 'count']).reset_index()
sigma_values = sorted(df['sigma'].unique())

# Plot with error bars
ax.errorbar(sigma_stats['sigma'], sigma_stats['mean'], yerr=sigma_stats['std'],
            marker='s', markersize=8, capsize=5, capthick=2, linewidth=2,
            label='Mean ± Std Dev', color='darkgreen')

# Also show individual points with transparency
for sigma in sigma_values:
    data_at_sigma = df[df['sigma'] == sigma]['relative_change']
    x_jitter = np.random.normal(sigma, 0.02, size=len(data_at_sigma))
    ax.scatter(x_jitter, data_at_sigma, alpha=0.1, s=10, color='gray')

ax.set_xlabel('Std Noise (σ)', fontsize=12, fontweight='bold')
ax.set_ylabel('Relative SE Change (ΔSE / SE_before)', fontsize=12, fontweight='bold')
ax.set_title('Semantic Entropy Relative Change vs Std Noise', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plot2_path = plot_dir / 'relative_change_vs_sigma.png'
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plot2_path}")
plt.close()

# Create combined plot
print(f"\nCreating combined plot...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: vs mu
ax1.errorbar(mu_stats['mu'], mu_stats['mean'], yerr=mu_stats['std'], 
            marker='o', markersize=8, capsize=5, capthick=2, linewidth=2,
            label='Mean ± Std Dev')
for mu in mu_values:
    data_at_mu = df[df['mu'] == mu]['relative_change']
    x_jitter = np.random.normal(mu, 0.02, size=len(data_at_mu))
    ax1.scatter(x_jitter, data_at_mu, alpha=0.1, s=10, color='gray')
ax1.set_xlabel('Mean Noise (μ)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Relative SE Change (ΔSE / SE_before)', fontsize=12, fontweight='bold')
ax1.set_title('(a) Relative Change vs Mean Noise', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)
ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

# Right plot: vs sigma
ax2.errorbar(sigma_stats['sigma'], sigma_stats['mean'], yerr=sigma_stats['std'],
            marker='s', markersize=8, capsize=5, capthick=2, linewidth=2,
            label='Mean ± Std Dev', color='darkgreen')
for sigma in sigma_values:
    data_at_sigma = df[df['sigma'] == sigma]['relative_change']
    x_jitter = np.random.normal(sigma, 0.02, size=len(data_at_sigma))
    ax2.scatter(x_jitter, data_at_sigma, alpha=0.1, s=10, color='gray')
ax2.set_xlabel('Std Noise (σ)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Relative SE Change (ΔSE / SE_before)', fontsize=12, fontweight='bold')
ax2.set_title('(b) Relative Change vs Std Noise', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

plt.suptitle('Semantic Entropy Relative Change Under Noise Perturbations', 
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
combined_path = plot_dir / 'relative_change_combined.png'
plt.savefig(combined_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {combined_path}")
plt.close()

# Save statistics to CSV
print(f"\nSaving statistics...")
mu_stats.to_csv(plot_dir / 'mu_statistics.csv', index=False)
sigma_stats.to_csv(plot_dir / 'sigma_statistics.csv', index=False)
print(f"✓ Saved: {plot_dir / 'mu_statistics.csv'}")
print(f"✓ Saved: {plot_dir / 'sigma_statistics.csv'}")

print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
print(f"Corrected JSONL: {output_file}")
print(f"Plots directory: {plot_dir}")
print(f"  - relative_change_vs_mu.png")
print(f"  - relative_change_vs_sigma.png")
print(f"  - relative_change_combined.png")
print(f"\n✓ All tasks completed successfully!")
