import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Read the CSV file
df = pd.read_csv('results/se_after_noise_complete.csv')
df_subset = df.head(400)

# Separate data by correctness label
correct = df_subset[df_subset['correctness'] == 1.0]
incorrect = df_subset[df_subset['correctness'] == 0.0]

print(f"Total points: {len(df_subset)}")
print(f"Correct (correctness=1): {len(correct)}")
print(f"Incorrect (correctness=0): {len(incorrect)}")
print(f"\nCorrect answers - SE Before: Mean={correct['se_before'].mean():.4f}, Std={correct['se_before'].std():.4f}")
print(f"Incorrect answers - SE Before: Mean={incorrect['se_before'].mean():.4f}, Std={incorrect['se_before'].std():.4f}")

# Create figure with 3 different plot types
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# ===== Plot 1: Strip Plot with Jitter and Transparency =====
ax1 = axes[0]
# Add jitter to y-axis for better visibility
np.random.seed(42)
jitter_strength = 0.04
correct_jitter = np.random.normal(1, jitter_strength, len(correct))
incorrect_jitter = np.random.normal(0, jitter_strength, len(incorrect))

ax1.scatter(correct['se_before'], correct_jitter, 
           color='green', alpha=0.5, s=60, edgecolors='darkgreen', linewidth=0.5, label='Correct (correctness=1)')
ax1.scatter(incorrect['se_before'], incorrect_jitter, 
           color='red', alpha=0.5, s=60, edgecolors='darkred', linewidth=0.5, label='Incorrect (correctness=0)')

ax1.set_xlabel('SE Before', fontsize=12, fontweight='bold')
ax1.set_ylabel('Correctness Label', fontsize=12, fontweight='bold')
ax1.set_yticks([0, 1])
ax1.set_yticklabels(['Incorrect (0)', 'Correct (1)'])
ax1.set_title('Scatter Plot with Jitter\n(Handles Overlapping)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3, axis='x')
ax1.set_ylim(-0.3, 1.3)

# ===== Plot 2: Violin Plot =====
ax2 = axes[1]
data_to_plot = [incorrect['se_before'].values, correct['se_before'].values]
parts = ax2.violinplot(data_to_plot, positions=[0, 1], widths=0.7, showmeans=True, showmedians=True)

# Color the violin plots
colors = ['red', 'green']
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.6)
    pc.set_edgecolor(f'dark{colors[i]}')

ax2.set_xticks([0, 1])
ax2.set_xticklabels(['Incorrect (0)', 'Correct (1)'])
ax2.set_ylabel('SE Before', fontsize=12, fontweight='bold')
ax2.set_title('Violin Plot\n(Shows Distribution Density)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# ===== Plot 3: Box Plot with Swarm Points =====
ax3 = axes[2]
# Create DataFrame for seaborn
plot_df = pd.DataFrame({
    'SE Before': pd.concat([correct['se_before'], incorrect['se_before']], ignore_index=True),
    'Correctness': ['Correct (1)'] * len(correct) + ['Incorrect (0)'] * len(incorrect)
})

# Box plot
bp = ax3.boxplot([incorrect['se_before'].values, correct['se_before'].values],
                   labels=['Incorrect (0)', 'Correct (1)'],
                   patch_artist=True, widths=0.6)

# Color the boxes
bp['boxes'][0].set_facecolor('red')
bp['boxes'][0].set_alpha(0.6)
bp['boxes'][1].set_facecolor('green')
bp['boxes'][1].set_alpha(0.6)

# Add individual points with jitter
x_positions = [1, 2]
for i, data in enumerate([incorrect['se_before'].values, correct['se_before'].values]):
    x = np.random.normal(x_positions[i], 0.04, size=len(data))
    ax3.scatter(x, data, alpha=0.4, s=40, color=['red', 'green'][i], edgecolors=['darkred', 'darkgreen'][i], linewidth=0.5)

ax3.set_ylabel('SE Before', fontsize=12, fontweight='bold')
ax3.set_title('Box Plot with Points\n(Distribution + Statistics)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('plots/correctness_vs_se_before.png', dpi=300, bbox_inches='tight')
print("\nPlot saved to: plots/correctness_vs_se_before.png")

# ===== Statistical Comparison =====
print(f"\n{'='*70}")
print(f"STATISTICAL COMPARISON: SE Before vs Correctness Label")
print(f"{'='*70}")

from scipy.stats import mannwhitneyu, ttest_ind

# T-test
t_stat, t_pval = ttest_ind(incorrect['se_before'], correct['se_before'])
print(f"\nIndependent t-test:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {t_pval:.6f}")

# Mann-Whitney U test (non-parametric alternative)
u_stat, u_pval = mannwhitneyu(incorrect['se_before'], correct['se_before'], alternative='two-sided')
print(f"\nMann-Whitney U test (non-parametric):")
print(f"  U-statistic: {u_stat:.4f}")
print(f"  p-value: {u_pval:.6f}")

# Effect size (Cohen's d)
mean_diff = incorrect['se_before'].mean() - correct['se_before'].mean()
pooled_std = np.sqrt((incorrect['se_before'].std()**2 + correct['se_before'].std()**2) / 2)
cohens_d = mean_diff / pooled_std

print(f"\nEffect Size (Cohen's d):")
print(f"  Cohen's d: {cohens_d:.4f}")
if abs(cohens_d) < 0.2:
    effect = "negligible"
elif abs(cohens_d) < 0.5:
    effect = "small"
elif abs(cohens_d) < 0.8:
    effect = "medium"
else:
    effect = "large"
print(f"  Interpretation: {effect} effect")

print(f"\n{'='*70}")
print(f"DESCRIPTIVE STATISTICS")
print(f"{'='*70}")
print(f"\nIncorrect Answers (Confabulations):")
print(f"  Count: {len(incorrect)}")
print(f"  Mean: {incorrect['se_before'].mean():.4f}")
print(f"  Median: {incorrect['se_before'].median():.4f}")
print(f"  Std Dev: {incorrect['se_before'].std():.4f}")
print(f"  Min: {incorrect['se_before'].min():.4f}, Max: {incorrect['se_before'].max():.4f}")
print(f"  Q1: {incorrect['se_before'].quantile(0.25):.4f}, Q3: {incorrect['se_before'].quantile(0.75):.4f}")

print(f"\nCorrect Answers:")
print(f"  Count: {len(correct)}")
print(f"  Mean: {correct['se_before'].mean():.4f}")
print(f"  Median: {correct['se_before'].median():.4f}")
print(f"  Std Dev: {correct['se_before'].std():.4f}")
print(f"  Min: {correct['se_before'].min():.4f}, Max: {correct['se_before'].max():.4f}")
print(f"  Q1: {correct['se_before'].quantile(0.25):.4f}, Q3: {correct['se_before'].quantile(0.75):.4f}")

plt.show()
