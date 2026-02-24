import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc

# Read the CSV file
df = pd.read_csv('results/se_after_noise_complete.csv')
df_subset = df.head(400)

# Separate data by correctness label
correct = df_subset[df_subset['correctness'] == 1.0]['se_mean'].values
incorrect = df_subset[df_subset['correctness'] == 0.0]['se_mean'].values

print(f"Correct answers - Mean SE: {correct.mean():.4f}, Std: {correct.std():.4f}")
print(f"Incorrect answers - Mean SE: {incorrect.mean():.4f}, Std: {incorrect.std():.4f}")

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ===== Plot 1: Box Plot =====
ax1 = axes[0, 0]
box_data = [correct, incorrect]
bp = ax1.boxplot(box_data, labels=['Correct', 'Incorrect'], patch_artist=True)
bp['boxes'][0].set_facecolor('green')
bp['boxes'][0].set_alpha(0.6)
bp['boxes'][1].set_facecolor('red')
bp['boxes'][1].set_alpha(0.6)
ax1.set_ylabel('SE After (Mean)', fontsize=11, fontweight='bold')
ax1.set_title('Box Plot: SE Distribution by Correctness', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Add mean lines
ax1.hlines(correct.mean(), 0.7, 1.3, colors='darkgreen', linestyles='dashed', linewidth=2, label=f'Correct Mean: {correct.mean():.4f}')
ax1.hlines(incorrect.mean(), 1.7, 2.3, colors='darkred', linestyles='dashed', linewidth=2, label=f'Incorrect Mean: {incorrect.mean():.4f}')
ax1.legend(fontsize=10)

# ===== Plot 2: Violin Plot =====
ax2 = axes[0, 1]
parts = ax2.violinplot([correct, incorrect], positions=[1, 2], showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_alpha(0.6)
ax2.set_xticks([1, 2])
ax2.set_xticklabels(['Correct', 'Incorrect'])
ax2.set_ylabel('SE After (Mean)', fontsize=11, fontweight='bold')
ax2.set_title('Violin Plot: SE Distribution Density', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# ===== Plot 3: Overlapping Histograms with KDE =====
ax3 = axes[1, 0]
bins = np.linspace(min(correct.min(), incorrect.min()), max(correct.max(), incorrect.max()), 30)
ax3.hist(correct, bins=bins, alpha=0.6, label='Correct', color='green', density=True, edgecolor='darkgreen')
ax3.hist(incorrect, bins=bins, alpha=0.6, label='Incorrect', color='red', density=True, edgecolor='darkred')

# Add KDE curves
from scipy.stats import gaussian_kde
kde_correct = gaussian_kde(correct)
kde_incorrect = gaussian_kde(incorrect)
x_range = np.linspace(min(correct.min(), incorrect.min()), max(correct.max(), incorrect.max()), 200)
ax3.plot(x_range, kde_correct(x_range), 'g-', linewidth=2, label='Correct KDE')
ax3.plot(x_range, kde_incorrect(x_range), 'r-', linewidth=2, label='Incorrect KDE')

ax3.set_xlabel('SE After (Mean)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Density', fontsize=11, fontweight='bold')
ax3.set_title('Histogram + KDE: Probability Distribution', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# ===== Plot 4: Threshold Analysis with ROC =====
ax4 = axes[1, 1]

# Create binary labels for ROC
y_true = np.concatenate([np.ones(len(correct)), np.zeros(len(incorrect))])
y_scores = np.concatenate([correct, incorrect])

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
ax4.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax4.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random classifier')

# Find optimal threshold (Youden's index)
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]

ax4.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', s=100, label=f'Optimal Threshold: {optimal_threshold:.4f}')

ax4.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
ax4.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
ax4.set_title('ROC Curve for Optimal Threshold Detection', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/se_threshold_analysis_clustered.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: plots/se_threshold_analysis_clustered.png")

# ===== Threshold Statistics =====
print(f"\n{'='*60}")
print(f"THRESHOLD ANALYSIS FOR CONFABULATION DETECTION")
print(f"{'='*60}")
print(f"\nCorrect Answers (Accurate):")
print(f"  Mean: {correct.mean():.4f}")
print(f"  Median: {np.median(correct):.4f}")
print(f"  Std Dev: {correct.std():.4f}")
print(f"  Min: {correct.min():.4f}, Max: {correct.max():.4f}")

print(f"\nIncorrect Answers (Confabulated/Wrong):")
print(f"  Mean: {incorrect.mean():.4f}")
print(f"  Median: {np.median(incorrect):.4f}")
print(f"  Std Dev: {incorrect.std():.4f}")
print(f"  Min: {incorrect.min():.4f}, Max: {incorrect.max():.4f}")

print(f"\n{'='*60}")
print(f"SUGGESTED THRESHOLDS FOR CONFABULATION DETECTION:")
print(f"{'='*60}")

# Simple thresholds based on statistics
threshold_mean = (correct.mean() + incorrect.mean()) / 2
threshold_median = (np.median(correct) + np.median(incorrect)) / 2
threshold_youden = optimal_threshold

print(f"\n1. Midpoint of Means: {threshold_mean:.4f}")
print(f"   - Correct answers > {threshold_mean:.4f}: {(correct > threshold_mean).sum()} ({(correct > threshold_mean).sum()/len(correct)*100:.1f}%)")
print(f"   - Incorrect answers > {threshold_mean:.4f}: {(incorrect > threshold_mean).sum()} ({(incorrect > threshold_mean).sum()/len(incorrect)*100:.1f}%)")

print(f"\n2. Midpoint of Medians: {threshold_median:.4f}")
print(f"   - Correct answers > {threshold_median:.4f}: {(correct > threshold_median).sum()} ({(correct > threshold_median).sum()/len(correct)*100:.1f}%)")
print(f"   - Incorrect answers > {threshold_median:.4f}: {(incorrect > threshold_median).sum()} ({(incorrect > threshold_median).sum()/len(incorrect)*100:.1f}%)")

print(f"\n3. Youden's Index (ROC-based Optimal): {threshold_youden:.4f} (AUC: {roc_auc:.3f})")
print(f"   - Correct answers > {threshold_youden:.4f}: {(correct > threshold_youden).sum()} ({(correct > threshold_youden).sum()/len(correct)*100:.1f}%)")
print(f"   - Incorrect answers > {threshold_youden:.4f}: {(incorrect > threshold_youden).sum()} ({(incorrect > threshold_youden).sum()/len(incorrect)*100:.1f}%)")

# Calculate accuracy for each threshold
print(f"\n{'='*60}")
print(f"ACCURACY METRICS AT SUGGESTED THRESHOLDS:")
print(f"{'='*60}")

for name, thresh in [("Mean Midpoint", threshold_mean), ("Median Midpoint", threshold_median), ("Youden's Index", threshold_youden)]:
    tp = (correct > thresh).sum()  # correct answers flagged as potentially correct
    tn = (incorrect <= thresh).sum()  # incorrect answers flagged as potentially incorrect
    fp = (incorrect > thresh).sum()  # incorrect answers flagged as potentially correct
    fn = (correct <= thresh).sum()  # correct answers flagged as potentially incorrect
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n{name} (threshold = {thresh:.4f}):")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Sensitivity (True Positive Rate): {sensitivity:.4f}")
    print(f"  Specificity (True Negative Rate): {specificity:.4f}")
