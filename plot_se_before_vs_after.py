import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('results/se_after_noise_complete.csv')

# Take the first 400 questions
df_subset = df.head(400)

# Separate data by correctness label
correct = df_subset[df_subset['correctness'] == 1.0]
incorrect = df_subset[df_subset['correctness'] == 0.0]

print(f"Total points: {len(df_subset)}")
print(f"Correct (correctness=1): {len(correct)}")
print(f"Incorrect (correctness=0): {len(incorrect)}")

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot correct answers in one color
ax.scatter(correct['se_before'], correct['se_mean'], 
          color='green', alpha=0.6, s=80, label='Correct (correctness=1)', edgecolors='darkgreen', linewidth=0.5)

# Plot incorrect answers in another color
ax.scatter(incorrect['se_before'], incorrect['se_mean'], 
          color='red', alpha=0.6, s=80, label='Incorrect (correctness=0)', edgecolors='darkred', linewidth=0.5)

# Add labels and title
ax.set_xlabel('SE Before', fontsize=12, fontweight='bold')
ax.set_ylabel('SE After (Mean)', fontsize=12, fontweight='bold')
ax.set_title('Semantic Entropy Before vs After Noise\n(400 Questions)', fontsize=14, fontweight='bold')

# Add legend
ax.legend(fontsize=11, loc='best')

# Add grid for better readability
ax.grid(True, alpha=0.3, linestyle='--')

# Tight layout
plt.tight_layout()

# Save the plot
plt.savefig('plots/se_before_vs_after_400points.png', dpi=300, bbox_inches='tight')
print("Plot saved to: plots/se_before_vs_after_400points.png")

# Show the plot
plt.show()
