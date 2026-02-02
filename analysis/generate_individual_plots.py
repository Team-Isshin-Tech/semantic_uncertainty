"""Generate individual delta_se vs se_before plots for each noise configuration."""
import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_csv(csv_path):
    """Load noise robustness CSV."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Unique mu values: {sorted(df['mu'].unique())}")
    print(f"Unique sigma values: {sorted(df['sigma'].unique())}")
    return df


def plot_delta_vs_se_before_single(df_subset, mu, sigma, output_path):
    """
    Create scatter plot of delta_se vs se_before for a single (mu, sigma) pair.
    Color points by correctness: green=correct, red=incorrect.
    """
    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Split by correctness
    correct = df_subset[df_subset['correctness'] == 1.0].copy()
    incorrect = df_subset[df_subset['correctness'] == 0.0].copy()
    
    # Add small jitter to prevent perfect overlap (0.5% of range)
    np.random.seed(42)
    if len(correct) > 0:
        se_range = max(df_subset['se_before'].max() - df_subset['se_before'].min(), 0.01)
        delta_range = max(df_subset['delta_se'].max() - df_subset['delta_se'].min(), 0.01)
        correct['se_before_jitter'] = correct['se_before'] + np.random.normal(0, se_range * 0.005, len(correct))
        correct['delta_se_jitter'] = correct['delta_se'] + np.random.normal(0, delta_range * 0.005, len(correct))
    if len(incorrect) > 0:
        se_range = max(df_subset['se_before'].max() - df_subset['se_before'].min(), 0.01)
        delta_range = max(df_subset['delta_se'].max() - df_subset['delta_se'].min(), 0.01)
        incorrect['se_before_jitter'] = incorrect['se_before'] + np.random.normal(0, se_range * 0.005, len(incorrect))
        incorrect['delta_se_jitter'] = incorrect['delta_se'] + np.random.normal(0, delta_range * 0.005, len(incorrect))
    
    # Plot with larger markers and edge colors
    if len(correct) > 0:
        ax.scatter(correct['se_before_jitter'], correct['delta_se_jitter'], 
                   color='green', alpha=0.7, s=100, label=f'Correct ({len(correct)})', 
                   edgecolors='darkgreen', linewidths=1.5)
    if len(incorrect) > 0:
        ax.scatter(incorrect['se_before_jitter'], incorrect['delta_se_jitter'], 
                   color='red', alpha=0.7, s=100, label=f'Incorrect ({len(incorrect)})', 
                   edgecolors='darkred', linewidths=1.5)
    
    # Add zero reference line
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Labels and title
    ax.set_xlabel('SE_before (baseline entropy)', fontsize=12)
    ax.set_ylabel('Δ SE (SE_after_mean - SE_before)', fontsize=12)
    ax.set_title(f'Noise Robustness: μ={mu}, σ={sigma}\n(Per-token perturbation, n={len(df_subset)} questions)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Stats annotation
    n_total = len(df_subset)
    n_correct = len(correct)
    n_incorrect = len(incorrect)
    mean_delta = df_subset['delta_se'].mean()
    std_delta = df_subset['delta_se'].std()
    
    stats_text = f'N={n_total} (C={n_correct}, I={n_incorrect})\nΔSE: μ={mean_delta:.3f}, σ={std_delta:.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_delta_vs_mean_noise_single(df_subset, mu, sigma, output_path):
    """
    Create scatter plot of delta_se vs mean_noise for a single (mu, sigma) pair.
    Color points by correctness: green=correct, red=incorrect.
    """
    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Split by correctness
    correct = df_subset[df_subset['correctness'] == 1.0].copy()
    incorrect = df_subset[df_subset['correctness'] == 0.0].copy()
    
    # Add small jitter to prevent perfect overlap (0.5% of range)
    np.random.seed(42)
    if len(correct) > 0:
        noise_range = max(df_subset['mean_noise'].max() - df_subset['mean_noise'].min(), 0.01)
        delta_range = max(df_subset['delta_se'].max() - df_subset['delta_se'].min(), 0.01)
        correct['mean_noise_jitter'] = correct['mean_noise'] + np.random.normal(0, noise_range * 0.005, len(correct))
        correct['delta_se_jitter'] = correct['delta_se'] + np.random.normal(0, delta_range * 0.005, len(correct))
    if len(incorrect) > 0:
        noise_range = max(df_subset['mean_noise'].max() - df_subset['mean_noise'].min(), 0.01)
        delta_range = max(df_subset['delta_se'].max() - df_subset['delta_se'].min(), 0.01)
        incorrect['mean_noise_jitter'] = incorrect['mean_noise'] + np.random.normal(0, noise_range * 0.005, len(incorrect))
        incorrect['delta_se_jitter'] = incorrect['delta_se'] + np.random.normal(0, delta_range * 0.005, len(incorrect))
    
    # Plot with larger markers and edge colors
    if len(correct) > 0:
        ax.scatter(correct['mean_noise_jitter'], correct['delta_se_jitter'], 
                   color='green', alpha=0.7, s=100, label=f'Correct ({len(correct)})', 
                   edgecolors='darkgreen', linewidths=1.5)
    if len(incorrect) > 0:
        ax.scatter(incorrect['mean_noise_jitter'], incorrect['delta_se_jitter'], 
                   color='red', alpha=0.7, s=100, label=f'Incorrect ({len(incorrect)})', 
                   edgecolors='darkred', linewidths=1.5)
    
    # Add zero reference line for delta_se
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add vertical line for theoretical mean (mu)
    ax.axvline(x=mu, color='blue', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Theoretical μ={mu}')
    
    # Labels and title
    ax.set_xlabel('Mean Noise (empirical)', fontsize=12)
    ax.set_ylabel('Δ SE (SE_after_mean - SE_before)', fontsize=12)
    ax.set_title(f'Noise Impact: μ={mu}, σ={sigma}\n(Per-token perturbation, n={len(df_subset)} questions)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Stats annotation
    n_total = len(df_subset)
    n_correct = len(correct)
    n_incorrect = len(incorrect)
    mean_noise_empirical = df_subset['mean_noise'].mean()
    std_noise_empirical = df_subset['mean_noise'].std()
    
    stats_text = f'N={n_total} (C={n_correct}, I={n_incorrect})\nNoise: μ_emp={mean_noise_empirical:.3f}, σ_emp={std_noise_empirical:.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate individual plots for each noise config')
    parser.add_argument('--input', required=True, help='Path to noise_robustness_table.csv')
    parser.add_argument('--output-dir', required=True, help='Directory to save individual plots')
    args = parser.parse_args()
    
    # Load data
    df = load_csv(args.input)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Get unique (mu, sigma) pairs
    configs = df[['mu', 'sigma']].drop_duplicates().sort_values(['mu', 'sigma'])
    print(f"\nGenerating {len(configs) * 2} plots (2 per config: delta_se vs se_before + delta_se vs mean_noise)...")
    
    # Generate TWO plots per configuration
    plot_count = 0
    for idx, (_, row) in enumerate(configs.iterrows(), 1):
        mu = row['mu']
        sigma = row['sigma']
        
        # Filter data for this config
        df_subset = df[(df['mu'] == mu) & (df['sigma'] == sigma)].copy()
        
        # Plot 1: delta_se vs se_before
        output_filename_1 = f'plot_delta_se_vs_se_before_mean{mu}_std{sigma}.png'
        output_path_1 = os.path.join(args.output_dir, output_filename_1)
        plot_delta_vs_se_before_single(df_subset, mu, sigma, output_path_1)
        plot_count += 1
        print(f"  [{plot_count}/{len(configs)*2}] Generated: {output_filename_1} ({len(df_subset)} points)")
        
        # Plot 2: delta_se vs mean_noise
        output_filename_2 = f'plot_delta_se_vs_mean_noise_mean{mu}_std{sigma}.png'
        output_path_2 = os.path.join(args.output_dir, output_filename_2)
        plot_delta_vs_mean_noise_single(df_subset, mu, sigma, output_path_2)
        plot_count += 1
        print(f"  [{plot_count}/{len(configs)*2}] Generated: {output_filename_2} ({len(df_subset)} points)")
    
    print(f"\n✓ All {plot_count} plots saved to {args.output_dir}")
    print(f"  - {len(configs)} plots: delta_se vs se_before")
    print(f"  - {len(configs)} plots: delta_se vs mean_noise")


if __name__ == '__main__':
    main()
