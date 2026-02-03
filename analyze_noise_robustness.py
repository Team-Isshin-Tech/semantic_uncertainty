"""Analyze noise robustness results and generate plots and CSV files."""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
import sys
from scipy import stats

def load_noise_results(wandb_run_id):
    """Load SE_after noise results from wandb run directory."""
    
    user = os.environ.get('USER', 'unknown')
    scratch_dir = os.getenv('SCRATCH_DIR', '.')
    wandb_dir = f'{scratch_dir}/{user}/uncertainty'
    
    # Search for run directory
    run_patterns = [
        os.path.join(wandb_dir, f'run-*{wandb_run_id}', 'se_after_noise.jsonl'),
        os.path.join(wandb_dir, f'run-{wandb_run_id}', 'files', 'se_after_noise.jsonl'),
    ]
    
    jsonl_path = None
    for pattern in run_patterns:
        candidates = glob.glob(pattern)
        if candidates:
            jsonl_path = candidates[0]
            break
    
    if not jsonl_path:
        raise FileNotFoundError(f"Could not find se_after_noise.jsonl for run {wandb_run_id}")
    
    print(f"Loading noise results from: {jsonl_path}")
    
    # Load JSONL
    records = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    df = pd.DataFrame(records)
    print(f"✅ Loaded {len(df)} noise perturbation records")
    print(f"   Questions: {df['question_id'].nunique()}")
    print(f"   Noise configs: μ={sorted(df['mu'].unique())}, σ={sorted(df['sigma'].unique())}")
    
    return df


def generate_csv_files(df, output_dir='analysis_results'):
    """Generate summary CSV files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("GENERATING CSV FILES")
    print("=" * 80)
    
    # 1. Per-question summary
    question_summary = df.groupby('question_id').agg({
        'se_before': 'first',
        'correctness': 'first',
        'delta_se': ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    question_summary.columns = ['_'.join(col).strip('_') for col in question_summary.columns.values]
    question_summary_path = os.path.join(output_dir, 'per_question_summary.csv')
    question_summary.to_csv(question_summary_path, index=False)
    print(f"✅ Saved: {question_summary_path}")
    
    # 2. Per-noise-config summary
    noise_summary = df.groupby(['mu', 'sigma']).agg({
        'delta_se': ['mean', 'std'],
        'se_mean': 'mean',
        'se_std': 'mean',
        'correctness': 'mean'
    }).reset_index()
    
    noise_summary.columns = ['_'.join(col).strip('_') for col in noise_summary.columns.values]
    noise_summary_path = os.path.join(output_dir, 'per_noise_config_summary.csv')
    noise_summary.to_csv(noise_summary_path, index=False)
    print(f"✅ Saved: {noise_summary_path}")
    
    # 3. Correlation analysis
    correlations = []
    for mu in df['mu'].unique():
        for sigma in df['sigma'].unique():
            subset = df[(df['mu'] == mu) & (df['sigma'] == sigma)]
            if len(subset) > 1:
                corr, pval = stats.pearsonr(subset['delta_se'], subset['correctness'])
                correlations.append({
                    'mu': mu,
                    'sigma': sigma,
                    'correlation': corr,
                    'p_value': pval,
                    'n_samples': len(subset)
                })
    
    corr_df = pd.DataFrame(correlations)
    corr_path = os.path.join(output_dir, 'correlation_delta_se_correctness.csv')
    corr_df.to_csv(corr_path, index=False)
    print(f"✅ Saved: {corr_path}")
    print(f"   Correlations range: [{corr_df['correlation'].min():.4f}, {corr_df['correlation'].max():.4f}]")
    
    # 4. Full detailed results
    full_path = os.path.join(output_dir, 'full_noise_results.csv')
    df.to_csv(full_path, index=False)
    print(f"✅ Saved: {full_path}")
    
    return question_summary, noise_summary, corr_df


def generate_plots(df, output_dir='analysis_results/plots'):
    """Generate analysis plots."""
    
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style('whitegrid')
    
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    
    # 1. Delta SE vs Correctness (grid by mu and sigma)
    print("  Generating: delta_se_vs_correctness_grid.png...")
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    idx = 0
    for sigma in sorted(df['sigma'].unique()):
        for mu in sorted(df['mu'].unique()):
            ax = axes[idx]
            subset = df[(df['mu'] == mu) & (df['sigma'] == sigma)]
            
            correct = subset[subset['correctness'] == 1.0]
            incorrect = subset[subset['correctness'] == 0.0]
            
            ax.scatter(correct['se_before'], correct['delta_se'], 
                      alpha=0.6, label='Correct', c='green', s=50)
            ax.scatter(incorrect['se_before'], incorrect['delta_se'], 
                      alpha=0.6, label='Incorrect', c='red', s=50)
            
            ax.set_title(f'μ={mu}, σ={sigma}', fontsize=12, fontweight='bold')
            ax.set_xlabel('SE_before', fontsize=10)
            ax.set_ylabel('ΔSE', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            idx += 1
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'delta_se_vs_correctness_grid.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {plot_path}")
    
    # 2. Correlation heatmap
    print("  Generating: correlation_heatmap.png...")
    corr_matrix = df.groupby(['mu', 'sigma']).apply(
        lambda x: x[['delta_se', 'correctness']].corr().iloc[0, 1] if len(x) > 1 else 0
    ).unstack()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                center=0, vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'})
    plt.title('Correlation(ΔSE, Correctness) by Noise Configuration', fontsize=14, fontweight='bold')
    plt.xlabel('σ (Noise Std Dev)', fontsize=12)
    plt.ylabel('μ (Noise Mean)', fontsize=12)
    plt.tight_layout()
    
    heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {heatmap_path}")
    
    # 3. Distribution of Delta SE by correctness
    print("  Generating: delta_se_distribution.png...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    sigmas = sorted(df['sigma'].unique())
    for idx, sigma in enumerate(sigmas):
        ax = axes[idx]
        
        subset = df[df['sigma'] == sigma]
        correct = subset[subset['correctness'] == 1.0]['delta_se']
        incorrect = subset[subset['correctness'] == 0.0]['delta_se']
        
        ax.hist(correct, bins=30, alpha=0.6, label='Correct', color='green', density=True)
        ax.hist(incorrect, bins=30, alpha=0.6, label='Incorrect', color='red', density=True)
        
        ax.set_title(f'σ={sigma}', fontsize=12, fontweight='bold')
        ax.set_xlabel('ΔSE', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    dist_path = os.path.join(output_dir, 'delta_se_distribution.png')
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {dist_path}")
    
    # 4. Mean Delta SE by noise config
    print("  Generating: mean_delta_se_by_config.png...")
    mean_delta = df.groupby(['mu', 'sigma', 'correctness'])['delta_se'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for correctness, group in mean_delta.groupby('correctness'):
        label = 'Correct' if correctness == 1.0 else 'Incorrect'
        color = 'green' if correctness == 1.0 else 'red'
        
        x_labels = [f'μ={row.mu}\nσ={row.sigma}' for _, row in group.iterrows()]
        x_pos = np.arange(len(x_labels))
        
        ax.plot(x_pos, group['delta_se'].values, marker='o', label=label, 
               color=color, linewidth=2.5, markersize=8)
    
    ax.set_xlabel('Noise Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean ΔSE', fontsize=12, fontweight='bold')
    ax.set_title('Mean ΔSE by Noise Configuration and Correctness', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    mean_path = os.path.join(output_dir, 'mean_delta_se_by_config.png')
    plt.savefig(mean_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {mean_path}")
    
    # 5. SE Before vs SE Mean scatter
    print("  Generating: se_before_vs_se_mean.png...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    correct = df[df['correctness'] == 1.0]
    incorrect = df[df['correctness'] == 0.0]
    
    ax.scatter(correct['se_before'], correct['se_mean'], 
              alpha=0.5, label='Correct', c='green', s=30)
    ax.scatter(incorrect['se_before'], incorrect['se_mean'], 
              alpha=0.5, label='Incorrect', c='red', s=30)
    
    ax.set_xlabel('SE Before (Original)', fontsize=12, fontweight='bold')
    ax.set_ylabel('SE After (Noisy, Mean)', fontsize=12, fontweight='bold')
    ax.set_title('SE Before vs SE After Noise', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    scatter_path = os.path.join(output_dir, 'se_before_vs_se_mean.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {scatter_path}")


def main(wandb_run_id):
    """Main analysis pipeline."""
    
    print("=" * 80)
    print("NOISE ROBUSTNESS ANALYSIS")
    print("=" * 80)
    
    # Load data
    df = load_noise_results(wandb_run_id)
    
    # Generate CSVs
    generate_csv_files(df)
    
    # Generate plots
    generate_plots(df)
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nResults saved to:")
    print("  - analysis_results/*.csv")
    print("  - analysis_results/plots/*.png")
    print("\nFiles generated:")
    print("  CSV:")
    print("    - per_question_summary.csv")
    print("    - per_noise_config_summary.csv")
    print("    - correlation_delta_se_correctness.csv")
    print("    - full_noise_results.csv")
    print("\n  Plots:")
    print("    - delta_se_vs_correctness_grid.png")
    print("    - correlation_heatmap.png")
    print("    - delta_se_distribution.png")
    print("    - mean_delta_se_by_config.png")
    print("    - se_before_vs_se_mean.png")


if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_noise_robustness.py <wandb_run_id>")
        sys.exit(1)
    
    wandb_run_id = sys.argv[1]
    main(wandb_run_id)
