"""
STAGE 2: Noise Analysis (load logits, add noise, compute SE metrics)

This script focuses purely on:
1. Loading validation_generations.pkl from a Stage 1 run
2. Computing SE before noise
3. For each noise configuration:
   - Adding 100 samples of noise
   - Computing 100 noisy SE values
   - Computing mean and std
4. Storing detailed results to WandB

Output columns:
- se_before: SE without noise
- noise_mu, noise_sigma: noise parameters
- se_noisy_001 to se_noisy_100: individual noisy SE values
- se_mean: mean of 100 noisy SE values
- se_std: std of 100 noisy SE values
- delta_se: se_before - se_mean
- abs_delta_se: |delta_se|

Usage:
    python stage2_noise_analysis.py --eval_wandb_runid <run_id>
"""

import os
import sys
import json
import pickle
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    with open('.env') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, val = line.strip().split('=', 1)
                os.environ[key] = val

import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm

# Import from semantic_uncertainty module
sys.path.insert(0, str(Path(__file__).parent / 'semantic_uncertainty'))
from uncertainty_measures.semantic_entropy import predictive_entropy_rao

# Constants
NOISE_MEANS = [0.0, 0.5, 1.0, 2.0]
NOISE_STDS = [0.5, 1.0, 2.0]
N_NOISE_SAMPLES = 100
TORCH_DTYPE = torch.float32


def compute_semantic_entropy(token_log_probs: List[np.ndarray], semantic_ids: Dict[int, int]) -> float:
    """Compute semantic entropy from token log probabilities and semantic IDs.
    
    Args:
        token_log_probs: List of log-probability distributions (vocab_size,)
        semantic_ids: Dict mapping token_id -> semantic_id
        
    Returns:
        Semantic entropy value (clamped to [0, inf))
    """
    if not token_log_probs or not semantic_ids:
        return 0.0
    
    # Aggregate by semantic cluster
    semantic_logprobs = {}
    for idx, log_probs in enumerate(token_log_probs):
        log_probs_tensor = torch.from_numpy(log_probs).to(TORCH_DTYPE)
        
        for token_id, semantic_id in semantic_ids.items():
            if token_id < len(log_probs):
                if semantic_id not in semantic_logprobs:
                    semantic_logprobs[semantic_id] = []
                semantic_logprobs[semantic_id].append(log_probs_tensor[token_id].item())
    
    # Compute entropy by semantic cluster
    entropies = []
    for semantic_id, logprobs in semantic_logprobs.items():
        logprobs_array = np.array(logprobs)
        probs = np.exp(logprobs_array - np.max(logprobs_array))  # for numerical stability
        probs = probs / np.sum(probs)
        entropy = -np.sum(probs * logprobs_array)
        entropies.append(entropy)
    
    # Return mean entropy across clusters
    mean_entropy = np.mean(entropies) if entropies else 0.0
    # Clamp to avoid negative values from floating point errors
    return max(float(mean_entropy), 0.0)


def add_noise_to_logits(logits_per_token: List[np.ndarray], mu: float, sigma: float) -> List[np.ndarray]:
    """Add Gaussian noise to logits.
    
    Args:
        logits_per_token: List of logit arrays (vocab_size,)
        mu: Mean of noise
        sigma: Std of noise
        
    Returns:
        Noisy logits with same structure
    """
    noisy_logits = []
    for logits in logits_per_token:
        noise = np.random.normal(mu, sigma, size=logits.shape)
        noisy_logits.append(logits + noise)
    return noisy_logits


def logits_to_log_probs(logits_per_token: List[np.ndarray]) -> List[np.ndarray]:
    """Convert logits to log probabilities using log-softmax.
    
    Args:
        logits_per_token: List of logit arrays (vocab_size,)
        
    Returns:
        List of log probability arrays
    """
    log_probs = []
    for logits in logits_per_token:
        # Use log-softmax for numerical stability
        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)
        log_probs_item = (logits - logits_max) - np.log(np.sum(exp_logits))
        log_probs.append(log_probs_item)
    return log_probs


def main(args):
    """Main pipeline for Stage 2: Noise Analysis"""
    
    print("\n" + "="*80)
    print("STAGE 2: NOISE ANALYSIS")
    print("="*80)
    
    # Initialize WandB and connect to parent run
    print(f"\n[0/3] Connecting to WandB run: {args.eval_wandb_runid}")
    
    api = wandb.Api()
    entity = os.environ.get('WANDB_SEM_UNC_ENTITY', 'raveendiran-21-university-of-moratuwa')
    parent_run = api.run(f"{entity}/semantic_uncertainty/{args.eval_wandb_runid}")
    
    print(f"  ✓ Found parent run: {parent_run.name}")
    print(f"    Path: {parent_run.path}")
    
    # Initialize new run for analysis
    wandb.init(
        project="semantic_uncertainty",
        name=f"noise-analysis-{parent_run.name}",
        config={
            'parent_run_id': args.eval_wandb_runid,
            'noise_means': NOISE_MEANS,
            'noise_stds': NOISE_STDS,
            'n_noise_samples': N_NOISE_SAMPLES,
        },
        tags=["stage2-noise-analysis"],
    )
    
    print(f"\n✓ Analysis run initialized: {wandb.run.name}")
    print(f"  Run ID: {wandb.run.id}")
    
    # Download validation_generations.pkl
    print(f"\n[1/3] Downloading validation_generations.pkl from parent run")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_file = parent_run.file('validation_generations.pkl')
        pkl_file.download(root=tmpdir, replace=True)
        
        pkl_path = Path(tmpdir) / 'validation_generations.pkl'
        with open(pkl_path, 'rb') as f:
            generations = pickle.load(f)
        
        print(f"  ✓ Loaded {len(generations)} questions")
        
        # Verify structure
        sample_qid = list(generations.keys())[0]
        sample_q = generations[sample_qid]
        sample_response = sample_q['responses'][0]
        
        has_logits = len(sample_response) > 3 and sample_response[3] is not None
        has_semantic_ids = 'semantic_ids' in sample_q
        
        print(f"    ✓ Has logits: {has_logits}")
        print(f"    ✓ Has semantic_ids: {has_semantic_ids}")
        
        if not has_logits:
            raise ValueError("Logits not found in validation_generations.pkl!")
        if not has_semantic_ids:
            raise ValueError("Semantic IDs not found in validation_generations.pkl!")
        
        # Compute SE before noise and collect results
        print(f"\n[2/3] Computing SE metrics for {len(NOISE_MEANS) * len(NOISE_STDS)} noise configurations")
        print(f"      ({N_NOISE_SAMPLES} samples each)")
        
        all_results = []
        
        for qid_idx, qid in enumerate(tqdm(generations.keys(), desc="Questions")):
            question_data = generations[qid]
            semantic_ids = question_data.get('semantic_ids', {})
            
            # Compute SE before noise
            if len(question_data['responses']) > 0:
                first_response = question_data['responses'][0]
                if len(first_response) > 1:
                    token_log_probs_before = logits_to_log_probs(first_response[3])
                    se_before = compute_semantic_entropy(token_log_probs_before, semantic_ids)
                else:
                    se_before = 0.0
            else:
                se_before = 0.0
            
            # For each noise configuration
            for mu in NOISE_MEANS:
                for sigma in NOISE_STDS:
                    se_noisy_values = []
                    
                    # Generate N_NOISE_SAMPLES noisy versions
                    for sample_idx in range(N_NOISE_SAMPLES):
                        # Take first response and add noise
                        logits_per_token = first_response[3]
                        noisy_logits = add_noise_to_logits(logits_per_token, mu, sigma)
                        token_log_probs_noisy = logits_to_log_probs(noisy_logits)
                        se_noisy = compute_semantic_entropy(token_log_probs_noisy, semantic_ids)
                        se_noisy_values.append(se_noisy)
                    
                    # Compute statistics
                    se_mean = np.mean(se_noisy_values)
                    se_std = np.std(se_noisy_values)
                    delta_se = se_before - se_mean
                    abs_delta_se = abs(delta_se)
                    
                    # Store result
                    result_row = {
                        'question_id': str(qid),
                        'se_before': se_before,
                        'noise_mu': mu,
                        'noise_sigma': sigma,
                    }
                    
                    # Add individual noisy values
                    for sample_idx, se_val in enumerate(se_noisy_values, 1):
                        result_row[f'se_noisy_{sample_idx:03d}'] = se_val
                    
                    # Add statistics
                    result_row['se_mean'] = se_mean
                    result_row['se_std'] = se_std
                    result_row['delta_se'] = delta_se
                    result_row['abs_delta_se'] = abs_delta_se
                    
                    all_results.append(result_row)
        
        # Create DataFrame
        print(f"\n  ✓ Created {len(all_results)} result rows")
        df_results = pd.DataFrame(all_results)
        
        # Save to CSV
        print(f"\n[3/3] Saving results to WandB")
        
        csv_path = Path(wandb.run.dir) / 'noise_analysis_results.csv'
        df_results.to_csv(csv_path, index=False)
        print(f"  ✓ Saved to {csv_path}")
        print(f"    Shape: {df_results.shape}")
        print(f"    Columns: {list(df_results.columns)[:10]}...")
        
        # Upload to WandB
        wandb.save(str(csv_path))
        print(f"  ✓ Uploaded to WandB")
        
        # Log summary statistics
        print(f"\n  Summary statistics:")
        print(f"    SE_before: mean={df_results['se_before'].mean():.4f}, std={df_results['se_before'].std():.4f}")
        print(f"    SE_mean: mean={df_results['se_mean'].mean():.4f}, std={df_results['se_mean'].std():.4f}")
        print(f"    Delta_SE: mean={df_results['delta_se'].mean():.4f}, std={df_results['delta_se'].std():.4f}")
        print(f"    Abs_Delta_SE: mean={df_results['abs_delta_se'].mean():.4f}, std={df_results['abs_delta_se'].std():.4f}")
        
        # Log to WandB
        wandb.summary['num_questions'] = len(generations)
        wandb.summary['num_noise_configs'] = len(NOISE_MEANS) * len(NOISE_STDS)
        wandb.summary['total_rows'] = len(df_results)
        wandb.summary['se_before_mean'] = float(df_results['se_before'].mean())
        wandb.summary['se_before_std'] = float(df_results['se_before'].std())
        wandb.summary['se_mean_mean'] = float(df_results['se_mean'].mean())
        wandb.summary['se_mean_std'] = float(df_results['se_mean'].std())
        wandb.summary['delta_se_mean'] = float(df_results['delta_se'].mean())
        wandb.summary['delta_se_std'] = float(df_results['delta_se'].std())
        
        print("\n" + "="*80)
        print("✓ STAGE 2 COMPLETE")
        print("="*80)
        print(f"\nWandB Run: {wandb.run.path}")
        print(f"Results CSV: noise_analysis_results.csv ({len(df_results)} rows)")
        print(f"\nColumns in CSV:")
        print(f"  - question_id")
        print(f"  - se_before")
        print(f"  - noise_mu, noise_sigma")
        print(f"  - se_noisy_001 to se_noisy_{N_NOISE_SAMPLES:03d} ({N_NOISE_SAMPLES} columns)")
        print(f"  - se_mean, se_std")
        print(f"  - delta_se, abs_delta_se")
        
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 2: Noise analysis')
    
    parser.add_argument('--eval_wandb_runid', type=str, required=True,
                        help='WandB run ID from Stage 1 (generation)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for noise generation')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    main(args)
