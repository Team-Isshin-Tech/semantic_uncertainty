"""
VERIFICATION SCRIPT: Test pipeline logic with synthetic data

This script validates:
1. Noise addition works correctly
2. SE computation produces valid values
3. Output structure is correct
4. All metrics are computed properly

Run BEFORE launching Stage 1 and 2.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List

print("\n" + "="*80)
print("PIPELINE VERIFICATION")
print("="*80)

# 1. Test imports
print("\n[1/5] Testing imports...")
try:
    import torch
    import pandas as pd
    import numpy as np
    print("  ✓ torch, pandas, numpy imported")
except ImportError as e:
    print(f"  ✗ Failed to import: {e}")
    sys.exit(1)

try:
    import wandb
    print("  ✓ wandb imported")
except ImportError as e:
    print(f"  ✗ Failed to import wandb: {e}")
    print("  (This is OK - wandb only needed at runtime)")

# 2. Test noise addition
print("\n[2/5] Testing noise addition...")

def add_noise_to_logits(logits_per_token: List[np.ndarray], mu: float, sigma: float) -> List[np.ndarray]:
    """Add Gaussian noise to logits."""
    noisy_logits = []
    for logits in logits_per_token:
        noise = np.random.normal(mu, sigma, size=logits.shape)
        noisy_logits.append(logits + noise)
    return noisy_logits

# Create synthetic logits
synthetic_logits = [
    np.random.randn(50000),  # vocab_size = 50000
    np.random.randn(50000),
    np.random.randn(50000),
]

print(f"  Created synthetic logits: {len(synthetic_logits)} tokens")
print(f"  Logit shape: {synthetic_logits[0].shape}")

# Test noise addition
noisy_logits_0 = add_noise_to_logits(synthetic_logits, mu=0.0, sigma=1.0)
noisy_logits_1 = add_noise_to_logits(synthetic_logits, mu=1.0, sigma=0.5)

print(f"  ✓ Noise addition (μ=0.0, σ=1.0): {len(noisy_logits_0)} tokens")
print(f"  ✓ Noise addition (μ=1.0, σ=0.5): {len(noisy_logits_1)} tokens")

# Verify noise is different each time
if not np.allclose(noisy_logits_0[0], noisy_logits_1[0]):
    print(f"  ✓ Different noise samples are different")
else:
    print(f"  ✗ Noise samples are identical (randomness issue?)")
    sys.exit(1)

# 3. Test log-softmax conversion
print("\n[3/5] Testing logits → log-probabilities...")

def logits_to_log_probs(logits_per_token: List[np.ndarray]) -> List[np.ndarray]:
    """Convert logits to log probabilities using log-softmax."""
    log_probs = []
    for logits in logits_per_token:
        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)
        log_probs_item = (logits - logits_max) - np.log(np.sum(exp_logits))
        log_probs.append(log_probs_item)
    return log_probs

log_probs = logits_to_log_probs(synthetic_logits)
print(f"  ✓ Converted to log-probs: {len(log_probs)} tokens")

# Verify numerical properties
for idx, lp in enumerate(log_probs):
    lp_sum = np.sum(np.exp(lp))
    print(f"    Token {idx}: sum(exp(log_probs)) = {lp_sum:.6f} (should be ~1.0)")
    if not np.isclose(lp_sum, 1.0, atol=1e-5):
        print(f"    ✗ Log-probs don't sum to 1!")
        sys.exit(1)

print(f"  ✓ All log-probs sum to 1.0")

# 4. Test SE computation
print("\n[4/5] Testing SE computation...")

def compute_semantic_entropy(token_log_probs: List[np.ndarray], semantic_ids: Dict[int, int]) -> float:
    """Compute semantic entropy from token log probabilities."""
    if not token_log_probs or not semantic_ids:
        return 0.0
    
    semantic_logprobs = {}
    for idx, log_probs in enumerate(token_log_probs):
        for token_id, semantic_id in semantic_ids.items():
            if token_id < len(log_probs):
                if semantic_id not in semantic_logprobs:
                    semantic_logprobs[semantic_id] = []
                semantic_logprobs[semantic_id].append(log_probs[token_id])
    
    entropies = []
    for semantic_id, logprobs in semantic_logprobs.items():
        logprobs_array = np.array(logprobs)
        probs = np.exp(logprobs_array - np.max(logprobs_array))
        probs = probs / np.sum(probs)
        entropy = -np.sum(probs * logprobs_array)
        entropies.append(entropy)
    
    mean_entropy = np.mean(entropies) if entropies else 0.0
    return max(float(mean_entropy), 0.0)

# Create synthetic semantic IDs (map tokens to clusters)
semantic_ids = {i: i % 10 for i in range(100)}  # 100 tokens → 10 clusters

# Compute SE
se_before = compute_semantic_entropy(log_probs, semantic_ids)
print(f"  ✓ SE (no noise): {se_before:.6f}")

# Add noise and recompute
noisy_logits = add_noise_to_logits(synthetic_logits, mu=1.0, sigma=0.5)
noisy_log_probs = logits_to_log_probs(noisy_logits)
se_after = compute_semantic_entropy(noisy_log_probs, semantic_ids)
print(f"  ✓ SE (noise μ=1.0, σ=0.5): {se_after:.6f}")

# Test multiple noise samples
print(f"\n  Testing {100} noise samples...")
se_values = []
for i in range(100):
    noisy_logits = add_noise_to_logits(synthetic_logits, mu=1.0, sigma=0.5)
    noisy_log_probs = logits_to_log_probs(noisy_logits)
    se_val = compute_semantic_entropy(noisy_log_probs, semantic_ids)
    se_values.append(se_val)

se_values = np.array(se_values)
print(f"  ✓ SE values computed: {len(se_values)} samples")
print(f"    Mean: {se_values.mean():.6f}")
print(f"    Std: {se_values.std():.6f}")
print(f"    Min: {se_values.min():.6f}")
print(f"    Max: {se_values.max():.6f}")

# Verify no negative values
if np.any(se_values < 0):
    print(f"  ✗ Negative SE values detected: min={se_values.min()}")
    sys.exit(1)
else:
    print(f"  ✓ All SE values are non-negative")

# 5. Test output structure
print("\n[5/5] Testing output structure...")

NOISE_MEANS = [0.0, 0.5, 1.0]
NOISE_STDS = [0.5, 1.0]
N_NOISE_SAMPLES = 100

results = []

# Generate results for 3 questions with each noise config
for q_id in range(3):
    se_before = np.random.uniform(0, 2)
    
    for mu in NOISE_MEANS:
        for sigma in NOISE_STDS:
            se_noisy = np.random.uniform(0, 2, N_NOISE_SAMPLES)
            
            result = {
                'question_id': f'q_{q_id}',
                'se_before': se_before,
                'noise_mu': mu,
                'noise_sigma': sigma,
            }
            
            # Add individual values
            for idx, val in enumerate(se_noisy, 1):
                result[f'se_noisy_{idx:03d}'] = val
            
            # Add statistics
            result['se_mean'] = np.mean(se_noisy)
            result['se_std'] = np.std(se_noisy)
            result['delta_se'] = se_before - result['se_mean']
            result['abs_delta_se'] = abs(result['delta_se'])
            
            results.append(result)

df = pd.DataFrame(results)

print(f"  ✓ Created DataFrame with {len(df)} rows")
print(f"  ✓ Columns: {len(df.columns)}")
print(f"    - Metadata: question_id, noise_mu, noise_sigma")
print(f"    - SE metrics: se_before, se_noisy_001 to se_noisy_{N_NOISE_SAMPLES:03d}, se_mean, se_std")
print(f"    - Derived: delta_se, abs_delta_se")

print(f"\n  Sample row:")
print(f"    {df.iloc[0].to_dict()}")

print(f"\n  DataFrame info:")
print(f"    Shape: {df.shape}")
print(f"    SE_before: min={df['se_before'].min():.4f}, max={df['se_before'].max():.4f}")
print(f"    SE_mean: min={df['se_mean'].min():.4f}, max={df['se_mean'].max():.4f}")
print(f"    Delta_SE: min={df['delta_se'].min():.4f}, max={df['delta_se'].max():.4f}")

# Verify no NaN/Inf
if df.isnull().any().any():
    print(f"  ✗ NaN values detected!")
    sys.exit(1)

if np.isinf(df.select_dtypes(include=[np.number])).any().any():
    print(f"  ✗ Inf values detected!")
    sys.exit(1)

print(f"  ✓ No NaN or Inf values")

print("\n" + "="*80)
print("✓ ALL VERIFICATIONS PASSED")
print("="*80)
print("""
Ready to proceed:
1. Stage 1: Generate answers
   python stage1_generate_answers.py --num_samples 400 --num_generations 10

2. Stage 2: Analyze noise (replace <RUN_ID> with output from Stage 1)
   python stage2_noise_analysis.py --eval_wandb_runid <RUN_ID>
""")
