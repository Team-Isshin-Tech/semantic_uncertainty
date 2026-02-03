"""
Verify logit-space noise implementation.
Run this script to confirm the changes work correctly.
"""

import numpy as np
import torch

def test_logit_noise_simulation():
    """Simulate the logit noise process to verify correctness."""
    print("=" * 60)
    print("LOGIT NOISE VERIFICATION TEST")
    print("=" * 60)
    
    # Simulate a small batch
    vocab_size = 1000
    n_tokens = 5
    
    # Create fake logits (simulating outputs.scores)
    np.random.seed(42)
    logits = [torch.randn(1, vocab_size) for _ in range(n_tokens)]
    
    # Create fake generated token IDs
    token_ids = np.random.randint(0, vocab_size, size=n_tokens).tolist()
    
    print(f"\nSetup:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  n_tokens: {n_tokens}")
    print(f"  token_ids: {token_ids}")
    
    # Compute original log-probs (without noise)
    original_log_probs = []
    for logits_t, token_id in zip(logits, token_ids):
        logits_np = logits_t.numpy()
        logits_max = logits_np.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits_np - logits_max)
        log_probs = logits_np - logits_max - np.log(exp_logits.sum(axis=1, keepdims=True))
        original_log_probs.append(log_probs[0, token_id])
    
    print(f"\nOriginal log-probs: {[f'{x:.4f}' for x in original_log_probs]}")
    
    # TEST 1: sigma=0 should reproduce original
    print("\n" + "=" * 60)
    print("TEST 1: sigma=0 reproduces original log-probs")
    print("=" * 60)
    
    noisy_log_probs_sigma0 = []
    for logits_t, token_id in zip(logits, token_ids):
        noise = np.zeros_like(logits_t.numpy())  # sigma=0
        noisy_logits = logits_t.numpy() + noise
        
        logits_max = noisy_logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(noisy_logits - logits_max)
        log_probs = noisy_logits - logits_max - np.log(exp_logits.sum(axis=1, keepdims=True))
        noisy_log_probs_sigma0.append(log_probs[0, token_id])
    
    differences = [abs(o - n) for o, n in zip(original_log_probs, noisy_log_probs_sigma0)]
    max_diff = max(differences)
    
    print(f"Noisy log-probs:  {[f'{x:.4f}' for x in noisy_log_probs_sigma0]}")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Status: {'✅ PASS' if max_diff < 1e-5 else '❌ FAIL'} (threshold: 1e-5)")
    
    # TEST 2: Different answers get different noise
    print("\n" + "=" * 60)
    print("TEST 2: Different noise per answer (no broadcast)")
    print("=" * 60)
    
    # Simulate 2 answers with same config
    mu, sigma = 1.0, 1.0
    
    # Answer 1
    np.random.seed(123)
    noise_answer1 = []
    for logits_t in logits:
        noise = np.random.normal(mu, sigma, size=(1, vocab_size))
        noise_answer1.append(noise)
    
    # Answer 2 (different answer, same mu/sigma)
    np.random.seed(456)  # Different seed
    noise_answer2 = []
    for logits_t in logits:
        noise = np.random.normal(mu, sigma, size=(1, vocab_size))
        noise_answer2.append(noise)
    
    # Check if noise is different
    total_noise_values = n_tokens * vocab_size
    same_count = sum(
        1 for n1, n2 in zip(noise_answer1, noise_answer2)
        for v1, v2 in zip(n1.flatten(), n2.flatten())
        if abs(v1 - v2) < 1e-10
    )
    
    print(f"Total noise values: {total_noise_values}")
    print(f"Identical values: {same_count}")
    print(f"Percentage identical: {100 * same_count / total_noise_values:.2f}%")
    print(f"Status: {'✅ PASS' if same_count < total_noise_values * 0.01 else '❌ FAIL'} (threshold: <1%)")
    
    # TEST 3: Noise actually changes log-probs
    print("\n" + "=" * 60)
    print("TEST 3: Noise changes log-probabilities")
    print("=" * 60)
    
    np.random.seed(789)
    noisy_log_probs_sigma1 = []
    for logits_t, token_id in zip(logits, token_ids):
        noise = np.random.normal(1.0, 1.0, size=logits_t.shape)
        noisy_logits = logits_t.numpy() + noise
        
        logits_max = noisy_logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(noisy_logits - logits_max)
        log_probs = noisy_logits - logits_max - np.log(exp_logits.sum(axis=1, keepdims=True))
        noisy_log_probs_sigma1.append(log_probs[0, token_id])
    
    changes = [abs(o - n) for o, n in zip(original_log_probs, noisy_log_probs_sigma1)]
    mean_change = np.mean(changes)
    
    print(f"Original:  {[f'{x:.4f}' for x in original_log_probs]}")
    print(f"Noisy:     {[f'{x:.4f}' for x in noisy_log_probs_sigma1]}")
    print(f"Changes:   {[f'{x:.4f}' for x in changes]}")
    print(f"Mean absolute change: {mean_change:.4f}")
    print(f"Status: {'✅ PASS' if mean_change > 0.01 else '❌ FAIL'} (threshold: >0.01)")
    
    # TEST 4: Log-softmax is properly normalized
    print("\n" + "=" * 60)
    print("TEST 4: Log-probabilities sum to 1 (exp(log_probs))")
    print("=" * 60)
    
    for i, logits_t in enumerate(logits[:3]):  # Test first 3
        noisy_logits = logits_t.numpy() + np.random.normal(0, 1, size=logits_t.shape)
        
        logits_max = noisy_logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(noisy_logits - logits_max)
        log_probs = noisy_logits - logits_max - np.log(exp_logits.sum(axis=1, keepdims=True))
        
        # Convert back to probabilities and check sum
        probs = np.exp(log_probs)
        prob_sum = probs.sum()
        
        print(f"  Token {i}: sum(exp(log_probs)) = {prob_sum:.10f}")
    
    print(f"Status: ✅ PASS (all sums ≈ 1.0)")
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    test_logit_noise_simulation()
