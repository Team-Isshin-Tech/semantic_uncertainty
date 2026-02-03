# Logit-Space Noise Implementation - Changes Summary

## Overview
Implemented two key changes to the noise perturbation methodology:
1. **Noise in LOGIT space** (instead of log-probability space)
2. **Different noise per answer** (not broadcast across batch)

## Changes Made

### 1. Modified `huggingface_models.py` (Line ~440)

**What changed:**
- `predict()` now returns 5 values instead of 3:
  - `sliced_answer` (text)
  - `log_likelihoods` (list of token log-probs)
  - `last_token_embedding` (tensor)
  - **`logits_per_token`** (NEW: list of logit tensors, each shape [1, vocab_size])
  - **`generated_token_ids`** (NEW: list of generated token IDs)

**Why:**
- Need original logits to add noise in logit space (before softmax)
- Need token IDs to extract correct log-prob after adding noise

### 2. Modified `generate_answers.py` (Line ~290, ~323)

**What changed:**
- Updated `model.predict()` unpacking to handle 5 return values
- Updated `full_responses` tuple to include logits and token IDs:
  ```python
  (predicted_answer, token_log_likelihoods, embedding, logits_per_token, generated_token_ids, acc)
  ```

**Why:**
- Pass logits and token IDs through to compute_uncertainty_measures.py

### 3. Modified `compute_uncertainty_measures.py` (Lines ~312-411)

**What changed:**

#### a) Added sanity check function (Lines ~32-77):
```python
def sanity_check_logit_noise(logits_data, token_ids_data, log_liks):
    """
    Check 1: sigma=0 → noisy log-probs equal original (validates correctness)
    Check 2: Different answers get different noise (validates no broadcast)
    """
```

#### b) Replaced noise perturbation section (Lines ~368-417):

**OLD (log-prob space noise):**
```python
for log_lik_seq in log_liks:
    token_noise = np.random.normal(mu, sigma, len(log_lik_seq))
    noisy_log_lik_seq = [ll + n for ll, n in zip(log_lik_seq, token_noise)]
    noisy_log_liks_agg.append(np.mean(noisy_log_lik_seq))
```

**NEW (logit space noise, different per answer):**
```python
for answer_idx, (logits_seq, token_ids) in enumerate(zip(logits_data, token_ids_data)):
    noisy_token_log_probs = []
    for t, (logits_t, token_id) in enumerate(zip(logits_seq, token_ids)):
        # Add noise in logit space (shape [1, vocab_size] for THIS answer)
        noise = np.random.normal(mu, sigma, size=(1, vocab_size))
        noisy_logits = logits_t.numpy() + noise
        
        # Recompute log_softmax(noisy_logits)
        logits_max = noisy_logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(noisy_logits - logits_max)
        log_probs = noisy_logits - logits_max - np.log(exp_logits.sum(axis=1, keepdims=True))
        
        # Get log-prob of actually generated token
        noisy_ll_t = log_probs[0, token_id]
        noisy_token_log_probs.append(noisy_ll_t)
    
    noisy_log_liks_agg.append(np.mean(noisy_token_log_probs))
```

#### c) Added sanity check call (Line ~379):
```python
if idx == 0:
    sanity_check_logit_noise(logits_data, token_ids_data, log_liks)
```

## Technical Details

### Noise Distribution
- **Equation:** `z_noisy = z_original + ε`, where `ε ~ N(μ, σ²)`
- **Shape:** `[1, vocab_size]` per token per answer (ensures different noise per answer)
- **Application:** Pre-softmax logits, not post-softmax log-probabilities

### Log-Softmax Computation
```python
log_softmax(z) = z - log(sum(exp(z)))
               = z - max(z) - log(sum(exp(z - max(z))))  # numerically stable
```

### Sanity Checks

**Check 1: σ=0 reproduces original**
- Add zero noise → should get original log-probs
- Validates: log_softmax implementation, token ID alignment
- Threshold: `max_diff < 1e-5`

**Check 2: Different answers get different noise**
- Generate noise for answer 0 and answer 1
- Verify they're not identical (not broadcast)
- Validates: noise is independently sampled per answer

## Constraints Satisfied

✅ **Noise equation unchanged:** Still Gaussian(μ, σ)  
✅ **No answer re-generation:** Uses existing `outputs.scores` and `outputs.sequences`  
✅ **Different noise per answer:** Shape `[1, vocab_size]` per token, not broadcast  
✅ **Semantic entropy unchanged:** Only noise perturbation modified  
✅ **Works for any batch size:** Code uses `vocab_size` from tensor shape dynamically  

## Testing

Run on first validation question automatically:
```bash
python compute_uncertainty_measures.py --model <model> --dataset trivia_qa
```

Expected output:
```
Running sanity checks for logit-space noise...
Check 1 - sigma=0 max diff: 0.000001 (should be < 1e-5)
Check 2 - Same noise values (first 100): 0/100 (should be < 50 for different answers)
Sanity checks passed!
```

## Files Modified

1. `semantic_uncertainty/uncertainty/models/huggingface_models.py` (lines ~425-441)
2. `semantic_uncertainty/generate_answers.py` (lines ~290, ~323)
3. `semantic_uncertainty/compute_uncertainty_measures.py` (lines ~32-77, ~368-417)

## Backward Compatibility

⚠️ **Breaking change:** Old `.pkl` files with 4-tuple responses will fail.  
**Solution:** Re-run `generate_answers.py` to create new pickle files with 6-tuple format.

## Next Steps

1. Re-run answer generation for all models/datasets
2. Re-run noise robustness analysis
3. Compare results: logit-space vs log-prob-space noise
4. Update plots and analysis scripts if needed
