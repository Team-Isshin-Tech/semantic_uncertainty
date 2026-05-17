# Two-Stage Noise Analysis Pipeline

## Overview

This pipeline separates model answer generation from noise analysis, enabling flexible parameter tuning without regeneration.

```
┌─────────────────────┐
│  STAGE 1: GENERATE  │ → Generate answers + store logits
│   validate_answers.py │
└─────────────────────┘
         ↓
   (WandB artifact)
   validation_generations.pkl
         ↓
┌─────────────────────┐
│  STAGE 2: ANALYSIS  │ → Load logits + add noise + compute metrics
│ noise_analysis.py   │
└─────────────────────┘
         ↓
   (WandB artifact)
   noise_analysis_results.csv
```

---

## Stage 1: Generate Answers

**File:** `stage1_generate_answers.py`

**Purpose:** Generate multiple answers per question and store per-token logits.

### What it stores:

```python
validation_generations = {
    question_id: {
        'responses': [
            (
                predicted_answer,           # [0] str
                token_log_likelihoods,      # [1] List[np.ndarray]
                embedding,                  # [2] tensor
                logits_per_token,           # [3] List[np.ndarray] ← KEY DATA
                generated_token_ids,        # [4] List[int]
                accuracy                    # [5] float
            ),
            ...  # num_generations total
        ],
        'semantic_ids': {...},              # Cluster assignments
        'question': '...',
        'ground_truth': '...',
    },
    ...  # all questions
}
```

### Key features:

- ✓ Stores full per-token logits (vocabulary probability distribution)
- ✓ Stores semantic cluster assignments
- ✓ Uploads to WandB automatically
- ✓ Returns run ID for Stage 2

### Run command:

```bash
python stage1_generate_answers.py \
    --model_name Mistral-7B-Instruct-v0.3 \
    --dataset trivia_qa \
    --num_samples 400 \
    --num_generations 10 \
    --random_seed 42
```

### Output:

- WandB run with `validation_generations.pkl` artifact
- `generation_metadata.json` with run info

---

## Stage 2: Noise Analysis

**File:** `stage2_noise_analysis.py`

**Purpose:** Load logits from Stage 1 and compute noise robustness metrics.

### What it computes:

For each question and noise configuration `(μ, σ)`:

1. **SE_before**: Semantic entropy without noise
2. **SE_noisy_001 to SE_noisy_100**: Entropy with 100 independent noise samples
3. **SE_mean**: Mean of 100 noisy SE values
4. **SE_std**: Std of 100 noisy SE values
5. **Delta_SE**: `SE_before - SE_mean` (change in uncertainty)
6. **Abs_Delta_SE**: Absolute change

### Output structure:

```csv
question_id,se_before,noise_mu,noise_sigma,se_noisy_001,se_noisy_002,...,se_noisy_100,se_mean,se_std,delta_se,abs_delta_se
q_123,0.456,0.0,0.5,0.412,0.398,...,0.445,0.418,0.012,-0.038,0.038
...
```

### Noise configurations (default):

- **Means (μ):** [0.0, 0.5, 1.0, 2.0]
- **Stds (σ):** [0.5, 1.0, 2.0]
- **Samples per config:** 100
- **Total rows:** num_questions × 12 configs

### Run command:

```bash
# Replace with run ID from Stage 1
python stage2_noise_analysis.py \
    --eval_wandb_runid <RUN_ID_FROM_STAGE_1> \
    --random_seed 42
```

### Output:

- WandB run with `noise_analysis_results.csv` artifact
- 4800 rows (400 questions × 12 configs)

---

## Pipeline Workflow

### Quick Start:

```bash
# 1. Verify everything works
python verify_pipeline.py

# 2. Run Stage 1 (generation ~1-2 hours for 400 Qs with Mistral)
python stage1_generate_answers.py \
    --model_name Mistral-7B-Instruct-v0.3 \
    --dataset trivia_qa \
    --num_samples 400

# 3. Copy the run ID from Stage 1 output
# 4. Run Stage 2 (noise analysis ~30 mins)
python stage2_noise_analysis.py \
    --eval_wandb_runid <STAGE_1_RUN_ID>

# 5. Download results from WandB
```

### For different noise parameters:

If you want to change noise configurations **without regenerating answers**:

1. Edit `NOISE_MEANS` and `NOISE_STDS` in `stage2_noise_analysis.py`
2. Rerun Stage 2 with the same `--eval_wandb_runid`
3. Get new results in seconds instead of hours!

---

## Data Flow

### Stage 1 Output Size:

For 400 questions, 10 answers, Mistral-7B:
- `validation_generations.pkl`: ~3.9 GB
  - Per token: logits (50K floats × 8 bytes) = 400 KB
  - Multiple answers + clustering info increases size

### Stage 2 Output Size:

For 400 questions, 12 noise configs:
- `noise_analysis_results.csv`: ~20 MB
  - 4800 rows
  - 107 columns (100 noisy values + 7 metadata)

---

## Noise Implementation

### How noise is applied:

```python
# For each token:
noisy_logits = original_logits + N(μ, σ)

# Then convert to probabilities:
log_probs = log_softmax(noisy_logits)

# Recompute SE from noisy probabilities
```

### Why this matters:

- ✓ Noise is applied **per-token independently**
- ✓ Different random sample for each of 100 iterations
- ✓ Tests robustness to inference-time perturbations
- ✓ Answers stay the same (only likelihoods change)

---

## Verification Checklist

Run before production:

```bash
python verify_pipeline.py
```

Checks:
- ✓ Noise addition works
- ✓ Log-softmax produces valid probabilities
- ✓ SE computation is numerically stable
- ✓ Output structure is correct
- ✓ No NaN/Inf values

---

## Troubleshooting

### Issue: "Logits not found in validation_generations.pkl"

**Cause:** Generation didn't store logits (old code)  
**Fix:** Use Stage 1 script or regenerate with updated code

### Issue: "semantic_ids NOT FOUND"

**Cause:** Semantic clustering not computed  
**Fix:** Ensure semantic_entropy.py is imported and run

### Issue: WandB artifact too large

**Cause:** Too many questions or answers  
**Fix:** Reduce `--num_samples` or `--num_generations`

### Issue: Negative SE values in output

**Cause:** Floating-point errors in entropy computation  
**Fix:** Clamping is applied (line 246 of semantic_entropy.py)

---

## Integration with Existing Code

### What changed:

- ✓ New: `stage1_generate_answers.py` - focused generation
- ✓ New: `stage2_noise_analysis.py` - focused analysis
- ✓ New: `verify_pipeline.py` - validation script
- ✓ Existing: Original `generate_answers.py` still works
- ✓ Existing: Original `compute_uncertainty_measures.py` still works

### Backward compatibility:

- Old scripts unaffected
- Can run both old and new pipelines
- WandB project contains both

---

## Next Steps

1. Run verification: `python verify_pipeline.py`
2. Test Stage 1 on small sample: `python stage1_generate_answers.py --num_samples 10`
3. Test Stage 2: `python stage2_noise_analysis.py --eval_wandb_runid <TEST_RUN_ID>`
4. If all pass: run full pipeline on 400 samples

---

## Output Interpretation

### Key metrics from results CSV:

- **delta_se > 0:** Noise adds uncertainty (SE increases)
- **delta_se < 0:** Noise reduces uncertainty (SE decreases)
- **delta_se ≈ 0:** Model robust to noise
- **se_std (large):** Noisy results vary significantly
- **se_std (small):** Noisy results stable

### Example analysis:

```python
import pandas as pd

df = pd.read_csv('noise_analysis_results.csv')

# Average change for each noise level
by_config = df.groupby(['noise_mu', 'noise_sigma'])['delta_se'].agg(['mean', 'std', 'count'])
print(by_config)

# Questions most affected by noise
df['vulnerability'] = df['abs_delta_se']
most_vulnerable = df.nlargest(10, 'vulnerability')
print(most_vulnerable[['question_id', 'noise_mu', 'noise_sigma', 'delta_se']])
```

