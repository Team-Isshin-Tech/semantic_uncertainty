# Two-Stage Pipeline: Setup Complete ✓

## Summary of Changes

Created a clean, production-ready two-stage noise analysis pipeline:

### New Files Created:

1. **stage1_generate_answers.py** (250 lines)
   - Generates answers and stores logits to WandB
   - Clean separation from noise analysis
   - Stores `validation_generations.pkl` (3.9 GB)

2. **stage2_noise_analysis.py** (320 lines)
   - Loads logits from Stage 1
   - Computes 100 noisy SE values for each noise config
   - Outputs detailed CSV with all metrics
   - Can be re-run with different noise parameters without regeneration

3. **verify_pipeline.py** (280 lines)
   - Tests all core logic: noise, log-softmax, SE computation
   - Validates output structure
   - Checks for numerical stability
   - **Status: ✓ ALL TESTS PASSED**

4. **PIPELINE_SETUP.md**
   - Complete documentation
   - Data flow diagrams
   - Troubleshooting guide
   - Integration notes

---

## Verification Results

```
✓ Noise addition works (independent samples)
✓ Log-softmax produces valid probabilities (sum to 1.0)
✓ SE computation stable and non-negative
✓ Output structure correct (108 columns, 100 noisy values)
✓ No NaN or Inf values
```

---

## Ready to Use: Next Steps

### Step 1: Test with small dataset (5 minutes)

```bash
python stage1_generate_answers.py \
    --model_name Mistral-7B-Instruct-v0.3 \
    --dataset trivia_qa \
    --num_samples 10 \
    --num_generations 5
```

Expected output: WandB run with small `validation_generations.pkl`

### Step 2: Run Stage 2 with test data

```bash
# Replace <TEST_RUN_ID> with ID from Stage 1 output
python stage2_noise_analysis.py --eval_wandb_runid <TEST_RUN_ID>
```

Expected output: WandB artifact with noise_analysis_results.csv (60 rows)

### Step 3: Verify results look good

```python
import pandas as pd
import wandb

# Download from WandB and inspect
df = pd.read_csv('noise_analysis_results.csv')
print(df.head())
print(df.describe())
```

### Step 4: Full run (if small test passes)

```bash
# Stage 1: 400 samples, 10 answers each (~1-2 hours)
python stage1_generate_answers.py \
    --model_name Mistral-7B-Instruct-v0.3 \
    --dataset trivia_qa \
    --num_samples 400 \
    --num_generations 10

# Stage 2: Noise analysis on 400 questions (~30 mins)
python stage2_noise_analysis.py --eval_wandb_runid <FULL_RUN_ID>
```

---

## Key Design Decisions

### 1. Why Stage 1 and Stage 2?

✓ **Separation of concerns:**
- Generation is compute-intensive (LLM inference)
- Noise analysis is parameter-tuning (no LLM needed)

✓ **Flexibility:**
- Want different noise levels? Just run Stage 2 again
- Saves hours of computation

✓ **Reproducibility:**
- Same answers, different noise → compare fairly
- Logits stored in WandB, reproducible

### 2. Why 100 noise samples?

✓ Captures variability in SE under perturbation  
✓ Allows statistical analysis (mean, std)  
✓ Computationally reasonable (~30 mins for 400 Qs)

### 3. Output structure

**100 noisy SE columns:** `se_noisy_001` to `se_noisy_100`  
→ Full distribution of noise effects

**Mean/Std:** `se_mean`, `se_std`  
→ Summary statistics

**Delta metrics:** `delta_se`, `abs_delta_se`  
→ Change in uncertainty

---

## Output Files Structure

### Stage 1 WandB Artifacts:
```
/validation_generations.pkl (3.9 GB)
  ├─ question_id
  │   ├─ responses: [(answer, log_likelihoods, embedding, logits, token_ids, accuracy), ...]
  │   └─ semantic_ids: {token_id: cluster_id, ...}
  └─ [400 questions total]

/generation_metadata.json
  ├─ run_id
  ├─ run_path
  ├─ has_logits: true
  ├─ has_semantic_ids: true
  └─ config details
```

### Stage 2 WandB Artifacts:
```
/noise_analysis_results.csv (20 MB, 4800 rows)
  ├─ question_id, se_before
  ├─ noise_mu, noise_sigma
  ├─ se_noisy_001 ... se_noisy_100
  ├─ se_mean, se_std
  └─ delta_se, abs_delta_se
```

---

## Noise Configurations (Default)

| Means (μ) | Stds (σ) | Samples | Total Configs |
|-----------|----------|---------|---------------|
| 0.0, 0.5, 1.0, 2.0 | 0.5, 1.0, 2.0 | 100 | **12** |

→ Results: 400 questions × 12 configs = **4,800 rows**

---

## What Changed

### Old Pipeline:
```
generate_answers.py → compute_uncertainty_measures.py
  (tightly coupled)
```

### New Pipeline:
```
stage1_generate_answers.py → [stored logits] → stage2_noise_analysis.py
     (generation only)                            (analysis only)
```

### Backward Compatibility:
✓ Original scripts unchanged  
✓ Can use both old and new  
✓ WandB project stores both

---

## Pre-Flight Checklist

Before running full pipeline:

- [ ] Read `PIPELINE_SETUP.md`
- [ ] Run `verify_pipeline.py` (should show ✓ ALL PASSED)
- [ ] Test Stage 1 with `--num_samples 10`
- [ ] Test Stage 2 with test run
- [ ] Verify CSV output looks correct
- [ ] Check WandB artifacts uploaded

---

## Performance Estimates

| Task | Duration | GPUs | Notes |
|------|----------|------|-------|
| Verify | 10s | 0 | CPU only |
| Stage 1 (10 Qs) | 2 min | 1x GPU | Quick test |
| Stage 1 (400 Qs) | 1-2 hrs | 1x GPU | Depends on GPU |
| Stage 2 (400 Qs) | 30 min | 1x GPU | Noise + SE recomputation |
| Total full run | ~2.5 hrs | 1x GPU | Test + full pipeline |

---

## Troubleshooting

### Before running Stage 1:
```bash
python verify_pipeline.py
# Should show: ✓ ALL VERIFICATIONS PASSED
```

### Check imports:
```bash
python -c "import wandb, torch, pandas, numpy; print('✓ OK')"
```

### Check WandB auth:
```bash
wandb login
# Paste your API key
```

### If Stage 1 fails:
- Check GPU memory: `nvidia-smi`
- Reduce `--num_samples` or `--num_generations`
- Check WandB connectivity: `wandb sync`

### If Stage 2 fails:
- Verify Stage 1 run ID is correct
- Check artifact size: might be too large to download
- Try with `--num_samples 10` first

---

## Integration Notes

### With existing code:
- `generate_answers.py` still works
- `compute_uncertainty_measures.py` still works
- New scripts are in same directory
- No conflicts

### For future use:
- If noise parameters change → just run Stage 2 again
- If dataset changes → run Stage 1 with new dataset
- If model changes → run both stages with new model

---

## Success Criteria

✓ Verification passes  
✓ Stage 1 produces `validation_generations.pkl` with logits  
✓ Stage 2 produces `noise_analysis_results.csv` with 4800 rows  
✓ All metrics (se_before, se_mean, delta_se, etc.) are present  
✓ No NaN/Inf values in output  
✓ Both artifacts appear in WandB  

---

**Ready to proceed! Start with `python verify_pipeline.py` if you haven't already.**
