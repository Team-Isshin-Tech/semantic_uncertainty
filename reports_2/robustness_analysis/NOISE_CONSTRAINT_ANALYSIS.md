# Analysis: Log-Probability Range Preservation Under Noise

## Question
During noise perturbation, are log-probabilities constrained to [-∞, 0] and probabilities to [0, 1]?  
Or does noise cause values to violate these ranges?

## Implementation Analysis

### Step-by-Step Noise Application (from compute_uncertainty_measures.py:207-216)

```python
# Line 208-209: Add noise to logits
noise = np.random.normal(mu, sigma, size=(1, vocab_size))
noisy_logits = logits_t.numpy() + noise  # logits can become ANYTHING

# Lines 211-213: Apply log-softmax (numerically stable version)
logits_max = noisy_logits.max(axis=1, keepdims=True)
exp_logits = np.exp(noisy_logits - logits_max)
log_probs = noisy_logits - logits_max - np.log(exp_logits.sum(axis=1, keepdims=True))
```

### Constraint Analysis

#### Stage 1: After adding noise (Line 209)
```
noisy_logits = logits_t + noise
```
**Range**: UNBOUNDED
- Original logits: arbitrary real values ℝ
- Noise: ε ~ N(μ, σ²) → arbitrary real values ℝ  
- Result: CAN be any real number ✗

**Example with noise parameters μ=1.0, σ=2.0:**
- If logit = -10, noise = +5 → noisy_logit = -5 ✓
- If logit = -10, noise = +20 → noisy_logit = +10 ✓ (UNBOUNDED)
- If logit = +5, noise = -10 → noisy_logit = -5 ✓
- If logit = +5, noise = -1000 → noisy_logit = -995 ✓ (UNBOUNDED)

#### Stage 2: After log-softmax (Lines 211-213)

**Mathematical definition of log-softmax:**

For logits $z = [z_1, z_2, ..., z_V]$:

$$\text{log\_softmax}(z_i) = z_i - \log\left(\sum_{j=1}^{V} \exp(z_j)\right)$$

**Constrained probability from softmax:**

$$p_i = \text{softmax}(z_i) = \frac{\exp(z_i)}{\sum_{j=1}^{V} \exp(z_j)} \in [0,1]$$

**Log of probability:**

$$\log(p_i) = \log\left(\frac{\exp(z_i)}{\sum_{j=1}^{V} \exp(z_j)}\right) = z_i - \log\left(\sum_{j=1}^{V} \exp(z_j)\right) \in [-\infty, 0]$$

**Why this constraint ALWAYS holds:**

1. **Lower bound:** Log of any probability p ≤ 1 gives log(p) ≤ 0 ✓
   - For p → 0: log(p) → -∞ ✓
   - For p = 1: log(p) = 0 (maximum) ✓

2. **Upper bound:** No probability exceeds 1
   - Softmax divides by sum of ALL exponentials
   - Any single probability = exp(z_i) / Σ(exp(z)) ≤ 1 ✓

3. **This holds REGARDLESS of logit values:**
   - Even if logits range from -1000 to +1000
   - Even if noise causes logits to range from -10000 to +10000
   - Softmax normalization always constrains probabilities to [0,1]

### Numerical Stability Implementation

The code uses the log-sum-exp trick (lines 211-213) for numerical stability:

```python
logits_max = noisy_logits.max(axis=1, keepdims=True)
exp_logits = np.exp(noisy_logits - logits_max)  # Subtract max first
log_probs = noisy_logits - logits_max - np.log(exp_logits.sum(axis=1, keepdims=True))
```

**Why this works:**
- Subtracting `logits_max` prevents exp overflow/underflow
- `exp(noisy_logits - logits_max)` produces numerically stable values
- Log-sum-exp formula remains mathematically equivalent to log-softmax
- **Result: Still constrained to [-∞, 0]** ✓

### Example Walkthrough

Question: "Who sang Grenade?"  
Vocabulary size: 30,522 tokens

**Before noise:**
```
logits for "Bruno": 3.5, "Mars": 2.1, "Rihanna": 1.8, ..., "dog": -8.2
softmax(3.5):  p("Bruno") = 0.15
log(0.15) = -1.897 ✓ in [-∞, 0]
```

**After adding noise (μ=1.0, σ=2.0):**
```
Noise for "Bruno": +4.2 → noisy_logit = 7.7
Noise for "Mars": -1.5 → noisy_logit = 0.6
Noise for "Rihanna": +0.8 → noisy_logit = 2.6
...
Noise for "dog": -10.1 → noisy_logit = -18.3

New softmax(7.7): p("Bruno") = 0.45
log(0.45) = -0.799 ✓ in [-∞, 0]

New softmax(0.6): p("Mars") = 0.08
log(0.08) = -2.526 ✓ in [-∞, 0]

New softmax(-18.3): p("dog") ≈ 0.00001
log(0.00001) = -11.513 ✓ in [-∞, 0]
```

**Critical observation:** No matter how extreme the noise, softmax **always** produces [0,1], so log always produces [-∞, 0]!

## Answer: **ALWAYS PRESERVED** ✓

### Constraint Status at Each Stage:

| Stage | Value Range | Constraint Maintained? |
|-------|------------|------------------------|
| Original logits | ℝ (unbounded) | N/A |
| After noise | ℝ (unbounded) | N/A |
| After softmax | [0, 1] | ✅ YES |
| Log-probabilities | [-∞, 0] | ✅ YES |

### Mathematical Proof

For any noisy logits $\tilde{z} = z + \epsilon$ where $\epsilon \sim \mathcal{N}(\mu, \sigma^2)$:

$$\tilde{p}_i = \text{softmax}(\tilde{z}_i) = \frac{\exp(\tilde{z}_i)}{\sum_{j=1}^{V} \exp(\tilde{z}_j)}$$

**By definition of softmax:**
- Numerator: $\exp(\tilde{z}_i) > 0$ (exponentials always positive)
- Denominator: $\sum_{j=1}^{V} \exp(\tilde{z}_j) > 0$ (sum of positive numbers)
- Therefore: $0 < \tilde{p}_i < 1$ ✓ (strict inequalities for finite vocabulary)

**For log-probability:**
$$\log(\tilde{p}_i) = \log\left(\frac{\exp(\tilde{z}_i)}{\sum_j \exp(\tilde{z}_j)}\right) = \tilde{z}_i - \log\left(\sum_j \exp(\tilde{z}_j)\right) \leq 0$$

Since denominator ≥ numerator: $\log(\tilde{p}_i) \leq 0$ ✓

## Implementation Verification

The code's sanity check (lines 41-65) actually confirms this:
```python
def sanity_check_logit_noise(logits_data, token_ids_data, log_liks):
    """Sanity checks for logit-space noise implementation"""
    # Check 1: sigma=0 → original log-probs
    # This implicitly verifies that log-probs are computed correctly
```

## Conclusion

**The ranges [0, 1] for probabilities and [-∞, 0] for log-probabilities are MATHEMATICALLY GUARANTEED to be preserved** by the softmax/log-softmax operation, **regardless of how large or small the noise perturbations are**.

The implementation is **correct and safe** because:
1. ✅ Softmax is defined to always produce [0, 1]
2. ✅ Log of [0, 1] is always [-∞, 0]
3. ✅ This holds for ANY logit input values (with or without noise)
4. ✅ Numerical stability is maintained via log-sum-exp trick
5. ✅ No constraints or clamping is needed

The only thing that changes is the **distribution** of these probabilities, not their **range**.
