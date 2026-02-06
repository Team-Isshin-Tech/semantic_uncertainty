# Top 10 Uncertain Questions - Complete Data for Cluster Analysis

## Files Generated

### 1. **TOP_10_UNCERTAIN_WITH_PROBABILITIES.txt** (Detailed View)
- **Purpose:** Detailed human-readable analysis of each question
- **Contents per question:**
  - Question text
  - Ground truth answers
  - 10 generated answers with:
    - Answer text (sequence)
    - Overall log-likelihood (sum of token log-probs)
    - Probability (exp of overall LL)
    - Token count (sequence length)
    - Per-token log-likelihoods [list]
    - Statistical summary: mean, min, max, std of token LLs
  - SE metrics (before, mean, std, delta)
  - Correctness flag

### 2. **TOP_10_UNCERTAINTY_ANALYSIS.csv** (For Clustering)
- **Purpose:** Easy import into Excel/Python for statistical analysis
- **Format:** 10 rows (one per question) × 25 columns

---

## Available Metrics for Cluster Analysis

### Semantic Entropy Metrics
```
- SE_before: Baseline entropy (before noise)
- SE_mean: Average entropy across noise samples
- SE_std: Variance of entropy across noise samples
- Delta_SE: Change in entropy (SE_mean - SE_before)
```

### Log-Likelihood Statistics (across 10 answers)
```
- Avg_OverallLL: Mean of 10 answer log-likelihoods
- Min_OverallLL: Minimum log-likelihood
- Max_OverallLL: Maximum log-likelihood
- Std_OverallLL: Standard deviation of log-likelihoods
```

### Probability Statistics (across 10 answers)
```
- Avg_Probability: Mean probability of 10 answers
- Min_Probability: Lowest probability answer
- Max_Probability: Highest probability answer
- Std_Probability: Spread of probabilities
- ProbRange: (Max - Min) probability
```

### Token-Level Statistics
```
- Avg_TokenCount: Average tokens per answer
- Min_TokenCount: Shortest answer (tokens)
- Max_TokenCount: Longest answer (tokens)
```

### Probability Distribution Properties
```
- Avg_MeanTokenLL: Average of per-token log-likelihoods
- Min_MeanTokenLL: Minimum token-level mean LL
- Max_MeanTokenLL: Maximum token-level mean LL
- Entropy_of_Probs: Entropy of answer probability distribution
```

---

## How to Use for Cluster Analysis

### Option 1: Python/Pandas Analysis
```python
import pandas as pd

df = pd.read_csv('TOP_10_UNCERTAINTY_ANALYSIS.csv')

# Find questions with high probability spread
high_spread = df[df['ProbRange'] > 0.1]

# Correlate SE with probability entropy
corr = df['SE_before'].corr(df['Entropy_of_Probs'])

# Identify tight vs loose probability distributions
tight_dist = df[df['Std_Probability'] < 0.01]  # All answers similar prob
loose_dist = df[df['Std_Probability'] > 0.03]  # Wide probability spread
```

### Option 2: Excel Analysis
- Open CSV in Excel
- Create pivot tables by Correctness
- Plot SE_before vs ProbRange (scatter)
- Identify clustering patterns

### Option 3: Manual Inspection
- Read detailed TXT file
- Look for patterns in answer diversity
- Compare token-level statistics across questions

---

## Key Insights You Can Extract

### 1. **Probability Spread Analysis**
- **High ProbRange** → Model very uncertain, answers have wildly different probabilities
- **Low ProbRange** → Model somewhat confident but wrong, similar-prob wrong guesses
- **Relationship to SE?** Should correlate with SE_before

### 2. **Token-Level Dynamics**
- **Avg_TokenCount vs SE:** Longer answers = higher entropy?
- **Avg_MeanTokenLL vs SE:** Token-level confidence related to overall uncertainty?

### 3. **Probability Distribution Shape**
- **Entropy_of_Probs:** How "spread out" is the model's belief?
  - High entropy = model equally uncertain about all 10 answers
  - Low entropy = one answer dominates, others nearly impossible
- **Relationship to SE:** Should be highly correlated (both measure distribution spread)

### 4. **Outlier Detection**
- Questions with Delta_SE << mean (large negative shift under noise)
- Questions where Max_Probability >> Avg_Probability (one dominant wrong answer)
- Questions with all very low probabilities (all answers nearly impossible)

---

## Example: Rank 1 Question ("Yes Prime Minister")

| Metric | Value | Interpretation |
|--------|-------|---|
| SE_before | 2.254 | Very high uncertainty |
| Delta_SE | -0.292 | Large drop under noise (outlier!) |
| Max_Probability | 0.111 | Best guess only 11% likely |
| Min_Probability | 0.000 | Some answers essentially impossible |
| ProbRange | 0.111 | Very wide spread |
| Entropy_of_Probs | 0.057 | Distribution fairly concentrated |
| Avg_TokenCount | 4.7 | Average answer ~5 tokens |

**Interpretation:**
- Model is VERY uncertain (SE=2.254)
- But most answers are quite short (avg 5 tokens)
- Probability spread is large (0.111 range)
- BUT entropy of distribution is low (0.057) → one answer "David Cameron" dominates (11%)
- Under noise, SE drops significantly (-0.292), suggesting this question is sensitive to logit perturbations

---

## Recommended Cluster Analysis

### Clustering Approach 1: K-Means on Probability Space
```
Features: [SE_before, ProbRange, Entropy_of_Probs, Avg_TokenCount]
K=3: Tight-confident, Spread-uncertain, Mixed-ambiguous
```

### Clustering Approach 2: Hierarchical on Token Statistics
```
Features: [Avg_TokenCount, Avg_MeanTokenLL, Std_OverallLL]
Find groups with similar answer complexity patterns
```

### Clustering Approach 3: SE Sensitivity Analysis
```
Features: [SE_before, Delta_SE, Std_OverallLL]
Group by how much SE drops under noise (low/medium/high sensitivity)
```

---

## Files Location

- Detailed text: `reports_2/robustness_analysis/TOP_10_UNCERTAIN_WITH_PROBABILITIES.txt`
- CSV for analysis: `reports_2/robustness_analysis/TOP_10_UNCERTAINTY_ANALYSIS.csv`

---

## Next Steps

1. **Load CSV** into Python/Excel
2. **Compute correlations** between SE and probability metrics
3. **Identify clusters** by probability distribution shape
4. **Compare with detailed TXT** to find patterns in question properties
5. **Investigate outliers** (e.g., why does Rank 1 have Delta_SE=-0.292 vs mean=-0.15?)
