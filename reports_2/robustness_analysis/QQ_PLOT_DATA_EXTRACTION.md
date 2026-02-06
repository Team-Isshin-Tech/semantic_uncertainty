# How Y-Axis and X-Axis are Obtained from JSONL File

## Overview
The data flows from JSONL file → Python DataFrame → Statistical Computation → Q-Q Plot

---

## Step-by-Step Data Flow

### Step 1: Load Data from JSONL File

**File location:** `results/se_after_noise_complete.jsonl`

**Code:**
```python
import pandas as pd
import json

# Load JSONL file (4,800 records)
with open('../results/se_after_noise_complete.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# Convert to DataFrame
df = pd.DataFrame(data)
```

**What the JSONL contains (per record):**
```json
{
  "question_id": "bt_1031--61/61_537330.txt#0_0",
  "mu": 0.5,
  "sigma": 2.0,
  "se_before": 2.253579,
  "se_mean": 2.236359,
  "se_std": 0.021227,
  "delta_se": -0.017220,
  "correctness": 0.0
}
```

**Resulting DataFrame shape:** 4,800 rows × 6 columns

---

### Step 2: Extract Data for One Configuration (e.g., μ=0.5, σ=2.0)

**Code (from the notebook):**
```python
for (mu, sigma), group in df.groupby(['mu', 'sigma']):
    # group now contains 400 rows (one per question)
    # Each row has: delta_se values from the JSONL
```

**For μ=0.5, σ=2.0:**
```
group has 400 rows with delta_se values:
[-0.017220, -0.023402, -0.005666, -0.016567, ..., -0.012345]
         ↑                                             ↑
    1st question                                 400th question
```

---

### Step 3: Obtain Y-Axis Values (Actual/Ordered Values)

**These come DIRECTLY from the JSONL file's delta_se column:**

```python
# Y-axis = Actual delta_se values from JSONL
delta_se_values = group['delta_se'].values
# Example: [-0.0172, -0.0234, -0.0057, -0.0166, ..., -0.0123]

# Sort them (for quantile ordering)
sorted_delta_se = np.sort(delta_se_values)
# Example: [-0.0370, -0.0345, -0.0332, -0.0320, ..., -0.0001]
```

**Source:** JSONL file → `delta_se` column → Sorted

---

### Step 4: Obtain X-Axis Values (Theoretical Quantiles)

**These are COMPUTED mathematically from the sorted data, NOT from the JSONL:**

```python
from scipy import stats

# Code (line 344 from notebook):
stats.probplot(group['delta_se'], dist="norm", plot=ax)
```

**What `scipy.stats.probplot` does internally:**

```python
def generate_theoretical_quantiles(data, dist="norm"):
    """
    For each data point, calculate what value a Normal distribution would predict
    """
    n = len(data)  # 400 points
    
    # Step 1: Calculate percentile positions
    percentiles = np.arange(1, n+1) / (n + 1)
    # For 400 points: [0.0025, 0.005, 0.0075, ..., 0.9975]
    #                  (1/401)  (2/401) (3/401)     (400/401)
    
    # Step 2: Get theoretical Normal values at those percentiles
    theoretical_quantiles = stats.norm.ppf(percentiles)
    # ppf = Percent Point Function (inverse CDF of Normal)
    # Returns: [-2.81, -2.33, -2.05, ..., -0.01, 0, 0.01, ..., 2.05, 2.33, 2.81]
    
    return theoretical_quantiles
```

**Mathematical detail:**
```
For Normal distribution with mean=0, std=1:
  - ppf(0.0025) = -2.81  (2.5th percentile of Normal)
  - ppf(0.50)   = 0.00   (50th percentile = median)
  - ppf(0.975)  = 1.96   (97.5th percentile)
  - ppf(0.9975) = 2.81   (99.75th percentile)

For your data with mean μ and std σ:
  theoretical_value = μ + σ * ppf(percentile)
```

---

## Complete Tracing Example

### Given (μ=0.5, σ=2.0) configuration with 400 questions:

**Y-Axis (from JSONL delta_se values, sorted):**
```
Original from JSONL:  [-0.017, -0.023, -0.006, -0.016, ..., -0.012]
After sorting:        [-0.037, -0.034, -0.033, -0.032, ..., -0.0001]
Index in sorted:       [1st,   2nd,   3rd,   4th,  ..., 400th]
```

**X-Axis (computed from sorted percentiles):**
```python
# For 400 points, percentiles are:
percentiles = [1/401, 2/401, 3/401, ..., 400/401]
            = [0.0025, 0.005, 0.0075, ..., 0.9975]

# Get Normal distribution inverse CDF values:
from scipy.stats import norm
theoretical = norm.ppf(percentiles)
            = [-2.81, -2.33, -2.05, -1.88, ..., 2.81]
```

**The plot then shows:**
```
Point 1:   (x=-2.81,  y=-0.037)    ← 1st sorted ΔSE vs 0.25th percentile Normal
Point 2:   (x=-2.33,  y=-0.034)    ← 2nd sorted ΔSE vs 0.5th percentile Normal
Point 3:   (x=-2.05,  y=-0.033)
...
Point 200: (x=0.0,    y=-0.0015)   ← 200th sorted ΔSE vs 50th percentile (median)
...
Point 400: (x=2.81,   y=-0.0001)   ← 400th sorted ΔSE vs 99.75th percentile Normal
```

---

## Summary: Data Sources

| Axis | Source | How Obtained | From File? |
|------|--------|------------|-----------|
| **Y-axis** | `delta_se` column in JSONL | Read from file → Sort | ✅ **YES, directly from JSONL** |
| **X-axis** | Theoretical Normal quantiles | Compute using `scipy.stats.norm.ppf()` | ❌ **NO, computed mathematically** |

### The JSONL Provides:
- ✓ All 400 delta_se values per configuration
- ✓ mu and sigma parameters
- ✓ se_before and se_mean for validation

### The Code Computes:
- Percentile positions (1/401, 2/401, ..., 400/401)
- Theoretical quantiles from Standard Normal CDF inverse
- Plot coordinates (theoretical, actual)

---

## Why This Method?

The Q-Q plot compares your actual data distribution against a **reference distribution** (Normal).

To make this comparison:
1. **Your data** (Y-axis) shows what you actually measured
2. **Reference distribution** (X-axis) shows what Normal predicts
3. If they match → straight line
4. If they diverge → reveals distribution shape differences

The theoretical quantiles are **never in the JSONL** because they're only meaningful **after you decide what reference distribution to use**. Here, we chose Normal distribution, so we compute its quantiles.

---

## Code Location in Notebook

**File:** `semantic_entropy_robustness_analysis.ipynb`  
**Section:** "Section 5: Generate Distribution Analysis Plots"  
**Lines:** 300-345

```python
# Load JSONL → df (lines 75-85)
with open('../results/se_after_noise_complete.jsonl', 'r') as f:
    records = [json.loads(line) for line in f]
df = pd.DataFrame(records)

# For each (mu, sigma) grouping:
for (mu, sigma), group in df.groupby(['mu', 'sigma']):  # line 302
    # group contains 400 rows with delta_se values from JSONL
    
    # Create Q-Q plot:
    stats.probplot(group['delta_se'], dist="norm", plot=ax)  # line 344
    # This automatically:
    # 1. Extracts 'delta_se' values (Y-axis)
    # 2. Computes theoretical quantiles (X-axis)
    # 3. Plots them as scatter points
```

---

## Key Insight

**Y-axis (Actual):** 100% from JSONL file  
**X-axis (Theoretical):** 0% from JSONL file, 100% computed mathematically

The JSONL file provides only the **raw measurements** (delta_se values). The **statistical framework** (Normal distribution percentiles) is applied **by the code**, not stored in data.
