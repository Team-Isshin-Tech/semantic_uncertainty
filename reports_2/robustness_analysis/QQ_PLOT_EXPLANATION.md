# Q-Q Plot: A Comprehensive Guide

## Quick Answer

A **Q-Q plot** (Quantile-Quantile plot) compares the **actual distribution of your data** against a **theoretical distribution** (usually normal/Gaussian). It visually shows whether your data follows an expected distribution.

---

## Core Concepts

### 1. What is a Quantile?

A **quantile** is a value that divides data into equal-sized groups.

**Examples:**
- **Median** = 0.5 quantile (divides data in half)
- **Quartiles** = 0.25, 0.50, 0.75 quantiles (divides into 4 equal parts)
- **Percentiles** = divide into 100 equal parts
  - 25th percentile = 0.25 quantile
  - 90th percentile = 0.90 quantile

**For 10 sorted data points [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:**
```
Quantile 0.0  (min):     1
Quantile 0.25 (25%):     3.25
Quantile 0.50 (median):  5.5
Quantile 0.75 (75%):     7.75
Quantile 1.0  (max):     10
```

### 2. Theoretical Quantile vs Ordered Values

#### **Ordered Values (Empirical Quantiles)**
These are your actual data points, sorted from smallest to largest.

**Example:** Your semantic entropy measurements sorted
```
SE values (ordered): [0.012, 0.045, 0.089, 0.156, 0.234, 0.456, 0.678, 0.912, 1.234, 2.113]
Position in plot:   [1st,  2nd,  3rd,  4th,  5th,  6th,  7th,  8th,  9th,  10th]
```

#### **Theoretical Quantiles**
These are the expected values from a reference distribution (usually Normal distribution) if your data truly came from that distribution.

**Example:** If SE follows Normal(μ=0.5, σ=0.4)
```
Theoretical quantiles: [-0.87, -0.52, -0.28, -0.10, 0.04, 0.96, 1.48, 2.10, 2.85, 3.52]
(These are from the theoretical Normal distribution)
```

---

## How Q-Q Plot Works

### Step 1: Sort Your Data
```
Original SE data: [0.678, 0.045, 2.113, 0.089, ...]
Sorted (ordered): [0.012, 0.045, 0.089, 0.156, 0.234, ...]
```

### Step 2: Calculate Percentile Positions
For each data point, calculate what **percentile** it represents:
```
1st of 4800 points  = 0.0002 percentile (0.02%)
2nd of 4800 points  = 0.0004 percentile (0.04%)
...
2400th of 4800 pts  = 0.50 percentile (50%)
...
4800th of 4800 pts  = 1.0 percentile (100%)
```

### Step 3: Get Theoretical Values
For each percentile, find what value a **Normal distribution** would predict:
```
For 0.02 percentile (Normal):  -2.81
For 0.04 percentile (Normal):  -2.05
For 50 percentile (Normal):     0.00 (mean)
For 99.98 percentile (Normal):  2.81
```

### Step 4: Create the Plot
Plot pairs of (theoretical quantile, ordered value):

```
     Actual SE value
            ↑
         2.5 |     ●
             |   ●
             | ●
           1 |●
             |○
           0 |───●───── Theoretical Normal value
             |  ○ 
        -0.5 |○
             |
        -1.5 |_____________→
            -2 -1  0  1  2  3
```

- **●** = data points
- **○** = reference line (perfect fit)
- **X-axis** = theoretical quantiles (Normal distribution)
- **Y-axis** = ordered values (your actual data)

---

## What the Plot Reveals

### Perfect Normal Distribution
```
        Actual ↑
            3 |        ●
              |      ●
              |    ●
            0 |  ●
              | ●
           -3 |●_____________
              -3  0  3 Theoretical
```
**Interpretation:** Points form a **straight diagonal line** → Data is normally distributed ✓

### Right-Skewed Distribution (long tail on right)
```
        Actual ↑
            3 |            ●
              |          ●
              |        ●
            0 |  ●●●●●●
              | ●
           -3 |●________________
              -3  0  3 Theoretical
```
**Interpretation:** Points curve **upward** at right end → Data has right tail ✗

### Left-Skewed Distribution
```
        Actual ↑
            3 |●              
              | ●
              |   ●
            0 |●●●●●●
              |        ●
           -3 |________________●
              -3  0  3 Theoretical
```
**Interpretation:** Points curve **downward** at right end → Data has left tail ✗

### Heavy-Tailed Distribution (outliers)
```
        Actual ↑
            5 |            ●  
              |              ●
              |          ●
            0 |    ●●●●
              |  ●
           -3 |●________________
              -3  0  3 Theoretical
```
**Interpretation:** Points deviate at **both ends** → Data has more outliers than Normal ✗

### Light-Tailed Distribution (fewer outliers)
```
        Actual ↑
            2 |            ●
              |          ●
              |        ●
            0 |      ●●●
              |    ●
           -2 |  ●______________
              -3  0  3 Theoretical
```
**Interpretation:** Points stay **closer to center** → Data is more concentrated ✗

---

## Real Example: Your SE Data

From your analysis, the Q-Q plots showed SE distributions across different noise levels.

**What could the plots tell:**

1. **σ=0.5 (small noise):** Points close to diagonal
   - SE is approximately normally distributed
   - Small perturbations don't change distribution shape

2. **σ=2.0 (large noise):** Points deviate at extremes
   - Extreme SE values show deviations
   - Distribution becomes heavier-tailed under extreme noise

3. **Comparison:** If Q-Q plots for σ=0.0, σ=0.5, σ=1.0, σ=2.0 look similar
   - SE distribution is **robust** to noise
   - Validates that noise doesn't change fundamental shape

---

## Why Plot It? (Instead of Just Computing Numbers)

| Approach | Advantage |
|----------|-----------|
| **Normality test** (Shapiro-Wilk) | Single p-value | Misses where/why deviation |
| **Histogram** | Visual distribution | Depends on bin width choice |
| **Q-Q Plot** | **Shows EXACTLY where deviations occur** | Perfect for identifying problems |

**Q-Q Plot advantage:** You can see:
- ✓ If deviation is at tails or center
- ✓ If deviation is symmetric or one-sided
- ✓ How severe the deviation is
- ✓ Whether it matters for your application

---

## Interpretation Guidelines

### Points On the Line
```
y ≈ x  (or y ≈ 1.5x + 0.3)
```
→ Data follows the theoretical distribution ✓

### Points Above the Line
→ Actual values are **larger** than theoretical expects → **Right skew** or **heavy tail**

### Points Below the Line  
→ Actual values are **smaller** than theoretical expects → **Left skew** or **light tail**

### Curved Pattern (S-shape)
→ Distribution has **different tails** than Normal → Might be exponential or other

### Step-like Pattern
→ Data is **discrete** → Not continuous distribution

---

## Application to Your Semantic Entropy Analysis

### What Your Q-Q Plots Showed (12 plots for 12 configs):

**Purpose:** Verify that SE values under each (μ, σ) configuration follow similar distributions

**Expected pattern:** 
- ✓ Points mostly on diagonal line → Robust distribution
- ✓ Minor deviations at extremes → Some outliers (normal for real data)
- ✓ Consistent across all 12 configs → Noise doesn't change shape

**If you saw deviations:**
- Extreme points far from line at σ=2.0 → Large noise creates outliers
- Left deviation at all σ → SE distribution skews left consistently

---

## Mathematical Definition

For dataset with $n$ values:

**Sample quantile** (empirical):
$$q_s = \text{sorted}[i] \quad \text{at position } i = (k+1) \times (i-1) \text{ for } k=1,2,...,n$$

**Theoretical quantile**:
$$q_t = F^{-1}(p) \quad \text{where } p = \frac{i}{n+1}$$

For Normal distribution with mean μ and std σ:
$$q_t = \mu + \sigma \times \Phi^{-1}(p)$$

where $\Phi^{-1}$ is the **inverse of the Normal CDF**.

---

## Summary Table

| Feature | What It Is | What It Means |
|---------|-----------|---------------|
| **Quantile** | A threshold value dividing sorted data | Understanding data distribution shape |
| **Ordered values** | Your data sorted from smallest to largest | Y-axis of Q-Q plot |
| **Theoretical quantiles** | Expected values from reference distribution | X-axis of Q-Q plot |
| **Q-Q Plot** | Scatter plot comparing actual vs theoretical | Visual normality test |
| **Diagonal line** | Perfect match to theory | Your data follows the distribution |
| **Curved pattern** | Deviations from theory | Your data deviates (skew, kurtosis, etc.) |

---

## For Your Semantic Entropy Research

**Q-Q plots answered:**
1. ✓ Is SE normally distributed?
2. ✓ Does distribution shape change with noise (μ, σ)?
3. ✓ Where are the problematic outliers?
4. ✓ Is the distribution symmetric or skewed?
5. ✓ Are there more/fewer extreme values than Normal predicts?

This complements the **Spearman ρ analysis** (rank preservation) by showing whether the **shape** of the distribution is preserved, not just the ranking.
