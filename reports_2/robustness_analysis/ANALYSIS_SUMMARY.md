# Semantic Entropy Robustness Analysis - Complete Summary

## Project Overview
Comprehensive scientific analysis of semantic entropy (SE) stability under Gaussian logit-space noise perturbations using 400 TriviaQA questions evaluated on Falcon-7B-Instruct model.

---

## Dataset Specifications
- **Total Records**: 4,800 (400 questions × 12 configurations)
- **Noise Configurations**: 12 (μ, σ) pairs
  - Means (μ): [0.0, 0.5, 1.0, 2.0]
  - Standard Deviations (σ): [0.5, 1.0, 2.0]
- **Samples per Configuration**: 100 noise perturbations
- **Model**: Falcon-7B-Instruct (4-bit quantized)
- **Dataset**: TriviaQA validation split

---

## Key Findings

### 1. **Stability Assessment**
- **Global Mean |ΔSE|**: 0.049804
- **Range**: 0.007875 to 0.111977 (13.22x increase)
- **Monotonic Relationship**: SE change increases proportionally with σ

| Noise Level | Mean |ΔSE| |
|---|---|
| σ = 0.5 | 0.008046 |
| σ = 1.0 | 0.029788 |
| σ = 2.0 | 0.111579 |

**Interpretation**: Small noise (σ ≤ 0.5) causes negligible changes; large noise (σ ≥ 2.0) causes substantial but bounded changes.

### 2. **Bias Structure**
- **Mean |Bias|**: 0.046507
- **Bias Direction**: 100% negative across ALL configurations
- **Effect**: Noise systematically REDUCES semantic entropy by compressing logit distributions

This is NOT random noise but a **structured, directional effect** of logit-space compression on probability distributions.

### 3. **Rank Preservation (CRITICAL)**
- **Mean Spearman ρ**: 0.996793
- **Minimum Spearman ρ**: 0.993348
- **All Configurations**: ρ > 0.99 (EXCELLENT)

**Critical Finding**: Despite absolute entropy changes, the relative ordering of uncertainties is **extremely robust**. Semantic entropy can reliably rank questions by confidence even under substantial perturbation.

### 4. **Statistical Significance**
- **Paired t-test**: All 12 configurations show p < 0.001 (****)
- **Effect Size Distribution**:
  - Negligible: 12 configurations
  - Small: 0 configurations
  - Medium: 0 configurations
  - Large: 0 configurations
- **Cohen's d Mean**: 0.0356 (negligible but statistically significant)

All perturbations produce **statistically significant but practically small** changes.

### 5. **Question Sensitivity Analysis**
- **High Sensitivity Cases**: 160 (top 10%)
- **Most Sensitive**: qw_5145--123/123_1138870.txt#0_1 (variance = 0.051046)
- **Sensitivity Threshold (90th %ile)**: 0.016988

Certain questions exhibit unusual sensitivity—likely due to ambiguous answer distributions or logits near critical thresholds.

---

## Generated Outputs

### 📊 Data Files
1. **robustness_metrics.csv** (12 rows)
   - Core metrics per (μ, σ): Mean |ΔSE|, Pearson r, Spearman ρ, Bias, Variance, RMSE, SE_std

2. **statistical_tests.csv** (12 rows)
   - t-test, Wilcoxon, Cohen's d, effect size, significance codes

3. **per_question_sensitivity.csv** (1,600 rows)
   - Question-level sensitivity analysis: variance, mean |ΔSE|, max |ΔSE|

4. **summary_statistics.csv**
   - Key aggregated metrics

### 📈 Visualizations (34 plots)

#### Scatter Plots (12 plots)
- SE_before vs SE_mean with regression lines
- One per (μ, σ) pair
- Directory: `plots/scatter_plots/`

#### Distribution Analysis (12 plots)
- Histograms of SE_before, SE_mean, ΔSE
- Q-Q plots for normality assessment
- One 4-panel set per (μ, σ) pair
- Directory: `plots/distributions/`

#### Stability Analysis (3 plots)
- Mean |ΔSE| vs σ (curves for each μ)
- SE_std vs σ (variance across noise samples)
- Bias vs σ (structured negative bias)
- Directory: `plots/stability/`

#### Heatmaps (4 plots)
- Mean |ΔSE| over (μ, σ) grid
- Mean SE_std over (μ, σ) grid
- Bias over (μ, σ) grid
- Pearson correlation over (μ, σ) grid
- Directory: `plots/heatmaps/`

#### Rank Preservation (2 plots)
- Spearman correlation vs σ
- Pearson vs Spearman comparison
- Directory: `plots/rank_preservation/`

#### Summary Dashboard (1 plot)
- 9-panel comprehensive overview
- File: `plots/00_SUMMARY_DASHBOARD.png`

### 📄 Written Analysis
- **ROBUSTNESS_ANALYSIS_REPORT.txt**: Comprehensive scientific report with interpretations

---

## Scientific Conclusions

### ✅ What We Know
1. **Semantic entropy is ROBUSTLY RANKED-PRESERVING** (ρ > 0.99)
2. **Noise introduces SYSTEMATIC NEGATIVE BIAS** (all configs: bias < 0)
3. **Effect is MONOTONIC WITH NOISE LEVEL** (σ determines magnitude)
4. **Changes are STATISTICALLY SIGNIFICANT** but practically small
5. **High-sensitivity questions EXIST** and can be identified

### ⚠️ Practical Implications
- ✅ **Use for Ranking**: Semantic entropy is excellent for ranking questions by confidence
- ⚠️ **Absolute Values**: Account for systematic bias when interpreting actual entropy values
- ✅ **Active Learning**: Suitable for confidence-based question selection
- ✅ **Small Noise Tolerance**: σ ≤ 0.5 causes negligible effects
- ⚠️ **Large Noise Caution**: σ ≥ 2.0 requires careful interpretation

### 🔬 Limitations
1. **Single Model**: Results specific to Falcon-7B-Instruct
2. **Single Dataset**: TriviaQA only—generalization unknown
3. **Synthetic Noise**: May not reflect realistic perturbations
4. **No Ensemble**: Single-attempt uncertainty only

---

## Recommendations for Future Work

1. **Extended Evaluation**
   - Test on other LLM architectures
   - Evaluate on diverse QA datasets (SQuAD, HotpotQA, Natural Questions)
   - Investigate alternative noise distributions

2. **Mechanistic Analysis**
   - Study logit properties of high-sensitivity questions
   - Develop theoretical bounds on entropy perturbation
   - Analyze answer candidate competition under noise

3. **Practical Applications**
   - Design confidence-calibrated decision rules
   - Develop uncertainty intervals for SE under estimated noise
   - Create robust aggregation methods for noisy conditions

---

## Files Location
```
reports_2/robustness_analysis/
├── ROBUSTNESS_ANALYSIS_REPORT.txt      [Comprehensive analysis]
├── ANALYSIS_SUMMARY.md                 [This file]
├── robustness_metrics.csv              [Core metrics by config]
├── statistical_tests.csv               [Statistical test results]
├── per_question_sensitivity.csv        [Question-level analysis]
├── summary_statistics.csv              [Key aggregates]
└── plots/
    ├── 00_SUMMARY_DASHBOARD.png        [9-panel overview]
    ├── scatter_plots/                  [12 SE_before vs SE_mean plots]
    ├── distributions/                  [12 sets of distribution plots]
    ├── stability/                      [3 stability analysis plots]
    ├── heatmaps/                       [4 configuration heatmaps]
    └── rank_preservation/              [2 correlation analysis plots]
```

---

## Analysis Execution Details

- **Analysis Date**: February 4, 2026
- **Total Execution Time**: ~20 minutes
- **Python Version**: 3.13.7
- **Key Libraries**: pandas, numpy, scipy, matplotlib, seaborn, scikit-learn
- **Notebook**: `semantic_entropy_robustness_analysis.ipynb`

---

## Quick Reference: Configuration Effects

### By Standard Deviation (σ)
| σ | Mean ΔSE | Spearman ρ | Interpretation |
|---|----------|-----------|---|
| 0.5 | 0.008 | 0.9987 | Minimal effect, excellent ranking |
| 1.0 | 0.030 | 0.9980 | Moderate effect, excellent ranking |
| 2.0 | 0.112 | 0.9936 | Substantial effect, excellent ranking |

### By Mean (μ)
All μ values (0.0, 0.5, 1.0, 2.0) produce **equivalent effects**—mean acts as offset, doesn't affect robustness.

---

## Statistical Test Summary

All 12 (μ, σ) configurations show:
- ✅ Paired t-test: p < 0.001 (highly significant)
- ✅ Wilcoxon test: p < 0.001 (non-parametric confirmation)
- ✅ Cohen's d: Negligible (0.009-0.037, practically small)
- ✅ Spearman ρ: 0.993-0.999 (excellent preservation)

**Conclusion**: Changes are real and measurable but small in practical terms, with ordering preserved.

---

## Questions Answered

**Q1: Is semantic entropy stable under noise?**
A: Moderately stable. Absolute values change but follow predictable pattern with σ.

**Q2: Does noise bias semantic entropy?**
A: Yes, systematically toward lower values due to logit compression.

**Q3: Can semantic entropy rank questions reliably?**
A: Yes, exceptionally well (ρ > 0.99). This is the key strength.

**Q4: Which questions are problematic?**
A: 160 high-sensitivity cases identified. See per_question_sensitivity.csv.

**Q5: When is noise impact negligible?**
A: σ ≤ 0.5 causes minimal changes (ΔSE ~ 0.008).

**Q6: What's the main limitation?**
A: Results specific to one model and dataset. Generalization needs testing.

---

*Report generated by comprehensive scientific analysis pipeline*  
*For detailed findings, see ROBUSTNESS_ANALYSIS_REPORT.txt*
