"""
Create a CSV file with summary statistics for each top 10 question
for easy clustering and analysis in Excel/Python
"""
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Load data
jsonl_path = Path('../results/se_after_noise_complete.jsonl')
generations_path = Path('../results/validation_generations.pkl')

print("Loading data...")
with open(jsonl_path, 'r') as f:
    se_data = {json.loads(line)['question_id']: json.loads(line) for line in f}

with open(generations_path, 'rb') as f:
    generations = pickle.load(f)

# Find top 10 by SE_before
sorted_questions = sorted(se_data.items(), key=lambda x: x[1]['se_before'], reverse=True)
top_10 = sorted_questions[:10]

# Create data list
rows = []

for rank, (qid, se_info) in enumerate(top_10, 1):
    if qid in generations:
        gen_data = generations[qid]
        responses = gen_data.get('responses', [])
        
        # Calculate probability statistics across 10 answers
        overall_lls = []
        token_counts = []
        mean_token_lls = []
        
        for response in responses:
            if isinstance(response, tuple) and len(response) >= 2:
                token_loglikelihoods = response[1] if len(response) > 1 else []
                
                overall_ll = sum(token_loglikelihoods) if token_loglikelihoods else 0.0
                overall_lls.append(overall_ll)
                token_counts.append(len(token_loglikelihoods))
                
                if token_loglikelihoods:
                    mean_token_lls.append(np.mean(token_loglikelihoods))
                else:
                    mean_token_lls.append(0.0)
        
        # Calculate statistics
        if overall_lls:
            prob_values = [np.exp(ll) for ll in overall_lls]
            
            row = {
                'Rank': rank,
                'Question_ID': qid,
                'SE_before': se_info['se_before'],
                'SE_mean': se_info['se_mean'],
                'SE_std': se_info['se_std'],
                'Delta_SE': se_info['delta_se'],
                'Correctness': int(se_info.get('correctness', 0)),
                'Num_Answers': len(overall_lls),
                'Avg_OverallLL': np.mean(overall_lls),
                'Min_OverallLL': np.min(overall_lls),
                'Max_OverallLL': np.max(overall_lls),
                'Std_OverallLL': np.std(overall_lls),
                'Avg_Probability': np.mean(prob_values),
                'Min_Probability': np.min(prob_values),
                'Max_Probability': np.max(prob_values),
                'Std_Probability': np.std(prob_values),
                'Avg_TokenCount': np.mean(token_counts),
                'Min_TokenCount': np.min(token_counts),
                'Max_TokenCount': np.max(token_counts),
                'Avg_MeanTokenLL': np.mean(mean_token_lls),
                'Min_MeanTokenLL': np.min(mean_token_lls),
                'Max_MeanTokenLL': np.max(mean_token_lls),
                'Entropy_of_Probs': -np.sum([p * np.log(p + 1e-10) for p in prob_values if p > 0]) / len(prob_values) if prob_values else 0.0,
                'ProbRange': np.max(prob_values) - np.min(prob_values),
            }
            
            rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows)

# Save to CSV
output_dir = Path('../reports_2/robustness_analysis')
csv_path = output_dir / 'TOP_10_UNCERTAINTY_ANALYSIS.csv'
df.to_csv(csv_path, index=False)

print(f"\n✓ CSV file created: {csv_path}")
print(f"\nColumns for cluster analysis:")
print(f"  - SE metrics: SE_before, SE_mean, SE_std, Delta_SE")
print(f"  - Log-likelihood stats: Avg_OverallLL, Min/Max/Std_OverallLL")
print(f"  - Probability stats: Avg_Probability, Min/Max/Std_Probability, ProbRange")
print(f"  - Token stats: Avg_TokenCount, Min/Max_TokenCount")
print(f"  - Entropy: Entropy_of_Probs (entropy of answer probability distribution)")
print(f"\nDataFrame preview:")
print(df.to_string())
