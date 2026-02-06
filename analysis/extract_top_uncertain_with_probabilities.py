"""
Extract top 10 uncertain questions with complete probability and sequence information
for cluster analysis and SE value investigation
"""
import pickle
import json
import numpy as np
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
print("Finding top 10 most uncertain questions...")
sorted_questions = sorted(se_data.items(), key=lambda x: x[1]['se_before'], reverse=True)
top_10 = sorted_questions[:10]

# Output file
output_dir = Path('../reports_2/robustness_analysis')
output_path = output_dir / 'TOP_10_UNCERTAIN_WITH_PROBABILITIES.txt'

with open(output_path, 'w', encoding='utf-8') as out:
    out.write("="*120 + "\n")
    out.write("TOP 10 MOST UNCERTAIN QUESTIONS - WITH TOKEN PROBABILITIES & SEQUENCES\n")
    out.write("(Ranked by SE_before - Baseline Semantic Entropy)\n")
    out.write("="*120 + "\n\n")
    
    for rank, (qid, se_info) in enumerate(top_10, 1):
        out.write("\n" + "="*120 + "\n")
        out.write(f"RANK {rank}: SE_before = {se_info['se_before']:.6f}\n")
        out.write("="*120 + "\n")
        out.write(f"Question ID: {qid}\n\n")
        
        if qid in generations:
            gen_data = generations[qid]
            
            # Extract question
            question_text = gen_data.get('question', 'N/A')
            out.write(f"QUESTION:\n  {question_text}\n\n")
            
            # Extract ground truth
            reference = gen_data.get('reference', {})
            if isinstance(reference, dict):
                ground_truth = reference.get('answers', {})
                if isinstance(ground_truth, dict):
                    answers = ground_truth.get('text', [])
                else:
                    answers = []
            else:
                answers = []
            
            out.write(f"GROUND TRUTH ANSWERS:\n")
            if answers:
                for ans in answers:
                    out.write(f"  ✓ {ans}\n")
            else:
                out.write(f"  (None)\n")
            
            # Extract 10 generated responses with probabilities
            responses = gen_data.get('responses', [])
            
            out.write(f"\n{'─'*120}\n")
            out.write(f"10 GENERATED ANSWERS (with token log-likelihoods)\n")
            out.write(f"{'─'*120}\n\n")
            
            if responses and len(responses) > 0:
                for i, response in enumerate(responses, 1):
                    # response = (answer_text, token_loglikelihoods, embedding, token_ids, token_indices, overall_ll)
                    
                    if isinstance(response, tuple) and len(response) >= 2:
                        answer_text = response[0]
                        token_loglikelihoods = response[1] if len(response) > 1 else []
                        
                        # Calculate overall log-likelihood (sum of token log-probs)
                        overall_ll = sum(token_loglikelihoods) if token_loglikelihoods else 0.0
                        
                        # Convert log-likelihood to probability
                        prob = np.exp(overall_ll)
                        
                        out.write(f"Answer {i:2d}:\n")
                        out.write(f"  Text: {answer_text}\n")
                        out.write(f"  Overall Log-Likelihood: {overall_ll:.6f}\n")
                        out.write(f"  Probability: {prob:.6f}\n")
                        out.write(f"  Token Count: {len(token_loglikelihoods)}\n")
                        
                        if token_loglikelihoods:
                            out.write(f"  Per-Token Log-Likelihoods: {[f'{ll:.4f}' for ll in token_loglikelihoods]}\n")
                            out.write(f"  Token Log-LL Stats:\n")
                            out.write(f"    Mean: {np.mean(token_loglikelihoods):.6f}\n")
                            out.write(f"    Min: {np.min(token_loglikelihoods):.6f}\n")
                            out.write(f"    Max: {np.max(token_loglikelihoods):.6f}\n")
                            out.write(f"    Std: {np.std(token_loglikelihoods):.6f}\n")
                        
                        out.write("\n")
            else:
                out.write("[No response data available]\n\n")
            
            # Extract SE metrics
            out.write(f"{'─'*120}\n")
            out.write(f"SEMANTIC ENTROPY METRICS:\n")
            out.write(f"{'─'*120}\n")
            out.write(f"  SE before noise:          {se_info['se_before']:.6f}\n")
            out.write(f"  SE mean after noise:      {se_info['se_mean']:.6f}\n")
            out.write(f"  SE std (across samples):  {se_info['se_std']:.6f}\n")
            out.write(f"  Delta SE (mean - before): {se_info['delta_se']:.6f}\n")
            out.write(f"  Model correctness:        {'✓ CORRECT' if se_info.get('correctness') else '✗ INCORRECT'}\n")
            
            # Most likely answer
            most_likely_dict = gen_data.get('most_likely_answer', {})
            if isinstance(most_likely_dict, dict):
                most_likely = most_likely_dict.get('response', 'N/A')
            else:
                most_likely = str(most_likely_dict)
            out.write(f"  Most likely answer:       {most_likely}\n")

print(f"\n✓ Output written to: {output_path}")
print(f"\nFile contains:")
print(f"  - Question text")
print(f"  - Ground truth answers")
print(f"  - 10 generated answers with:")
print(f"    * Answer text")
print(f"    * Overall log-likelihood (sum of token log-probs)")
print(f"    * Probability (exp of overall LL)")
print(f"    * Token count (sequence length)")
print(f"    * Per-token log-likelihoods list")
print(f"    * Statistical summary (mean, min, max, std)")
print(f"  - SE metrics for cluster analysis")
