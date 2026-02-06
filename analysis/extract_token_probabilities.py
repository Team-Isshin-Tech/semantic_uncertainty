"""
Extract detailed token-level and sequence-level probabilities for top 10 questions
Creates structured output for in-depth sequence analysis
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

# Create detailed token-level analysis
output_dir = Path('../reports_2/robustness_analysis')

# File 1: Detailed token probabilities (JSON for structured data)
detailed_data = {}

for rank, (qid, se_info) in enumerate(top_10, 1):
    if qid in generations:
        gen_data = generations[qid]
        responses = gen_data.get('responses', [])
        question_text = gen_data.get('question', 'N/A')
        
        question_entry = {
            'rank': rank,
            'question_id': qid,
            'question': question_text,
            'se_before': float(se_info['se_before']),
            'se_mean': float(se_info['se_mean']),
            'se_std': float(se_info['se_std']),
            'delta_se': float(se_info['delta_se']),
            'answers': []
        }
        
        for ans_idx, response in enumerate(responses, 1):
            if isinstance(response, tuple) and len(response) >= 2:
                answer_text = response[0]
                token_loglikelihoods = response[1] if len(response) > 1 else []
                token_ids = response[4] if len(response) > 4 else []
                
                # Calculate probabilities from log-likelihoods
                token_probs = [np.exp(ll) for ll in token_loglikelihoods]
                
                # Overall statistics
                overall_ll = sum(token_loglikelihoods) if token_loglikelihoods else 0.0
                overall_prob = np.exp(overall_ll)
                
                answer_entry = {
                    'answer_index': ans_idx,
                    'answer_text': answer_text,
                    'overall_log_likelihood': float(overall_ll),
                    'overall_probability': float(overall_prob),
                    'token_count': len(token_loglikelihoods),
                    'tokens': []
                }
                
                # Add per-token information
                for token_idx, (token_ll, token_prob) in enumerate(zip(token_loglikelihoods, token_probs)):
                    answer_entry['tokens'].append({
                        'position': token_idx,
                        'log_likelihood': float(token_ll),
                        'probability': float(token_prob),
                        'contribution_to_entropy': float(-token_prob * np.log(token_prob + 1e-10))
                    })
                
                # Token-level statistics
                if token_loglikelihoods:
                    answer_entry['token_stats'] = {
                        'mean_log_likelihood': float(np.mean(token_loglikelihoods)),
                        'std_log_likelihood': float(np.std(token_loglikelihoods)),
                        'min_log_likelihood': float(np.min(token_loglikelihoods)),
                        'max_log_likelihood': float(np.max(token_loglikelihoods)),
                        'mean_probability': float(np.mean(token_probs)),
                        'std_probability': float(np.std(token_probs)),
                        'min_probability': float(np.min(token_probs)),
                        'max_probability': float(np.max(token_probs))
                    }
                
                question_entry['answers'].append(answer_entry)
        
        detailed_data[qid] = question_entry

# Save as JSON
json_path = output_dir / 'TOP_10_TOKEN_DETAILS.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(detailed_data, f, indent=2)

print(f"✓ Detailed token data saved: {json_path}")

# File 2: Flattened CSV for token-level analysis (one row per token per answer)
token_rows = []

for rank, (qid, se_info) in enumerate(top_10, 1):
    if qid in generations:
        gen_data = generations[qid]
        responses = gen_data.get('responses', [])
        question_text = gen_data.get('question', 'N/A')
        
        for ans_idx, response in enumerate(responses, 1):
            if isinstance(response, tuple) and len(response) >= 2:
                answer_text = response[0]
                token_loglikelihoods = response[1] if len(response) > 1 else []
                
                token_probs = [np.exp(ll) for ll in token_loglikelihoods]
                overall_ll = sum(token_loglikelihoods) if token_loglikelihoods else 0.0
                
                for token_pos, (token_ll, token_prob) in enumerate(zip(token_loglikelihoods, token_probs)):
                    token_rows.append({
                        'Rank': rank,
                        'Question_ID': qid,
                        'Answer_Index': ans_idx,
                        'Answer_Text': answer_text,
                        'Token_Position': token_pos,
                        'Token_LogLikelihood': float(token_ll),
                        'Token_Probability': float(token_prob),
                        'Token_Entropy': float(-token_prob * np.log(token_prob + 1e-10)),
                        'Overall_AnswerLL': float(overall_ll),
                        'Overall_AnswerProb': float(np.exp(overall_ll)),
                        'Token_Count_in_Answer': len(token_loglikelihoods),
                        'SE_before': float(se_info['se_before'])
                    })

token_df = pd.DataFrame(token_rows)
csv_path = output_dir / 'TOP_10_TOKEN_LEVEL_ANALYSIS.csv'
token_df.to_csv(csv_path, index=False)

print(f"✓ Token-level CSV saved: {csv_path}")
print(f"  Total rows: {len(token_df)}")
print(f"  Tokens per question: {len(token_df[token_df['Question_ID'] == token_df.iloc[0]['Question_ID']])} avg")

# File 3: Summary report
with open(output_dir / 'TOKEN_ANALYSIS_SUMMARY.txt', 'w', encoding='utf-8') as out:
    out.write("="*100 + "\n")
    out.write("TOKEN-LEVEL AND SEQUENCE PROBABILITY ANALYSIS\n")
    out.write("="*100 + "\n\n")
    
    out.write("FILES GENERATED:\n")
    out.write("1. TOP_10_TOKEN_DETAILS.json - Full hierarchical structure\n")
    out.write("2. TOP_10_TOKEN_LEVEL_ANALYSIS.csv - Flattened for analysis\n\n")
    
    out.write("WHAT YOU CAN EXTRACT:\n\n")
    
    out.write("1. SEQUENCE PROBABILITIES:\n")
    out.write("   - Each token has its own probability (exponential of log-likelihood)\n")
    out.write("   - Sequence: how probability changes across answer tokens\n")
    out.write("   - Example: 'David' (0.85) 'Cameron' (0.12) = answer_prob = 0.85 × 0.12\n\n")
    
    out.write("2. TOKEN IMPORTANCE (Entropy Contribution):\n")
    out.write("   - High probability tokens → low entropy → certain\n")
    out.write("   - Low probability tokens → high entropy → uncertain\n")
    out.write("   - Token_Entropy = -p × log(p) shows which tokens are uncertain\n\n")
    
    out.write("3. SEQUENCE PATTERNS:\n")
    out.write("   - Decreasing probability: model confident early, uncertain later\n")
    out.write("   - Increasing probability: model uncertain early, confident later\n")
    out.write("   - Variable probability: inconsistent confidence across answer\n\n")
    
    out.write("4. ANSWER CLUSTERING BY TOKEN PATTERNS:\n")
    out.write("   - Short answers (2-3 tokens): model guesses single word confidently\n")
    out.write("   - Long answers (5+ tokens): model generates explanation, may be less confident\n")
    out.write("   - Token variance: how much confidence varies within answer\n\n")
    
    out.write("="*100 + "\n")
    out.write("EXAMPLE ANALYSIS:\n")
    out.write("="*100 + "\n\n")
    
    # Show first question as example
    if top_10:
        qid, se_info = top_10[0]
        if qid in generations:
            gen_data = generations[qid]
            responses = gen_data.get('responses', [])
            
            out.write(f"Question (Rank 1): {gen_data.get('question', 'N/A')}\n\n")
            
            if responses and len(responses) > 0:
                response = responses[0]
                if isinstance(response, tuple) and len(response) >= 2:
                    answer_text = response[0]
                    token_loglikelihoods = response[1]
                    token_probs = [np.exp(ll) for ll in token_loglikelihoods]
                    
                    out.write(f"Example Answer: '{answer_text}'\n")
                    out.write(f"Token-by-Token Probability:\n")
                    
                    words = answer_text.split()
                    for word_idx, (word, token_prob) in enumerate(zip(words[:len(token_probs)], token_probs)):
                        out.write(f"  Token {word_idx}: '{word}' → Probability = {token_prob:.6f}\n")
                    
                    overall_prob = np.exp(sum(token_loglikelihoods))
                    out.write(f"\nOverall Answer Probability: {overall_prob:.6f}\n")
                    out.write(f"Note: Answer_prob ≈ Product of token probabilities\n")
                    out.write(f"      (Due to independence assumption in token generation)\n")

print(f"\n✓ Summary report created: {output_dir / 'TOKEN_ANALYSIS_SUMMARY.txt'}")

print("\n" + "="*100)
print("SUMMARY: TOKEN AND SEQUENCE PROBABILITIES")
print("="*100)
print(f"""
You can now analyze:
  ✓ Token-level probabilities (per position in sequence)
  ✓ Sequence probability patterns (confidence changes across answer)
  ✓ Token entropy (which tokens are uncertain)
  ✓ Answer clustering by token statistics
  ✓ How SE relates to token-level uncertainty

Files available:
  1. TOP_10_TOKEN_DETAILS.json - Hierarchical, human-readable
  2. TOP_10_TOKEN_LEVEL_ANALYSIS.csv - {len(token_df)} rows for statistical analysis
  3. TOKEN_ANALYSIS_SUMMARY.txt - Guide to interpretation
""")
