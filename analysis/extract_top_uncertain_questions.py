"""
Extract top 10 most uncertain questions (highest SE_before) with their 10 generated answers
and semantic entropy metrics
"""
import pickle
import json
from pathlib import Path
from collections import defaultdict

print("=" * 100)
print("EXTRACTING TOP 10 MOST UNCERTAIN QUESTIONS")
print("=" * 100)

# Step 1: Load JSONL to find top 10 by SE_before
print("\n1. Loading JSONL data...")
se_data = {}
with open('../results/se_after_noise_complete.jsonl', 'r') as f:
    for line in f:
        record = json.loads(line)
        qid = record['question_id']
        
        # Take first occurrence (mu=0.0, sigma=0.5 typically)
        if qid not in se_data:
            se_data[qid] = {
                'se_before': record['se_before'],
                'se_mean': record['se_mean'],
                'se_std': record['se_std'],
                'delta_se': record['delta_se'],
                'correctness': record['correctness']
            }

print(f"   Loaded {len(se_data)} unique questions")

# Step 2: Sort by SE_before and get top 10
print("\n2. Finding top 10 most uncertain questions...")
sorted_questions = sorted(se_data.items(), key=lambda x: x[1]['se_before'], reverse=True)
top_10 = sorted_questions[:10]

print(f"\n   Top 10 SE_before values:")
for i, (qid, data) in enumerate(top_10, 1):
    print(f"   {i:2d}. {qid[:50]:50s}  SE={data['se_before']:.4f}")

# Step 3: Load validation_generations.pkl to get question text and 10 answers
print("\n3. Loading validation_generations.pkl...")
generations_path = Path('../results/validation_generations.pkl')

if not generations_path.exists():
    print(f"   ERROR: {generations_path} not found!")
    exit(1)

with open(generations_path, 'rb') as f:
    generations = pickle.load(f)

print(f"   Loaded {len(generations)} questions with generations")

# Step 4: Extract details and write to file
print("\n4. Extracting details and writing to file...")
output_path = Path('../reports_2/robustness_analysis/TOP_10_UNCERTAIN_QUESTIONS.txt')

with open(output_path, 'w', encoding='utf-8') as out:
    out.write("=" * 100 + "\n")
    out.write("TOP 10 MOST UNCERTAIN QUESTIONS\n")
    out.write("(Ranked by SE_before - Baseline Semantic Entropy)\n")
    out.write("=" * 100 + "\n\n")
    
    for rank, (qid, se_info) in enumerate(top_10, 1):
        out.write("\n" + "=" * 100 + "\n")
        out.write(f"RANK {rank}: SE_before = {se_info['se_before']:.6f}\n")
        out.write("=" * 100 + "\n")
        out.write(f"Question ID: {qid}\n\n")
        
        # Get generation data
        if qid in generations:
            gen_data = generations[qid]
            
            # Extract question text and answers based on data structure
            # Handle dict format
            if isinstance(gen_data, dict):
                question_text = gen_data.get('question', 'N/A')
                
                # Get ground truth from 'reference' dict
                reference = gen_data.get('reference', {})
                if isinstance(reference, dict):
                    ground_truth = reference.get('answers', [])
                    correctness = reference.get('correctness', 'N/A')
                else:
                    ground_truth = []
                    correctness = 'N/A'
                
                # Extract the 10 generated responses from 'responses' key
                responses = gen_data.get('responses', [])
                if responses and len(responses) > 0:
                    # Each response is a tuple where index 0 is the answer text
                    generations_list = [resp[0] if isinstance(resp, tuple) and len(resp) > 0 else str(resp) 
                                       for resp in responses]
                else:
                    generations_list = []
                
                # Get most likely answer
                most_likely_dict = gen_data.get('most_likely_answer', {})
                if isinstance(most_likely_dict, dict):
                    most_likely = most_likely_dict.get('response', 'N/A')
                else:
                    most_likely = str(most_likely_dict)
                    
            # Handle tuple/list format
            elif isinstance(gen_data, (list, tuple)) and len(gen_data) >= 3:
                question_text = gen_data[0] if len(gen_data) > 0 else "N/A"
                ground_truth = gen_data[1] if len(gen_data) > 1 else []
                generations_list = gen_data[2] if len(gen_data) > 2 else []
                most_likely = gen_data[3] if len(gen_data) > 3 else "N/A"
                correctness = gen_data[4] if len(gen_data) > 4 else "N/A"
            else:
                question_text = "N/A"
                ground_truth = []
                generations_list = []
                most_likely = "N/A"
                correctness = "N/A"
            
            if question_text != "N/A":
                out.write(f"QUESTION TEXT:\n")
                out.write(f"  {question_text}\n\n")
                
                out.write(f"GROUND TRUTH ANSWERS:\n")
                if isinstance(ground_truth, list):
                    for ans in ground_truth:
                        out.write(f"  - {ans}\n")
                else:
                    out.write(f"  {ground_truth}\n")
                
                out.write(f"\n{'─' * 100}\n")
                out.write(f"10 GENERATED ANSWERS (with log-likelihoods):\n")
                out.write(f"{'─' * 100}\n\n")
                
                # Extract 10 generated answers
                if generations_list:
                    for i, gen in enumerate(generations_list[:10], 1):
                        if isinstance(gen, dict):
                            response = gen.get('response', 'N/A')
                            logits = gen.get('token_log_likelihoods', [])
                            accuracy = gen.get('accuracy', 'N/A')
                            
                            out.write(f"  Answer {i:2d}: {response}\n")
                            if logits:
                                avg_logit = sum(logits) / len(logits) if logits else 0
                                out.write(f"             Avg log-likelihood: {avg_logit:.6f}\n")
                                out.write(f"             Token log-liks: {logits[:5]}{'...' if len(logits) > 5 else ''}\n")
                            out.write(f"             Accuracy: {accuracy}\n\n")
                        elif isinstance(gen, str):
                            out.write(f"  Answer {i:2d}: {gen}\n\n")
                        else:
                            out.write(f"  Answer {i:2d}: {str(gen)[:100]}\n\n")
                else:
                    out.write(f"  [No generations available]\n\n")
                
                out.write(f"\n{'─' * 100}\n")
                out.write(f"SEMANTIC ENTROPY METRICS:\n")
                out.write(f"{'─' * 100}\n")
                out.write(f"  SE before noise:          {se_info['se_before']:.6f}\n")
                out.write(f"  SE mean after noise:      {se_info['se_mean']:.6f}\n")
                out.write(f"  SE std (across samples):  {se_info['se_std']:.6f}\n")
                out.write(f"  Delta SE (mean - before): {se_info['delta_se']:.6f}\n")
                out.write(f"  Model correctness:        {'✓ CORRECT' if se_info['correctness'] == 1.0 else '✗ INCORRECT'}\n")
                out.write(f"  Most likely answer:       {most_likely if isinstance(most_likely, str) else most_likely.get('response', 'N/A') if isinstance(most_likely, dict) else 'N/A'}\n")
                
            else:
                out.write(f"  [Unable to parse generation data for this question]\n")
        else:
            out.write(f"  [Question not found in generations]\n")
        
        out.write("\n")
    
    # Summary statistics
    out.write("\n" + "=" * 100 + "\n")
    out.write("SUMMARY STATISTICS\n")
    out.write("=" * 100 + "\n\n")
    
    avg_se_before = sum(data['se_before'] for _, data in top_10) / 10
    avg_se_after = sum(data['se_mean'] for _, data in top_10) / 10
    avg_se_std = sum(data['se_std'] for _, data in top_10) / 10
    avg_delta = sum(data['delta_se'] for _, data in top_10) / 10
    correct_count = sum(1 for _, data in top_10 if data['correctness'] == 1.0)
    
    out.write(f"Average SE before:        {avg_se_before:.6f}\n")
    out.write(f"Average SE after:         {avg_se_after:.6f}\n")
    out.write(f"Average SE std:           {avg_se_std:.6f}\n")
    out.write(f"Average Delta SE:         {avg_delta:.6f}\n")
    out.write(f"Correct answers:          {correct_count}/10 ({correct_count*10}%)\n")
    out.write(f"\nSE range:                 {top_10[-1][1]['se_before']:.6f} - {top_10[0][1]['se_before']:.6f}\n")
    
    out.write("\n" + "=" * 100 + "\n")
    out.write("INTERPRETATION\n")
    out.write("=" * 100 + "\n\n")
    out.write("These questions have the HIGHEST baseline semantic entropy, indicating:\n")
    out.write("  1. Model produces DIVERSE answers across 10 generations\n")
    out.write("  2. High epistemic uncertainty - model doesn't \"know\" the answer\n")
    out.write("  3. Answer distribution is SPREAD OUT (not concentrated)\n")
    out.write("  4. These are ideal candidates for:\n")
    out.write("     - Human review in active learning\n")
    out.write("     - Identifying knowledge gaps\n")
    out.write("     - Testing model calibration\n\n")
    
    if correct_count < 5:
        out.write("NOTE: Low correctness rate confirms high SE correlates with uncertainty.\n")
    
    out.write("\n" + "=" * 100 + "\n")

print(f"\n✓ Output written to: {output_path}")
print(f"\n5. File generation complete!")
print("=" * 100)
