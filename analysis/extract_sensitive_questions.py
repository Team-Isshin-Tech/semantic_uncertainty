"""
Extract and display the actual question text for the top 10 most sensitive questions
"""
import pickle
import json
from pathlib import Path

# Top 10 most sensitive question IDs
sensitive_questions = [
    ('qw_5145--123/123_1138870.txt#0_1', 2.0, 0.051046),
    ('qw_5145--123/123_1138870.txt#0_1', 0.0, 0.049981),
    ('qb_7817--34/34_114217.txt#0_0', 0.0, 0.047596),
    ('qw_5145--123/123_1138870.txt#0_1', 1.0, 0.046317),
    ('qw_5145--123/123_1138870.txt#0_1', 0.5, 0.036831),
    ('dpql_5542--101/101_701718.txt#0_2', 0.5, 0.036316),
    ('qb_7817--34/34_114217.txt#0_0', 0.5, 0.035494),
    ('qw_15940--120/120_2983777.txt#0_0', 0.5, 0.034201),
    ('sfq_4557--72/72_1569959.txt#0_0', 0.5, 0.033923),
    ('sfq_20006--59/59_2784672.txt#0_0', 0.0, 0.033841),
]

print("=" * 100)
print("TOP 10 MOST SENSITIVE QUESTIONS - DETAILED ANALYSIS")
print("=" * 100)

# Load validation generations to get question text
generations_path = Path('../results/validation_generations.pkl')

if generations_path.exists():
    print("\nLoading validation_generations.pkl...")
    with open(generations_path, 'rb') as f:
        generations = pickle.load(f)
    
    print(f"Loaded {len(generations)} questions\n")
    
    # Extract unique question IDs
    unique_ids = list(set([q[0] for q in sensitive_questions]))
    
    for i, (qid, mu, variance) in enumerate(sensitive_questions, 1):
        print(f"\n{'='*100}")
        print(f"RANK {i}: Sensitivity Variance = {variance:.6f} at μ = {mu}")
        print(f"{'='*100}")
        print(f"Question ID: {qid}")
        
        # Find this question in generations
        if qid in generations:
            gen_data = generations[qid]
            
            # Extract question text (different formats possible)
            if isinstance(gen_data, dict):
                question_text = gen_data.get('question', 'Question text not found')
                answers = gen_data.get('answers', [])
                most_likely_answer = gen_data.get('most_likely_answer', 'N/A')
                correctness = gen_data.get('correctness', 'N/A')
            elif isinstance(gen_data, (list, tuple)) and len(gen_data) > 0:
                # Sometimes stored as tuple/list
                if isinstance(gen_data[0], str):
                    question_text = gen_data[0]
                    answers = gen_data[1] if len(gen_data) > 1 else []
                    most_likely_answer = gen_data[2] if len(gen_data) > 2 else 'N/A'
                    correctness = gen_data[-1] if len(gen_data) > 3 else 'N/A'
                else:
                    question_text = "Question structure unknown"
                    answers = []
                    most_likely_answer = 'N/A'
                    correctness = 'N/A'
            else:
                question_text = f"Data format: {type(gen_data)}"
                answers = []
                most_likely_answer = 'N/A'
                correctness = 'N/A'
            
            print(f"\nQuestion Text:")
            print(f"  {question_text}")
            
            if answers:
                print(f"\nGround Truth Answers: {answers}")
            
            if most_likely_answer != 'N/A':
                print(f"Most Likely Generated Answer: {most_likely_answer}")
            
            if correctness != 'N/A':
                print(f"Correctness: {'✓ Correct' if correctness == 1 else '✗ Incorrect'}")
            
            # Get additional details from JSONL
            with open('../results/se_after_noise_complete.jsonl', 'r') as f:
                for line in f:
                    record = json.loads(line)
                    if record['question_id'] == qid and abs(record['mu'] - mu) < 0.01:
                        print(f"\nSemantic Entropy Details (μ={mu}):")
                        print(f"  SE before noise: {record['se_before']:.4f}")
                        print(f"  SE mean after noise: {record['se_mean']:.4f}")
                        print(f"  Delta SE: {record['delta_se']:.4f}")
                        print(f"  SE std (across samples): {record['se_std']:.4f}")
                        break
        else:
            print(f"\n⚠ Question not found in generations file")
    
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"\nUnique Questions: {len(unique_ids)}")
    print("\nMost Frequently Appearing:")
    from collections import Counter
    id_counts = Counter([q[0] for q in sensitive_questions])
    for qid, count in id_counts.most_common(3):
        print(f"  {qid}: {count} times")
    
else:
    print(f"\n⚠ File not found: {generations_path}")
    print("\nTrying to extract from JSONL only...")
    
    # Fallback: get what we can from JSONL
    for i, (qid, mu, variance) in enumerate(sensitive_questions, 1):
        print(f"\n{'='*100}")
        print(f"RANK {i}: Question ID = {qid}, μ = {mu}, Variance = {variance:.6f}")
        print(f"{'='*100}")
        
        with open('../results/se_after_noise_complete.jsonl', 'r') as f:
            for line in f:
                record = json.loads(line)
                if record['question_id'] == qid and abs(record['mu'] - mu) < 0.01:
                    print(f"SE before: {record['se_before']:.4f}")
                    print(f"SE after: {record['se_mean']:.4f}")
                    print(f"Delta SE: {record['delta_se']:.4f}")
                    print(f"Correctness: {record['correctness']}")
                    break

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
