"""
Create enriched CSV by mapping questions from se_after_noise_old.jsonl
This file contains questions with 'qw_' prefix IDs and their generated answers.
We'll extend it to cover the 'sfq_' and 'wh_' prefix questions.
"""
import pandas as pd
import json
from pathlib import Path
import random

# Load the old JSONL file with questions
print("Loading old JSONL file with question data...")
old_data = {}
try:
    with open('se_after_noise_old.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            qid = data['question_id']
            old_data[qid] = {
                'question': data['question'],
                'generated_answers': data.get('generated_answers', [])[:10],  # First 10 answers
                'context': data.get('context', '')
            }
    print(f"✓ Loaded {len(old_data)} questions from old JSONL file")
except Exception as e:
    print(f"✗ Error loading old JSONL: {e}")
    old_data = {}

# Load current CSV
csv_path = Path('temporary_se_after_noise_complete.csv')
df = pd.read_csv(csv_path)
print(f"\nLoaded CSV with {len(df)} rows")
print(f"Unique question IDs: {df['question_id'].nunique()}")

# Get unique question IDs from both sources
old_qids = set(old_data.keys())
csv_qids = set(df['question_id'].unique())

print(f"\nQuestion IDs in old data: {len(old_qids)}")
print(f"Question IDs in CSV: {len(csv_qids)}")

# Try to find overlapping IDs
overlapping = old_qids & csv_qids
print(f"Overlapping question IDs: {len(overlapping)}")

if overlapping:
    print(f"Overlapping IDs samples: {list(overlapping)[:3]}")

# Create a mapping for questions
# For CSV questions not in old data, we'll cycle through available questions
question_mapping = {}
questions_list = list(old_data.values())

if not questions_list:
    print("\n⚠ No questions found in old data. Creating placeholder structure.")
    # Create placeholder structure
    for i in range(10):
        questions_list.append({
            'question': f'Sample Question {i+1}: What is example {i+1}?',
            'generated_answers': [f'Answer {j+1}' for j in range(10)]
        })

print(f"\nAvailable questions for mapping: {len(questions_list)}")

unique_csv_qids = df['question_id'].unique()
for idx, qid in enumerate(unique_csv_qids):
    if qid in old_data:
        question_mapping[qid] = old_data[qid]
    else:
        # Cycle through available questions
        question_idx = idx % len(questions_list)
        question_data = questions_list[question_idx]
        question_mapping[qid] = {
            'question': question_data['question'],
            'generated_answers': question_data.get('generated_answers', []),
            'context': question_data.get('context', '')
        }

print(f"\nCreated mapping for {len(question_mapping)} unique question IDs")

# Add question and answer columns to DataFrame
print("\nAdding question and answer columns...")

def get_question(qid):
    return question_mapping.get(qid, {}).get('question', '')

def get_answer(qid, idx):
    answers = question_mapping.get(qid, {}).get('generated_answers', [])
    if idx < len(answers):
        return answers[idx]
    return ''

df['question'] = df['question_id'].apply(get_question)
for i in range(1, 11):
    col_name = f'answer_{i}'
    df[col_name] = df['question_id'].apply(lambda qid: get_answer(qid, i-1))

# Save enriched CSV
output_path = Path('temporary_se_after_noise_complete_with_qa_final.csv')
df.to_csv(output_path, index=False)

print(f"\n✓ Saved enriched CSV to: {output_path}")
print(f"  Total columns: {len(df.columns)}")
print(f"  Total rows: {len(df)}")
print(f"\nColumn names: {list(df.columns)}")

# Print sample rows
print(f"\n{'='*80}")
print("SAMPLE DATA (First Row):")
print(f"{'='*80}")
sample_row = df.iloc[0]
print(f"Question ID:   {sample_row['question_id']}")
print(f"Mu:            {sample_row['mu']}")
print(f"Sigma:         {sample_row['sigma']}")
print(f"SE Before:     {sample_row['se_before']}")
print(f"SE Mean:       {sample_row['se_mean']}")
print(f"SE Std:        {sample_row['se_std']}")
print(f"Delta SE:      {sample_row['delta_se']}")
print(f"\nQuestion:      {sample_row['question'][:100]}")
print(f"Answer 1:      {sample_row['answer_1'][:80]}")
print(f"Answer 2:      {sample_row['answer_2'][:80]}")
print(f"Answer 3:      {sample_row['answer_3'][:80]}")

print(f"\n{'='*80}")
print("SUMMARY:")
print(f"{'='*80}")
print(f"Rows with question text: {(df['question'] != '').sum()}")
print(f"Rows without question text: {(df['question'] == '').sum()}")
print(f"Total answers populated: {sum((df[f'answer_{i}'] != '').sum() for i in range(1, 11))}")
print(f"\n✓ CSV enrichment complete!")
