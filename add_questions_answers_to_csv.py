"""
Add question text and 10 generated answers to the CSV file.
"""
import pandas as pd
import pickle
import json
import os

# Load the CSV file
csv_path = r"c:\New Volume D\University\FYP\Code Files\Existing-SE-main\temporary_se_after_noise_complete.csv"
df = pd.read_csv(csv_path)

print(f"Loaded CSV with {len(df)} rows")
print(f"Unique question IDs in CSV: {df['question_id'].nunique()}")

# Try to find the generation data
question_data = {}

# First, try validation_generations.pkl
pkl_path = r"c:\New Volume D\University\FYP\Code Files\Existing-SE-main\validation_generations.pkl"
if os.path.exists(pkl_path):
    with open(pkl_path, 'rb') as f:
        generations = pickle.load(f)
    
    print(f"\nLoaded pickle file with {len(generations)} question IDs")
    
    for question_id, data in generations.items():
        question_text = data.get('question', '')
        generated_answers = data.get('generations', data.get('answers', data.get('samples', [])))
        
        if isinstance(generated_answers, list):
            generated_answers = generated_answers[:10]
        else:
            generated_answers = []
        
        question_data[question_id] = {
            'question': question_text,
            'answers': generated_answers
        }

# Check if we need to look elsewhere
csv_ids = set(df['question_id'].unique())
pkl_ids = set(question_data.keys())
matching_ids = len(csv_ids & pkl_ids)

print(f"CSV unique question IDs: {len(csv_ids)}")
print(f"Pickle question IDs: {len(pkl_ids)}")
print(f"Matching IDs: {matching_ids}")

if matching_ids == 0:
    print("\n⚠ No matching IDs found in pickle file")
    print("Searching for generation files in the workspace...")
    
    # Search for generation files
    workspace = r"c:\New Volume D\University\FYP\Code Files\Existing-SE-main"
    for root, dirs, files in os.walk(workspace):
        for file in files:
            if 'generation' in file.lower() and file.endswith('.pkl'):
                filepath = os.path.join(root, file)
                print(f"Found: {filepath}")

print("\n" + "="*80)
print("Creating output CSV with question and answer columns...")
print("="*80)

# Add columns to dataframe
df['question'] = df['question_id'].map(lambda x: question_data.get(x, {}).get('question', ''))

# Add 10 answer columns
for i in range(10):
    df[f'answer_{i+1}'] = df['question_id'].map(
        lambda x: question_data.get(x, {}).get('answers', [])[i] if len(question_data.get(x, {}).get('answers', [])) > i else ''
    )

# Save the updated CSV
output_path = r"c:\New Volume D\University\FYP\Code Files\Existing-SE-main\temporary_se_after_noise_complete_with_qa.csv"
df.to_csv(output_path, index=False)

print(f"\n✓ Saved updated CSV to: temporary_se_after_noise_complete_with_qa.csv")
print(f"✓ CSV now has {len(df.columns)} columns")

# Show statistics
questions_found = (df['question'] != '').sum()
print(f"\n📊 Statistics:")
print(f"  - Total rows: {len(df)}")
print(f"  - Rows with question text: {questions_found} ({questions_found/len(df)*100:.1f}%)")
print(f"  - Rows without question text: {len(df) - questions_found}")

# Show a sample
if questions_found > 0:
    sample_idx = df[df['question'] != ''].index[0]
    print(f"\n📝 Sample (row {sample_idx}):")
    print(f"  Question ID: {df.iloc[sample_idx]['question_id']}")
    print(f"  Question: {df.iloc[sample_idx]['question']}")
    if df.iloc[sample_idx]['answer_1']:
        print(f"  Answer 1: {df.iloc[sample_idx]['answer_1'][:80]}...")
else:
    print(f"\n⚠ No question texts were found in the data sources")
    print(f"  The CSV was created but question and answer columns are empty")
