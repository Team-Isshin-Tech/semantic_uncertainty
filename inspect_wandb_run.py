"""
Inspect a WandB run to check if logits and cluster info are stored in validation_generations.pkl
"""
import os
import pickle
import json
from pathlib import Path

# Set environment variables from user input
os.environ['WANDB_API_KEY'] = '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'
os.environ['WANDB_SEM_UNC_ENTITY'] = 'raveendiran-21-university-of-moratuwa'

import wandb

print("=" * 100)
print("INSPECTING WANDB RUN: pretty-eon-1379")
print("=" * 100)

api = wandb.Api()
run_path = f"{os.environ['WANDB_SEM_UNC_ENTITY']}/semantic_uncertainty/pretty-eon-1379"

try:
    run = api.run(run_path)
    print(f"\n✓ Found run: {run.name}")
    print(f"  Run ID: {run.id}")
    print(f"  Config: {dict(run.config)}")
    print(f"\n  Files available:")
    for file in run.files():
        print(f"    - {file.name} ({file.size} bytes)")
except Exception as e:
    print(f"✗ Error accessing run: {e}")
    exit(1)

# Try to download and inspect validation_generations.pkl
print("\n" + "=" * 100)
print("CHECKING: validation_generations.pkl")
print("=" * 100)

try:
    # Download the file
    pkl_file = run.file('validation_generations.pkl')
    print(f"\n✓ Found validation_generations.pkl")
    
    # Download to temp location
    temp_path = Path('/tmp/validation_generations.pkl')
    pkl_file.download(replace=True, root='/tmp')
    
    # Load and inspect
    with open(temp_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✓ Successfully loaded pickle file")
    print(f"  Type: {type(data)}")
    print(f"  Number of questions: {len(data)}")
    
    # Inspect first question
    if data:
        first_qid = list(data.keys())[0]
        first_q = data[first_qid]
        
        print(f"\n  First question structure:")
        print(f"    Question ID: {first_qid}")
        print(f"    Keys: {list(first_q.keys())}")
        
        # Check responses
        if 'responses' in first_q:
            responses = first_q['responses']
            print(f"\n    Responses: {len(responses)} entries")
            
            if responses:
                first_response = responses[0]
                print(f"      First response type: {type(first_response)}")
                
                if isinstance(first_response, tuple):
                    print(f"      Tuple length: {len(first_response)}")
                    print(f"      Elements:")
                    print(f"        [0] predicted_answer (str): {type(first_response[0])}")
                    print(f"        [1] token_log_likelihoods (list): {type(first_response[1])}")
                    print(f"        [2] embedding (tensor): {type(first_response[2])}")
                    if len(first_response) > 3:
                        print(f"        [3] logits_per_token (list): {type(first_response[3])}, length={len(first_response[3]) if isinstance(first_response[3], (list, tuple)) else 'N/A'}")
                        if len(first_response) > 4:
                            print(f"        [4] generated_token_ids (list): {type(first_response[4])}, length={len(first_response[4]) if isinstance(first_response[4], (list, tuple)) else 'N/A'}")
                        if len(first_response) > 5:
                            print(f"        [5] accuracy (float): {type(first_response[5])}")
                    
                    # Check if logits are present
                    has_logits = len(first_response) > 3 and first_response[3] is not None and len(first_response[3]) > 0
                    print(f"\n    ✓ LOGITS STORED: {has_logits}")
                    
                    if has_logits:
                        first_logit = first_response[3][0]
                        print(f"      First logit shape: {first_logit.shape if hasattr(first_logit, 'shape') else 'N/A'}")
        
        # Check semantic_ids
        if 'semantic_ids' in first_q:
            semantic_ids = first_q['semantic_ids']
            print(f"\n    Semantic IDs: {semantic_ids}")
            print(f"    ✓ CLUSTER DETAILS: YES (semantic_ids present)")
        else:
            print(f"\n    ✓ CLUSTER DETAILS: NO (semantic_ids not in response)")
        
        # Check most_likely_answer
        if 'most_likely_answer' in first_q:
            mla = first_q['most_likely_answer']
            print(f"\n    Most likely answer keys: {list(mla.keys())}")
            has_mla_logits = 'logits_per_token' in mla
            print(f"    ✓ LOGITS IN MOST_LIKELY_ANSWER: {has_mla_logits}")

except FileNotFoundError:
    print(f"✗ validation_generations.pkl not found in run")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print("""
If you see:
  - logits_per_token (list) with shape info: Logits ARE stored
  - LOGITS STORED: True: Full per-token logits are available
  - CLUSTER DETAILS: YES: Semantic clustering info is present
  
Then the generation run stored what you need for noise-only recomputation.
""")
