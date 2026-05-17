"""
Inspect uncertainty_measures.pkl which should have the validation data
"""
import os
os.environ['WANDB_API_KEY'] = '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'

import wandb
import pickle
import tempfile
import glob

api = wandb.Api()
entity = 'raveendiran-21-university-of-moratuwa'
project = 'semantic_uncertainty'
run_id = '8zvki77b'

print(f"Accessing run: {entity}/{project}/{run_id}")
run = api.run(f"{entity}/{project}/{run_id}")

print(f"Run name: {run.name}")

# Get the uncertainty_measures.pkl file specifically
files = run.files()
uncertainty_file = None

for f in files:
    if 'uncertainty_measures.pkl' in f.name:
        uncertainty_file = f
        break

if not uncertainty_file:
    print("✗ uncertainty_measures.pkl not found")
    exit(1)

print(f"Found file: {uncertainty_file.name}\n")

with tempfile.TemporaryDirectory() as tmpdir:
    # Download file
    uncertainty_file.download(root=tmpdir, replace=True)
    
    # Find the downloaded file
    pkl_files_found = glob.glob(os.path.join(tmpdir, '*.pkl'))
    if not pkl_files_found:
        print(f"✗ No pkl files downloaded")
        print(f"Files in tmpdir: {os.listdir(tmpdir)}")
        exit(1)
    
    pkl_path = pkl_files_found[0]
    print(f"Loading from: {pkl_path}\n")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Data structure:")
    print(f"  Type: {type(data)}")
    print(f"  Keys: {list(data.keys())}")
    
    if isinstance(data, dict):
        # Check first few keys to understand structure
        for key_idx, key in enumerate(list(data.keys())[:2]):
            print(f"\n  Sample [{key_idx}] (key={key}):")
            value = data[key]
            print(f"    Type: {type(value)}")
            
            if isinstance(value, dict):
                print(f"    Keys: {list(value.keys())}")
                
                # Check for responses
                if 'responses' in value:
                    responses = value['responses']
                    print(f"    Responses: {len(responses)} items")
                    
                    if responses:
                        resp = responses[0]
                        print(f"      First response type: {type(resp)}")
                        
                        if isinstance(resp, tuple):
                            print(f"      Tuple length: {len(resp)}")
                            
                            # Print each element info
                            for i, elem in enumerate(resp):
                                elem_info = f"[{i}] {type(elem).__name__}"
                                if hasattr(elem, 'shape'):
                                    elem_info += f" shape={elem.shape}"
                                elif isinstance(elem, (list, tuple)):
                                    elem_info += f" len={len(elem)}"
                                elif isinstance(elem, str):
                                    elem_info += f" (text)"
                                print(f"        {elem_info}")
                
                # Check for semantic_ids
                if 'semantic_ids' in value:
                    sem_ids = value['semantic_ids']
                    print(f"\n    Semantic IDs present: {type(sem_ids)}")
                    if isinstance(sem_ids, dict):
                        print(f"      Sample keys: {list(sem_ids.keys())[:3]}")

print(f"\n" + "="*80)

# Check if logits are present in responses
print("\nCRITICAL CHECK: LOGITS AND SEMANTICS")
print("="*80)

found_logits = False
found_semantics = False

for key in list(data.keys())[:10]:  # Check first 10
    sample = data[key]
    if isinstance(sample, dict) and 'responses' in sample:
        responses = sample['responses']
        if responses and isinstance(responses[0], tuple):
            if len(responses[0]) >= 4:
                logits = responses[0][3]
                if hasattr(logits, 'shape'):
                    found_logits = True
                    print(f"✓ LOGITS FOUND: Shape {logits.shape}")
                    break

for key in list(data.keys())[:10]:
    sample = data[key]
    if isinstance(sample, dict) and 'semantic_ids' in sample:
        found_semantics = True
        print(f"✓ SEMANTIC_IDS FOUND: {type(sample['semantic_ids'])}")
        break

if not found_logits:
    print(f"✗ LOGITS NOT FOUND - responses may not have logits_per_token")

if not found_semantics:
    print(f"✗ SEMANTIC_IDS NOT FOUND - clusters not pre-computed")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
