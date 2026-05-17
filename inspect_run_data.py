"""
Inspect the WandB run data to check for logits and cluster info
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
print(f"Run ID: {run.id}")
print(f"\n" + "="*80)

# List all files in the run
files = run.files()
print(f"\nFiles in run ({len(files)} total):")
for f in sorted([f.name for f in files]):
    print(f"  - {f}")

print(f"\n" + "="*80)

# Look for validation_generations.pkl
pkl_candidates = [f for f in files if 'pkl' in f.name.lower()]
print(f"\nPickle files found: {len(pkl_candidates)}")
for f in pkl_candidates:
    print(f"  - {f.name}")

if not pkl_candidates:
    print("  ✗ No pickle files found!")
    exit(1)

# Download and inspect the first pickle file
pkl_file = pkl_candidates[0]
print(f"\nDownloading {pkl_file.name}...")

with tempfile.TemporaryDirectory() as tmpdir:
    # Use the proper wandb API to download
    try:
        pkl_file.download(root=tmpdir, replace=True)
        # The file is downloaded to tmpdir with its original name
        pkl_path = os.path.join(tmpdir, os.path.basename(pkl_file.name))
    except:
        # If that fails, try using run.file() to get a reference
        pkl_path = os.path.join(tmpdir, pkl_file.name)
    
    print(f"Looking for file at: {pkl_path}")
    
    # Find any pkl files that were downloaded
    pkl_files_found = glob.glob(os.path.join(tmpdir, '*.pkl'))
    if pkl_files_found:
        pkl_path = pkl_files_found[0]
        print(f"Found pickle file: {pkl_path}")
    
    if not os.path.exists(pkl_path):
        print(f"✗ File not found at {pkl_path}")
        print(f"Files in tmpdir: {os.listdir(tmpdir)}")
        exit(1)
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"\nPickle file structure:")
    print(f"  Type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"  Keys: {list(data.keys())[:5]}{'...' if len(data.keys()) > 5 else ''}")
        num_samples = len(data)
        print(f"  Total samples: {num_samples}")
        
        # Inspect the first sample
        if data:
            first_key = list(data.keys())[0]
            sample = data[first_key]
            print(f"\n  First sample (key={first_key}):")
            print(f"    Type: {type(sample)}")
            
            if isinstance(sample, dict):
                print(f"    Keys: {list(sample.keys())}")
                
                # Check for 'responses' key
                if 'responses' in sample:
                    responses = sample['responses']
                    print(f"\n    Responses: {len(responses)} items")
                    
                    if responses:
                        resp0 = responses[0]
                        print(f"      First response type: {type(resp0)}")
                        
                        if isinstance(resp0, tuple):
                            print(f"      Tuple length: {len(resp0)}")
                            print(f"      Tuple element types:")
                            for i, elem in enumerate(resp0):
                                elem_type = type(elem).__name__
                                if hasattr(elem, 'shape'):
                                    print(f"        [{i}] {elem_type} with shape {elem.shape}")
                                elif isinstance(elem, (list, tuple)):
                                    print(f"        [{i}] {elem_type} with length {len(elem)}")
                                else:
                                    print(f"        [{i}] {elem_type}")
                        
                        # Check for logits in the response
                        if isinstance(resp0, tuple) and len(resp0) >= 4:
                            logits_candidate = resp0[3]
                            if hasattr(logits_candidate, 'shape'):
                                print(f"\n      ✓ LOGITS DETECTED at element [3]:")
                                print(f"        Shape: {logits_candidate.shape}")
                                print(f"        Type: {type(logits_candidate)}")
                            else:
                                print(f"\n      ? Element [3] is not an array: {type(logits_candidate)}")
                
                # Check for semantic_ids
                if 'semantic_ids' in sample:
                    sem_ids = sample['semantic_ids']
                    print(f"\n    ✓ SEMANTIC_IDS PRESENT:")
                    print(f"      Type: {type(sem_ids)}")
                    if isinstance(sem_ids, dict):
                        print(f"      Keys: {list(sem_ids.keys())[:5]}")
                    elif isinstance(sem_ids, list):
                        print(f"      Length: {len(sem_ids)}")
                        print(f"      First few: {sem_ids[:3]}")

print(f"\n" + "="*80)
print("INSPECTION COMPLETE")
