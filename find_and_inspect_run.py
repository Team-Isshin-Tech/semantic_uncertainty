"""
Find and inspect the specific WandB run
"""
import os
os.environ['WANDB_API_KEY'] = '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'

import wandb
import sys

api = wandb.Api()
entity = 'raveendiran-21-university-of-moratuwa'
project = 'semantic_uncertainty'
run_name = 'pretty-eon-1379'

print(f"Looking for run: {run_name}")
print(f"Entity: {entity}, Project: {project}\n")

try:
    # Try to get the run directly by constructing the path
    # The format could be run_name or run_id
    
    # Method 1: Try using run name as the path component
    print("Method 1: Trying to access by name...")
    try:
        run = api.run(f"{entity}/{project}/{run_name}")
        print(f"✓ Found run by name: {run.name} (id: {run.id})")
        
        # Now inspect the run
        print("\nChecking artifacts...")
        files = run.files()
        print(f"Run has {len(files)} files/artifacts")
        
        for f in files[:10]:
            print(f"  - {f.name}")
            
        # Try to download validation_generations.pkl
        pkl_files = [f for f in files if 'validation_generations.pkl' in f.name or 'generations' in f.name.lower()]
        if pkl_files:
            print(f"\n✓ Found pickle file: {pkl_files[0].name}")
        else:
            print(f"\n✗ No pickle files found with 'generations' in name")
            
    except Exception as e:
        print(f"  Method 1 failed: {e}")
        
        # Method 2: Try filtering runs
        print("\nMethod 2: Filtering recent runs...")
        try:
            # Get just the recent runs to avoid timeout
            filters = {"$or": [{"displayName": {"$regex": "pretty"}}]}
            runs = api.runs(f"{entity}/{project}", filters=filters, per_page=10)
            print(f"Filtered search returned:")
            for r in runs:
                print(f"  - {r.name} (id: {r.id})")
        except Exception as e2:
            print(f"  Method 2 failed: {e2}")
            
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
