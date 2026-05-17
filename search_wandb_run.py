"""
Search for the run with name 'pretty-eon-1379' or similar
"""
import os

os.environ['WANDB_API_KEY'] = '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'
os.environ['WANDB_SEM_UNC_ENTITY'] = 'raveendiran-21-university-of-moratuwa'

import wandb

print("Searching for run 'pretty-eon-1379'...")
api = wandb.Api()

# Try different approaches
entity = os.environ['WANDB_SEM_UNC_ENTITY']

print(f"\n1. Trying direct project lookup...")
try:
    runs = list(api.runs(f"{entity}/semantic_uncertainty", per_page=1000))
    print(f"   Found {len(runs)} runs in project 'semantic_uncertainty'")
    
    # Search for the run
    matching_runs = [r for r in runs if 'pretty-eon-1379' in r.name or r.id == 'pretty-eon-1379']
    if matching_runs:
        for r in matching_runs:
            print(f"   ✓ Found: {r.name} (id={r.id})")
    else:
        print(f"   ✗ No exact match for 'pretty-eon-1379'")
        print(f"\n   Recent runs:")
        for r in runs[:5]:
            print(f"     - {r.name} ({r.id})")
            
except Exception as e:
    print(f"   Error: {e}")

# Also check if there's a debug project
print(f"\n2. Checking for 'semantic_uncertainty_debug' project...")
try:
    runs = list(api.runs(f"{entity}/semantic_uncertainty_debug", per_page=100))
    print(f"   Found {len(runs)} runs in debug project")
    matching_runs = [r for r in runs if 'pretty-eon-1379' in r.name or r.id == 'pretty-eon-1379']
    if matching_runs:
        for r in matching_runs:
            print(f"   ✓ Found: {r.name} (id={r.id})")
except Exception as e:
    print(f"   Debug project not found or error: {e}")
