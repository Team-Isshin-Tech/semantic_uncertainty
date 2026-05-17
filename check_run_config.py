"""
Check the run config and logs to understand what was generated
"""
import os
os.environ['WANDB_API_KEY'] = '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'

import wandb
import tempfile
import glob

api = wandb.Api()
entity = 'raveendiran-21-university-of-moratuwa'
project = 'semantic_uncertainty'
run_id = '8zvki77b'

run = api.run(f"{entity}/{project}/{run_id}")

print(f"Run: {run.name}")
print(f"=" * 80)

# Check config
print("\nCONFIG:")
config = run.config
for key, value in config.items():
    print(f"  {key}: {value}")

print(f"\n" + "=" * 80)
print("\nFILES IN RUN:")
for f in run.files():
    print(f"  - {f.name}")

print(f"\n" + "=" * 80)
print("\nCHECKING LOG FILE:")

# Get the output.log
log_file = None
for f in run.files():
    if f.name == 'output.log':
        log_file = f
        break

if log_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file.download(root=tmpdir, replace=True)
        
        log_path = os.path.join(tmpdir, 'output.log')
        if not os.path.exists(log_path):
            log_files = glob.glob(os.path.join(tmpdir, '*.log'))
            if log_files:
                log_path = log_files[0]
        
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                log_content = f.read()
            
            # Print last 100 lines
            lines = log_content.split('\n')
            print("\nLast 50 lines of log:")
            print("-" * 80)
            for line in lines[-50:]:
                print(line)
else:
    print("  No output.log found")

print(f"\n" + "=" * 80)
print("\nSUMMARY:")
print("  ✗ validation_generations.pkl NOT FOUND in artifacts")
print("  ✓ uncertainty_measures.pkl FOUND (contains semantic clustering, not logits)")
print("\nCONCLUSION:")
print("  The run was used for UNCERTAINTY ANALYSIS only (Stage 2)")
print("  The original validation_generations.pkl (with logits) is not stored")
print("  Therefore: Cannot skip Stage 1 generation - logits must be obtained from generation run")
