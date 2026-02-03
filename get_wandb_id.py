"""Extract wandb run ID from logs."""
import re
import sys
import os

def extract_wandb_id_from_logs(run_dir):
    """Extract wandb run ID from generate_answers logs."""
    
    log_file = os.path.join(run_dir, 'logs', 'run.log')
    
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Look for wandb run patterns
    patterns = [
        r'wandb run.*?id[:\s]+([a-z0-9]+)',
        r'run.*?wandb[:\s]+([a-z0-9]+)',
        r'wandb.run.id.*?([a-z0-9]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Parse from wandb directory structure
    import glob
    user = os.environ.get('USER', 'unknown')
    scratch_dir = os.getenv('SCRATCH_DIR', '.')
    wandb_dir = f'{scratch_dir}/{user}/uncertainty'
    
    if os.path.exists(wandb_dir):
        wandb_runs = sorted(
            glob.glob(os.path.join(wandb_dir, 'run-*')),
            key=lambda p: os.path.getmtime(p),
            reverse=True
        )
        if wandb_runs:
            run_id = os.path.basename(wandb_runs[0]).replace('run-', '')
            return run_id
    
    return None


if __name__ == '__main__':
    run_dir = 'results/20260202_235535_falcon_7b_instruct_trivia_qa_400_10'
    
    wandb_id = extract_wandb_id_from_logs(run_dir)
    
    if wandb_id:
        print(f"✅ Found wandb run ID: {wandb_id}")
        print(f"\nNext command:")
        print(f"cd semantic_uncertainty")
        print(f"python compute_uncertainty_measures.py --eval_wandb_runid {wandb_id}")
    else:
        print(f"❌ Could not extract wandb run ID")
        print(f"Check logs manually:")
        print(f"cat {run_dir}/logs/run.log | grep -i wandb")
