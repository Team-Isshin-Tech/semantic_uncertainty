"""Find the most recent generate_answers.py run on server."""
import os
import json
import glob
from datetime import datetime

def find_last_run(results_dir='results'):
    """Find the most recent run directory and extract wandb run ID."""
    
    # Find all run directories
    run_dirs = glob.glob(os.path.join(results_dir, '*_*_*_*_*'))
    
    if not run_dirs:
        print("❌ No run directories found in results/")
        print(f"   Checked: {os.path.abspath(results_dir)}")
        return None
    
    # Sort by modification time (most recent first)
    run_dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    
    print("=" * 80)
    print("RECENT RUNS (sorted by latest first):")
    print("=" * 80)
    
    for i, run_dir in enumerate(run_dirs[:10], 1):
        run_info_path = os.path.join(run_dir, 'run_info.json')
        
        basename = os.path.basename(run_dir)
        mod_time = datetime.fromtimestamp(os.path.getmtime(run_dir))
        
        print(f"\n{i}. {basename}")
        print(f"   Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if os.path.exists(run_info_path):
            with open(run_info_path, 'r') as f:
                run_info = json.load(f)
            
            args = run_info.get('args', {})
            print(f"   Model: {args.get('model_name', 'unknown')}")
            print(f"   Dataset: {args.get('dataset', 'unknown')}")
            print(f"   Samples: {args.get('num_samples', 'unknown')}")
            print(f"   Generations: {args.get('num_generations', 'unknown')}")
    
    # Get the most recent run
    latest_run_dir = run_dirs[0]
    basename = os.path.basename(latest_run_dir)
    
    print("\n" + "=" * 80)
    print("SELECTED RUN (MOST RECENT):")
    print("=" * 80)
    print(f"Directory: {latest_run_dir}")
    print(f"Basename: {basename}")
    
    # Check for pickle files in multiple locations
    print("\n" + "Checking for pickle files:")
    
    artifacts_dir = os.path.join(latest_run_dir, 'artifacts')
    val_gen_pkl = os.path.join(artifacts_dir, 'validation_generations.pkl')
    train_gen_pkl = os.path.join(artifacts_dir, 'train_generations.pkl')
    
    if os.path.exists(val_gen_pkl):
        size_mb = os.path.getsize(val_gen_pkl) / (1024**2)
        print(f"  ✅ validation_generations.pkl ({size_mb:.2f} MB)")
    else:
        print(f"  ⚠️  validation_generations.pkl NOT in artifacts/")
        # Search in wandb directory
        user = os.environ.get('USER', 'unknown')
        scratch_dir = os.getenv('SCRATCH_DIR', '.')
        wandb_dir = f'{scratch_dir}/{user}/uncertainty'
        
        wandb_pkl = glob.glob(os.path.join(wandb_dir, '*/files/validation_generations.pkl'))
        if wandb_pkl:
            print(f"       Found in wandb: {wandb_pkl[0]}")
            val_gen_pkl = wandb_pkl[0]
        else:
            print(f"       Also NOT found in wandb cache")
    
    if os.path.exists(train_gen_pkl):
        size_mb = os.path.getsize(train_gen_pkl) / (1024**2)
        print(f"  ✅ train_generations.pkl ({size_mb:.2f} MB)")
    else:
        print(f"  ⚠️  train_generations.pkl (optional)")
    
    # Find wandb run ID from logs
    print("\n" + "Finding wandb run ID:")
    logs_dir = os.path.join(latest_run_dir, 'logs')
    run_log = os.path.join(logs_dir, 'run.log')
    wandb_run_id = None
    
    if os.path.exists(run_log):
        with open(run_log, 'r') as f:
            for line in f:
                if 'wandb' in line.lower() and 'run' in line.lower():
                    print(f"  Found in logs: {line.strip()}")
                # Look for wandb run ID pattern (typically a long string)
                if 'eval_wandb_runid' in line or 'wandb_id' in line.lower():
                    print(f"  {line.strip()}")
    
    # Try to extract from run_info.json
    run_info_path = os.path.join(latest_run_dir, 'run_info.json')
    if os.path.exists(run_info_path):
        with open(run_info_path, 'r') as f:
            run_info = json.load(f)
        if 'run_id' in run_info:
            wandb_run_id = run_info.get('run_id')
            print(f"  ✅ Found in run_info.json: {wandb_run_id}")
    
    # Print next steps
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    
    if os.path.exists(val_gen_pkl):
        print(f"\n✅ validation_generations.pkl is available")
        print(f"\nRun compute_uncertainty_measures.py:")
        print(f"cd ~/praveenasarma/Existing-SE-main/semantic_uncertainty")
        print(f"\npython compute_uncertainty_measures.py \\")
        print(f"    --eval_wandb_runid <YOUR_WANDB_RUN_ID> \\")
        print(f"    --num_workers 4 \\")
        print(f"    --num_eval_samples 100")
        print(f"\n⚠️  Replace <YOUR_WANDB_RUN_ID> with actual run ID from:")
        print(f"   - wandb.ai dashboard")
        print(f"   - {run_log}")
    else:
        print(f"\n❌ validation_generations.pkl NOT FOUND")
        print(f"\nPossible reasons:")
        print(f"1. generate_answers.py did not complete successfully")
        print(f"2. Pickle files are in a different location")
        print(f"3. Previous run failed before saving files")
        print(f"\nCheck the logs:")
        print(f"tail -100 {run_log}")
    
    return {
        'run_dir': latest_run_dir,
        'basename': basename,
        'artifacts_dir': artifacts_dir,
        'wandb_run_id': wandb_run_id,
        'validation_pkl_exists': os.path.exists(val_gen_pkl),
        'validation_pkl_path': val_gen_pkl
    }


if __name__ == '__main__':
    info = find_last_run()
