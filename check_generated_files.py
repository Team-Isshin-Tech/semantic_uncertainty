"""Check which generation files exist on server."""
import os
import json
from datetime import datetime

def check_generated_files():
    """List all generated files with details."""
    
    results_dir = 'results'
    
    if not os.path.exists(results_dir):
        print("❌ results/ directory not found")
        return
    
    runs = sorted(
        [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))],
        key=lambda x: os.path.getmtime(os.path.join(results_dir, x)),
        reverse=True
    )
    
    print("=" * 100)
    print("ALL GENERATION RUNS:")
    print("=" * 100)
    
    for i, run in enumerate(runs[:15], 1):
        run_path = os.path.join(results_dir, run)
        artifacts_dir = os.path.join(run_path, 'artifacts')
        
        # Get modification time
        mod_time = datetime.fromtimestamp(os.path.getmtime(run_path))
        
        print(f"\n{i}. {run}")
        print(f"   Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check run_info.json
        run_info_path = os.path.join(run_path, 'run_info.json')
        if os.path.exists(run_info_path):
            with open(run_info_path, 'r') as f:
                info = json.load(f)
            args = info.get('args', {})
            print(f"   Model: {args.get('model_name', 'unknown')}")
            print(f"   Dataset: {args.get('dataset', 'unknown')}")
            print(f"   Samples: {args.get('num_samples', 'unknown')}")
            print(f"   Generations: {args.get('num_generations', 'unknown')}")
        
        # Check pickle files
        if os.path.exists(artifacts_dir):
            val_pkl = os.path.join(artifacts_dir, 'validation_generations.pkl')
            train_pkl = os.path.join(artifacts_dir, 'train_generations.pkl')
            
            files_exist = []
            if os.path.exists(val_pkl):
                size_mb = os.path.getsize(val_pkl) / (1024**2)
                files_exist.append(f"validation_generations.pkl ({size_mb:.1f}MB)")
            if os.path.exists(train_pkl):
                size_mb = os.path.getsize(train_pkl) / (1024**2)
                files_exist.append(f"train_generations.pkl ({size_mb:.1f}MB)")
            
            if files_exist:
                print(f"   Files: {', '.join(files_exist)}")
            else:
                print(f"   Files: ❌ NONE (generation incomplete)")
        else:
            print(f"   Files: artifacts/ directory not found")
    
    # Find falcon + 400 specifically
    print("\n" + "=" * 100)
    print("FALCON + 400 RUNS:")
    print("=" * 100)
    
    falcon_runs = [r for r in runs if 'falcon' in r.lower() and '400' in r]
    
    if not falcon_runs:
        print("❌ No falcon + 400 runs found")
    else:
        for run in falcon_runs[:5]:
            run_path = os.path.join(results_dir, run)
            artifacts_dir = os.path.join(run_path, 'artifacts')
            
            print(f"\n✅ {run}")
            
            val_pkl = os.path.join(artifacts_dir, 'validation_generations.pkl')
            train_pkl = os.path.join(artifacts_dir, 'train_generations.pkl')
            
            if os.path.exists(val_pkl):
                size_mb = os.path.getsize(val_pkl) / (1024**2)
                print(f"   ✅ validation_generations.pkl ({size_mb:.1f}MB) - READY FOR COMPUTE")
            else:
                print(f"   ❌ validation_generations.pkl NOT FOUND")
            
            if os.path.exists(train_pkl):
                size_mb = os.path.getsize(train_pkl) / (1024**2)
                print(f"   ✅ train_generations.pkl ({size_mb:.1f}MB)")
            else:
                print(f"   ❌ train_generations.pkl NOT FOUND")

if __name__ == '__main__':
    check_generated_files()
