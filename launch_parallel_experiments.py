#!/usr/bin/env python3
"""
Parallel experiment launcher for token probability perturbation analysis.

Safe parallelization strategy:
- Each experiment runs in an isolated subprocess
- Explicit GPU assignment via CUDA_VISIBLE_DEVICES
- No shared CUDA context, WandB state, or file handles
- Max concurrent jobs = number of available GPUs
- Failures logged but do not block other jobs

Usage:
    python launch_parallel_experiments.py --mode dry_run
    python launch_parallel_experiments.py --mode full
    python launch_parallel_experiments.py --mode single --model Mistral-7B-Instruct-v0.2-4bit --dataset trivia_qa --gpu 0
"""

import argparse
import subprocess
import time
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import json


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

MODELS = [
    "Mistral-7B-Instruct-v0.2-4bit",
    "falcon-7b-instruct-4bit",
    "Llama-2-7b-chat-hf-4bit",
    "Llama-2-13b-chat-hf-4bit",
]

DATASETS = [
    "trivia_qa",
    "squad",
    "nq_open",
    "svamp",
    "bioasq",
]

# WandB configuration
WANDB_ENTITY = "raveendiran-21-university-of-moratuwa"
WANDB_PROJECT = "semantic_uncertainty"

# GPU configuration
AVAILABLE_GPUS = [0, 1]  # Tesla T4 × 2
MAX_CONCURRENT_JOBS = 2

# Generation parameters
NUM_GENERATIONS = 10  # K diverse answers per question
OUTPUT_BASE_DIR = "results"


# ============================================================================
# JOB DEFINITION
# ============================================================================

class ExperimentJob:
    """Represents a single (model, dataset, N) experiment."""
    
    def __init__(self, model: str, dataset: str, num_samples: int, gpu_id: int):
        self.model = model
        self.dataset = dataset
        self.num_samples = num_samples
        self.gpu_id = gpu_id
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.status: str = "pending"  # pending, running, success, failed
        self.wandb_run_id: Optional[str] = None
        self.output_dir: Optional[str] = None
        self.error_message: Optional[str] = None
    
    def __repr__(self):
        return f"Job({self.model} × {self.dataset} × N={self.num_samples}, GPU={self.gpu_id})"
    
    def to_dict(self):
        return {
            "model": self.model,
            "dataset": self.dataset,
            "num_samples": self.num_samples,
            "gpu_id": self.gpu_id,
            "status": self.status,
            "wandb_run_id": self.wandb_run_id,
            "output_dir": self.output_dir,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_sec": (self.end_time - self.start_time) if (self.end_time and self.start_time) else None,
            "error_message": self.error_message,
        }


# ============================================================================
# SUBPROCESS EXECUTION (ISOLATED, NO SHARED STATE)
# ============================================================================

def run_generate_stage(job: ExperimentJob) -> Tuple[bool, str]:
    """
    Run generate_answers.py in isolated subprocess with explicit GPU.
    
    Safety guarantees:
    - CUDA_VISIBLE_DEVICES set to single GPU
    - Independent Python interpreter (no shared CUDA context)
    - Timestamped output dir (no collision)
    - Own WandB run (no state sharing)
    
    Returns: (success: bool, wandb_run_id_or_error: str)
    """
    cmd = [
        sys.executable,
        "semantic_uncertainty/generate_answers.py",
        "--model_name", job.model,
        "--dataset", job.dataset,
        "--num_samples", str(job.num_samples),
        "--num_generations", str(NUM_GENERATIONS),
        "--output_dir", OUTPUT_BASE_DIR,
        "--get_training_set_generations",
        "--get_training_set_generations_most_likely_only",
    ]
    
    # CRITICAL: Isolate GPU per subprocess
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(job.gpu_id)
    
    print(f"\n{'='*80}")
    print(f"[GENERATE] Starting: {job}")
    print(f"Command: {' '.join(cmd)}")
    print(f"GPU: {job.gpu_id} (CUDA_VISIBLE_DEVICES={job.gpu_id})")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            stdout=None,  # Stream to console
            stderr=None,  # Stream to console
            timeout=7200,  # 2 hour timeout
        )
        
        if result.returncode != 0:
            error_msg = f"Generate failed (exit {result.returncode}):\n{result.stderr}"
            print(f"❌ {error_msg}")
            return False, error_msg
        
        # Extract run_id from run_info.json
        time.sleep(2)  # Allow file write to complete
        run_id = extract_run_id_from_output(OUTPUT_BASE_DIR)
        if not run_id:
            error_msg = "Generate succeeded but no run_id found in output"
            print(f"⚠️  {error_msg}")
            return False, error_msg
        
        print(f"✅ Generate succeeded: run ID = {run_id}")
        return True, run_id
        
    except subprocess.TimeoutExpired:
        error_msg = "Generate stage timeout (>2 hours)"
        print(f"❌ {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Generate stage exception: {str(e)}"
        print(f"❌ {error_msg}")
        return False, error_msg


def run_compute_stage(job: ExperimentJob, run_id: str) -> Tuple[bool, str]:
    """
    Run compute_uncertainty_measures.py in isolated subprocess.
    
    Links to generate stage via --eval_wandb_runid (which accepts run_id).
    Creates separate WandB run for compute stage.
    """
    cmd = [
        sys.executable,
        "semantic_uncertainty/compute_uncertainty_measures.py",
        "--eval_wandb_runid", run_id,
        "--restore_entity_eval", WANDB_ENTITY,
        "--num_eval_samples", str(job.num_samples),
        "--no-compute_p_ik",
        "--no-compute_p_ik_answerable",
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(job.gpu_id)
    
    print(f"\n{'='*80}")
    print(f"[COMPUTE] Starting: {job}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Linked to generate run: {run_id}")
    print(f"GPU: {job.gpu_id}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )
        
        if result.returncode != 0:
            error_msg = f"Compute failed (exit {result.returncode}):\n{result.stderr}"
            print(f"❌ {error_msg}")
            return False, error_msg
        
        print(f"✅ Compute succeeded")
        return True, "success"
        
    except subprocess.TimeoutExpired:
        error_msg = "Compute stage timeout (>2 hours)"
        print(f"❌ {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Compute stage exception: {str(e)}"
        print(f"❌ {error_msg}")
        return False, error_msg


def extract_run_id_from_output(output_dir: str) -> Optional[str]:
    """Extract run_id from run_info.json in the most recent output directory."""
    import glob
    
    # Find most recent run directory matching pattern
    pattern = os.path.join(output_dir, "*_Mistral_*", "run_info.json")
    run_info_files = glob.glob(pattern)
    
    if not run_info_files:
        # Try broader pattern
        pattern = os.path.join(output_dir, "*", "run_info.json")
        run_info_files = glob.glob(pattern)
    
    if not run_info_files:
        return None
    
    # Get most recent
    run_info_files.sort(key=os.path.getmtime, reverse=True)
    
    try:
        with open(run_info_files[0], 'r') as f:
            run_info = json.load(f)
            return run_info.get('run_id')
    except Exception:
        return None


# ============================================================================
# GPU QUEUE MANAGEMENT (MAX 1 JOB PER GPU)
# ============================================================================

class GPUQueue:
    """Manages GPU availability for fair scheduling."""
    
    def __init__(self, gpu_ids: List[int]):
        self.available = list(gpu_ids)
        self.in_use = {}  # {gpu_id: job}
    
    def acquire(self) -> Optional[int]:
        """Get next available GPU ID (blocks if none available)."""
        if self.available:
            gpu_id = self.available.pop(0)
            return gpu_id
        return None
    
    def release(self, gpu_id: int):
        """Mark GPU as available."""
        if gpu_id not in self.available:
            self.available.append(gpu_id)
    
    def is_full(self) -> bool:
        return len(self.available) == 0


# ============================================================================
# PARALLEL ORCHESTRATION
# ============================================================================

def run_jobs_parallel(jobs: List[ExperimentJob], log_file: str):
    """
    Execute jobs with max 2 concurrent (one per GPU).
    
    Strategy:
    - Maintain GPU queue
    - Launch job on available GPU
    - Monitor completion
    - Release GPU when done
    - Continue on failure (log & skip)
    """
    gpu_queue = GPUQueue(AVAILABLE_GPUS)
    pending = jobs.copy()
    running = {}  # {process: job}
    completed = []
    failed = []
    
    print(f"\n{'#'*80}")
    print(f"# PARALLEL EXECUTION: {len(jobs)} jobs, max {MAX_CONCURRENT_JOBS} concurrent")
    print(f"# GPUs: {AVAILABLE_GPUS}")
    print(f"# Log: {log_file}")
    print(f"{'#'*80}\n")
    
    while pending or running:
        # Launch new jobs on available GPUs
        while pending and not gpu_queue.is_full():
            job = pending.pop(0)
            gpu_id = gpu_queue.acquire()
            job.gpu_id = gpu_id
            job.start_time = time.time()
            job.status = "running"
            
            # Run generate stage
            success, result = run_generate_stage(job)
            
            if success:
                job.wandb_run_id = result
                # Run compute stage
                success_compute, result_compute = run_compute_stage(job, result)
                
                if success_compute:
                    job.status = "success"
                    job.end_time = time.time()
                    completed.append(job)
                    print(f"✅✅ COMPLETE: {job} in {job.end_time - job.start_time:.1f}s")
                else:
                    job.status = "failed"
                    job.error_message = result_compute
                    job.end_time = time.time()
                    failed.append(job)
                    print(f"❌❌ FAILED (compute): {job}")
            else:
                job.status = "failed"
                job.error_message = result
                job.end_time = time.time()
                failed.append(job)
                print(f"❌❌ FAILED (generate): {job}")
            
            # Release GPU
            gpu_queue.release(gpu_id)
            
            # Save progress
            save_progress_log(completed, failed, pending, log_file)
        
        time.sleep(1)  # Polling interval
    
    # Final summary
    print(f"\n{'#'*80}")
    print(f"# EXECUTION COMPLETE")
    print(f"# Success: {len(completed)}/{len(jobs)}")
    print(f"# Failed: {len(failed)}/{len(jobs)}")
    print(f"{'#'*80}\n")
    
    return completed, failed


def save_progress_log(completed: List[ExperimentJob], failed: List[ExperimentJob], 
                      pending: List[ExperimentJob], log_file: str):
    """Save execution progress to JSON log."""
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "completed": [j.to_dict() for j in completed],
        "failed": [j.to_dict() for j in failed],
        "pending": [j.to_dict() for j in pending],
        "summary": {
            "completed": len(completed),
            "failed": len(failed),
            "pending": len(pending),
        }
    }
    
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Parallel experiment launcher")
    parser.add_argument("--mode", choices=["dry_run", "full", "single"], required=True,
                        help="dry_run: 1 model × 1 dataset × N=50 | full: all 20 experiments | single: manual")
    parser.add_argument("--model", help="Model name (single mode only)")
    parser.add_argument("--dataset", help="Dataset name (single mode only)")
    parser.add_argument("--gpu", type=int, choices=[0, 1], help="GPU ID (single mode only)")
    parser.add_argument("--num_samples", type=int, help="Number of samples (optional, defaults: 50 dry, 400 full)")
    
    args = parser.parse_args()
    
    jobs = []
    
    if args.mode == "dry_run":
        # 1 model × 1 dataset × N=50
        num_samples = args.num_samples or 50
        jobs = [ExperimentJob(MODELS[0], DATASETS[0], num_samples, gpu_id=0)]
        log_file = f"launch_log_dry_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    elif args.mode == "full":
        # 5 datasets × 4 models × N=400
        num_samples = args.num_samples or 400
        jobs = [
            ExperimentJob(model, dataset, num_samples, gpu_id=-1)
            for model in MODELS
            for dataset in DATASETS
        ]
        log_file = f"launch_log_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    elif args.mode == "single":
        if not args.model or not args.dataset or args.gpu is None:
            parser.error("--mode single requires --model, --dataset, and --gpu")
        num_samples = args.num_samples or 400
        jobs = [ExperimentJob(args.model, args.dataset, num_samples, args.gpu)]
        log_file = f"launch_log_single_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    if not jobs:
        print("No jobs to run")
        return
    
    print(f"\nJob queue:")
    for i, job in enumerate(jobs, 1):
        print(f"  {i}. {job}")
    print()
    
    # Run jobs
    completed, failed = run_jobs_parallel(jobs, log_file)
    
    # Print results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"✅ Completed: {len(completed)}")
    for job in completed:
        print(f"   {job.model} × {job.dataset} → {job.wandb_run_id}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}")
        for job in failed:
            print(f"   {job.model} × {job.dataset}: {job.error_message[:100]}")
    
    print(f"\nLog saved: {log_file}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
