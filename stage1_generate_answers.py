"""
STAGE 1: Generate answers and store logits (no noise analysis)

This script focuses purely on:
1. Generating multiple answers per question
2. Storing per-token logits
3. Saving the full validation_generations.pkl with semantic clustering
4. Uploading to WandB

Usage:
    python stage1_generate_answers.py \
        --model_name Mistral-7B-Instruct-v0.3 \
        --dataset trivia_qa \
        --num_samples 400 \
        --num_generations 10
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback if python-dotenv not installed
    with open('.env') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, val = line.strip().split('=', 1)
                os.environ[key] = val

import numpy as np
import torch
import wandb
from tqdm import tqdm

# Import from semantic_uncertainty module
sys.path.insert(0, str(Path(__file__).parent / 'semantic_uncertainty'))
from generate_answers import (
    load_model_and_tokenizer,
    get_dataset,
    generate_answers_for_model
)

def main(args):
    """Main pipeline for Stage 1: Generation only"""
    
    # Initialize WandB
    print("\n" + "="*80)
    print("STAGE 1: GENERATION ONLY")
    print("="*80)
    
    wandb.init(
        project="semantic_uncertainty",
        name=f"generation-{args.model_name.split('/')[-1]}-{args.dataset}",
        config=vars(args),
        tags=["stage1-generation"]
    )
    
    print(f"\n✓ WandB run initialized: {wandb.run.name}")
    print(f"  Run ID: {wandb.run.id}")
    print(f"  Run path: {wandb.run.path}")
    
    # Load model and tokenizer
    print(f"\n[1/4] Loading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    print(f"  ✓ Model loaded successfully")
    
    # Load dataset
    print(f"\n[2/4] Loading dataset: {args.dataset}")
    dataset = get_dataset(
        args.dataset,
        split='validation',
        num_samples=args.num_samples,
        use_context=args.use_context
    )
    print(f"  ✓ Dataset loaded: {len(dataset)} samples")
    
    # Generate answers
    print(f"\n[3/4] Generating answers ({args.num_generations} per question)")
    generations = generate_answers_for_model(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_new_tokens=args.model_max_new_tokens,
        use_context=args.use_context
    )
    print(f"  ✓ Generated {len(generations)} question sets")
    
    # Verify structure
    print(f"\n[4/4] Verifying and uploading to WandB")
    
    sample_qid = list(generations.keys())[0]
    sample_q = generations[sample_qid]
    sample_response = sample_q['responses'][0]
    
    print(f"\n  Structure verification:")
    print(f"    ✓ Questions: {len(generations)}")
    print(f"    ✓ Responses per question: {len(sample_q['responses'])}")
    print(f"    ✓ Response tuple length: {len(sample_response)}")
    print(f"    ✓ Tuple contains:")
    print(f"      [0] predicted_answer: {type(sample_response[0]).__name__}")
    print(f"      [1] token_log_likelihoods: {type(sample_response[1]).__name__} (len={len(sample_response[1])})")
    print(f"      [2] embedding: {type(sample_response[2]).__name__}")
    
    # Check for logits
    if len(sample_response) > 3 and sample_response[3] is not None:
        logits = sample_response[3]
        print(f"      [3] logits_per_token: {type(logits).__name__} (len={len(logits)})")
        if len(logits) > 0:
            print(f"          First logit shape: {logits[0].shape if hasattr(logits[0], 'shape') else 'N/A'}")
            print(f"      ✓ LOGITS CONFIRMED: Available in tuple[3]")
    else:
        print(f"      [3] logits_per_token: NOT PRESENT")
        raise ValueError("Logits not found in response tuple! Generation failed.")
    
    if len(sample_response) > 4:
        print(f"      [4] generated_token_ids: {type(sample_response[4]).__name__} (len={len(sample_response[4])})")
    if len(sample_response) > 5:
        print(f"      [5] accuracy: {type(sample_response[5]).__name__}")
    
    # Check for semantic_ids
    if 'semantic_ids' in sample_q:
        print(f"    ✓ semantic_ids present: {len(sample_q['semantic_ids'])} clusters")
    else:
        print(f"    ✗ semantic_ids NOT FOUND")
    
    # Save locally
    pkl_path = Path(wandb.run.dir) / 'validation_generations.pkl'
    print(f"\n  Saving to: {pkl_path}")
    with open(pkl_path, 'wb') as f:
        pickle.dump(generations, f)
    print(f"    ✓ Saved {pkl_path.stat().st_size / 1e9:.2f} GB")
    
    # Upload to WandB
    wandb.save(str(pkl_path))
    print(f"    ✓ Uploaded to WandB")
    
    # Save metadata
    metadata = {
        'run_id': wandb.run.id,
        'run_path': wandb.run.path,
        'num_questions': len(generations),
        'num_responses_per_question': len(sample_q['responses']),
        'has_logits': True,
        'has_semantic_ids': 'semantic_ids' in sample_q,
        'model': args.model_name,
        'dataset': args.dataset,
        'num_samples': args.num_samples,
        'num_generations': args.num_generations,
    }
    
    metadata_path = Path(wandb.run.dir) / 'generation_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    wandb.save(str(metadata_path))
    
    # Log summary
    wandb.summary['total_questions'] = len(generations)
    wandb.summary['responses_per_question'] = len(sample_q['responses'])
    wandb.summary['has_logits'] = True
    wandb.summary['has_semantic_ids'] = 'semantic_ids' in sample_q
    
    print("\n" + "="*80)
    print("✓ STAGE 1 COMPLETE")
    print("="*80)
    print(f"\nWandB Run: {wandb.run.path}")
    print(f"Run ID to use in Stage 2: {wandb.run.id}")
    print(f"\nNext step: Run Stage 2 noise analysis with:")
    print(f"  python stage2_noise_analysis.py --eval_wandb_runid {wandb.run.id}")
    
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 1: Generate answers and store logits')
    
    parser.add_argument('--model_name', type=str, default='Mistral-7B-Instruct-v0.3',
                        help='Model to use for generation')
    parser.add_argument('--dataset', type=str, default='trivia_qa',
                        help='Dataset to use')
    parser.add_argument('--num_samples', type=int, default=400,
                        help='Number of samples to process')
    parser.add_argument('--num_generations', type=int, default=10,
                        help='Number of answers to generate per question')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for generation')
    parser.add_argument('--model_max_new_tokens', type=int, default=50,
                        help='Max new tokens to generate')
    parser.add_argument('--use_context', action='store_true',
                        help='Whether to use context in prompts')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    main(args)
