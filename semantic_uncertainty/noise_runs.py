"""
Run noise-before-softmax rescoring grid and produce CSV/JSONL/histograms/report.

This script expects `validation_generations.pkl` (as produced by `generate_answers.py`) to
contain a dict of generations keyed by example id. Each entry should include:
  - 'most_likely_answer' and 'responses' where 'responses' is a list of tuples
    (answer_text, token_log_likelihoods, embedding, acc, step_logits, gen_ids)

The script computes SE_before and then, for each (mu,sigma) in the noise grid,
performs R repeats of rescores using raw per-step logits (no re-generation).

Outputs written to runs/<timestamp>/ as specified in the spec.

"""
import argparse
import json
import os
import datetime
import pathlib
import platform
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from uncertainty.utils import utils
from uncertainty.uncertainty_measures.semantic_entropy import (
    get_semantic_ids, logsumexp_by_id, predictive_entropy_rao, EntailmentDeberta)
from uncertainty.uncertainty.se_rescore import se_run_from_logits, compute_baseline_se
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)

# Config (can be parameterized later)
N_QUESTIONS = 400
K = 10
R = 100
NOISE_MUS = [0.0, 0.5, 1.0, 2.0]
NOISE_SIGS = [0.5, 1.0, 2.0]
GLOBAL_SEED = 42

# Files to read (assumes generate_answers wrote this)
GENERATIONS_FILE = 'validation_generations.pkl'


def ensure_run_dir():
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = pathlib.Path('runs') / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def mean_logprob_from_token_logs(token_logprobs):
    import numpy as _np
    return float(_np.mean([float(_np.mean(x)) if len(x) > 0 else -1e9 for x in token_logprobs]))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true', help='Run small dev run (N=2,K=3,R=5)')
    args = parser.parse_args()

    if args.dev:
        logging.info('DEV mode enabled: using reduced N/K/R for fast checks')
        global N_QUESTIONS, K, R, NOISE_MUS, NOISE_SIGS
        N_QUESTIONS = 2
        K = 3
        R = 5
        NOISE_MUS = [0.0, 0.5, 1.0, 2.0]
        NOISE_SIGS = [0.5, 1.0, 2.0]

    # Load generations
    if not os.path.exists(GENERATIONS_FILE):
        raise FileNotFoundError(f'Expected {GENERATIONS_FILE} in cwd. Run generation stage first.')

    import pickle
    with open(GENERATIONS_FILE, 'rb') as f:
        generations = pickle.load(f)

    # Set up entailment model for clustering
    entailment_model = EntailmentDeberta()

    # Prepare outputs
    baseline_rows = []
    noise_runs_rows = []
    noise_summ_rows = []
    full_jsonl_rows = []

    rng_master = np.random.default_rng(GLOBAL_SEED)

    run_dir = ensure_run_dir()

    items = list(generations.items())
    for tid, example in tqdm(items[:N_QUESTIONS], desc='Questions'):
        qid = tid
        question = example.get('question', '')
        reference = example.get('reference', {})
        # Normalize reference to a string (first text if dict-like)
        reference_text = ''
        try:
            if isinstance(reference, dict):
                if 'answers' in reference and 'text' in reference['answers']:
                    ref_list = reference['answers']['text']
                    reference_text = ref_list[0] if isinstance(ref_list, list) and len(ref_list) > 0 else str(reference)
                else:
                    reference_text = str(reference)
            else:
                reference_text = str(reference)
        except Exception:
            reference_text = str(reference)

        # most_likely_answer stored separately; full_responses are higher-temp answers
        full_responses = example.get('responses', [])

        # Extract answers_texts and token_logprobs and step logits/ids
        answers_texts = [r[0] for r in full_responses]
        per_token_logprobs = [r[1] for r in full_responses]
        # Expect that extra items were appended: step_logits (list per-step tensors) and gen ids
        step_logits_list = [r[4] if len(r) > 4 else [] for r in full_responses]
        step_ids_list = [r[5] if len(r) > 5 else [] for r in full_responses]

        # Baseline SE: cluster based on texts using entailment model
        try:
            semantic_ids = get_semantic_ids(answers_texts, model=entailment_model, strict_entailment=False, example=example)
            # Build members list from ids
            unique_ids = sorted(list(set(semantic_ids)))
            members = [[] for _ in unique_ids]
            for i, sid in enumerate(semantic_ids):
                members[sid].append(i)
        except Exception:
            # Fallback: each answer its own cluster
            members = [[i] for i in range(len(answers_texts))]
            semantic_ids = [i for i in range(len(answers_texts))]

        # seq_logps: mean token log-probs per sequence
        seq_logps = [float(np.mean(lp)) if len(lp) > 0 else -1e9 for lp in per_token_logprobs]
        # cluster probs
        if len(answers_texts) > 0:
            log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, seq_logps, agg='sum_normalized')
            SE_before = predictive_entropy_rao(log_likelihood_per_semantic_id) if len(log_likelihood_per_semantic_id) > 0 else float('nan')
        else:
            log_likelihood_per_semantic_id = []
            SE_before = float('nan')

        # Eval answer and correctness label if available
        eval_answer = example.get('most_likely_answer', {}).get('response', '')
        correct_label = int(example.get('most_likely_answer', {}).get('accuracy', 0))

        # Build baseline row
        baseline_rows.append({
            'qid': qid,
            'question': question,
            'reference': reference_text,
            'eval_answer': eval_answer,
            'correct_label': correct_label,
            'SE_before': SE_before,
            'answers_json': json.dumps(answers_texts, ensure_ascii=False)
        })

        # Run noise grid
        noise_records = []
        for mu in NOISE_MUS:
            for sigma in NOISE_SIGS:
                SE_runs = []
                for r in range(R):
                    rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
                    se_r = se_run_from_logits(step_logits_list, step_ids_list, members, mu, sigma, rng)
                    SE_runs.append(float(se_r))
                    noise_runs_rows.append({
                        'qid': qid, 'mu': mu, 'sigma': sigma, 'run': r, 'SE_run': float(se_r)
                    })

                arr = np.asarray(SE_runs, dtype=np.float64)
                se_mean = float(arr.mean())
                se_std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
                delta = se_mean - SE_before
                noise_summ_rows.append({
                    'qid': qid, 'mu': mu, 'sigma': sigma,
                    'SE_before': SE_before, 'SE_mean': se_mean, 'SE_std': se_std, 'delta_SE': delta
                })

                noise_records.append({
                    'mu': mu, 'sigma': sigma, 'SE_runs': [float(x) for x in SE_runs],
                    'SE_mean': se_mean, 'SE_std': se_std, 'delta_SE': delta
                })

        full_jsonl_rows.append({
            'qid': qid,
            'question': question,
            'reference': reference_text,
            'eval_answer': eval_answer,
            'correct_label': correct_label,
            'answers': answers_texts,
            'SE_before': SE_before,
            'noise': noise_records
        })

        # Free memory for this Q
        for lst in (step_logits_list, step_ids_list):
            try:
                del lst[:]
            except Exception:
                pass

    # Write outputs
    pd.DataFrame(baseline_rows).to_csv(run_dir / 'baseline.csv', index=False)
    pd.DataFrame(noise_runs_rows).to_csv(run_dir / 'noise_runs.csv', index=False)
    pd.DataFrame(noise_summ_rows).to_csv(run_dir / 'noise_summary.csv', index=False)

    with open(run_dir / 'results.jsonl', 'w', encoding='utf-8') as f:
        for row in full_jsonl_rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    config = {
        'N': int(N_QUESTIONS), 'K': int(K), 'R': int(R),
        'noise_means': NOISE_MUS, 'noise_stds': NOISE_SIGS,
        'seed': int(GLOBAL_SEED),
        'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
        'python': platform.python_version(),
        'torch': torch.__version__
    }
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Make histograms for default (mu=0.0, sigma=1.0)
    msel, ssel = 0.0, 1.0
    summ = pd.DataFrame(noise_summ_rows)
    base = pd.DataFrame(baseline_rows)
    merged = summ.query('mu==@msel and sigma==@ssel')[['qid','SE_mean']].merge(base[['qid','SE_before']], on='qid', how='left')

    plt.figure()
    merged['SE_before'].hist(bins=40)
    plt.xlabel('SE (before noise)')
    plt.ylabel('Number of questions')
    plt.title('SE_before distribution')
    plt.tight_layout()
    plt.savefig(run_dir / 'hist_se_before.png')
    plt.close()

    plt.figure()
    merged['SE_mean'].hist(bins=40)
    plt.xlabel('Mean SE (after noise)')
    plt.ylabel('Number of questions')
    plt.title(f'SE_mean distribution (μ={msel}, σ={ssel})')
    plt.tight_layout()
    plt.savefig(run_dir / 'hist_se_mean.png')
    plt.close()

    plt.figure()
    (merged['SE_mean'] - merged['SE_before']).hist(bins=40)
    plt.xlabel('ΔSE = SE_after_mean − SE_before')
    plt.ylabel('Number of questions')
    plt.title(f'ΔSE distribution (μ={msel}, σ={ssel})')
    plt.tight_layout()
    plt.savefig(run_dir / 'hist_delta_se.png')
    plt.close()

    # Build PDF report
    with PdfPages(run_dir / 'report.pdf') as pdf:
        # Front page
        plt.figure(figsize=(8.5, 11)); plt.axis('off')
        plt.title('Semantic Entropy Stability Report', pad=20)
        txt = (f"N={N_QUESTIONS}, K={K}, R={R}\n"
               f"Noise μ∈{NOISE_MUS}, σ∈{NOISE_SIGS}\n"
               f"Seed: {GLOBAL_SEED}\n")
        plt.text(0.05, 0.9, txt, fontsize=11, va='top')
        pdf.savefig(); plt.close()

        for row in full_jsonl_rows:
            plt.figure(figsize=(8.5, 11)); plt.axis('off')
            y = 0.97
            def add(s, fs=9, dy=0.03):
                nonlocal y
                plt.text(0.05, y, s, fontsize=fs, va='top'); y -= dy

            add(f"QID: {row['qid']}", fs=10, dy=0.035)
            add(f"Question: {row['question']}")
            add(f"Reference: {row['reference']}")
            add(f"Eval answer: {row['eval_answer']}   |   Label(correct)={row['correct_label']}")
            add(f"SE_before: {row['SE_before']:.6f}", fs=10, dy=0.035)

            add('Answers (K=10):', fs=10)
            for i, a in enumerate(row['answers'], 1):
                add(f"{i:02d}. {a}")

            for rec in row['noise']:
                mu, sigma = rec['mu'], rec['sigma']
                se_mean, se_std, delta = rec['SE_mean'], rec['SE_std'], rec['delta_SE']
                add(f"\nNoise (μ={mu}, σ={sigma})  →  SE_mean={se_mean:.6f}  SE_std={se_std:.6f}  ΔSE={delta:.6f}", fs=10, dy=0.035)
                runs = rec['SE_runs']
                for i in range(0, len(runs), 10):
                    add('  ' + '  '.join(f"{x:.4f}" for x in runs[i:i+10]), fs=8, dy=0.022)

            pdf.savefig(); plt.close()

        # Append histograms
        for fname in ['hist_se_before.png','hist_se_mean.png','hist_delta_se.png']:
            img = plt.imread(run_dir / fname)
            plt.figure(figsize=(8.5, 11)); plt.imshow(img); plt.axis('off')
            pdf.savefig(); plt.close()

    logging.info('Wrote run outputs to %s', run_dir)


if __name__ == '__main__':
    main()
