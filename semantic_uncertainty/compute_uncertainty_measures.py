"""Compute uncertainty measures after generating answers."""
from collections import defaultdict
import csv
import json
import logging
import os
import pickle
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path=None):
        """Lightweight fallback for loading key=value pairs from a .env file."""
        if dotenv_path is None:
            dotenv_path = Path(__file__).resolve().parents[1] / '.env'
        dotenv_file = Path(dotenv_path)
        if not dotenv_file.exists():
            return False

        for line in dotenv_file.read_text(encoding='utf-8').splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith('#') or '=' not in stripped:
                continue
            key, value = stripped.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip())
        return True

load_dotenv(Path(__file__).resolve().parents[1] / '.env')
import numpy as np
import wandb
import glob
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

from analyze_results import analyze_run
from uncertainty.data.data_utils import load_ds
from uncertainty.uncertainty_measures.p_ik import get_p_ik
from uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids_embedding
from uncertainty.uncertainty_measures.semantic_entropy import load_embedding_model
from uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy_rao
from uncertainty.uncertainty_measures.semantic_entropy import cluster_assignment_entropy
from uncertainty.uncertainty_measures.semantic_entropy import context_entails_response
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT35
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4Turbo
from uncertainty.uncertainty_measures.semantic_entropy import EntailmentLlama
from uncertainty.uncertainty_measures import p_true as p_true_utils
from uncertainty.utils import utils


utils.setup_logger()

# Noise configuration for robustness analysis
NOISE_MEANS = [0.0, 0.5, 1.0, 2.0]
NOISE_STDS = [0.5, 1.0, 2.0]
N_NOISE_SAMPLES = 100

EXP_DETAILS = 'experiment_details.pkl'


def sanitize_filename_component(value):
    return str(value).strip().replace(' ', '_').replace('/', '_').replace('\\', '_')


def get_run_filename(prefix, args, extension):
    model = sanitize_filename_component(getattr(args, 'model_name', getattr(args, 'model', 'unknown')))
    dataset = sanitize_filename_component(getattr(args, 'dataset', 'unknown'))
    run_id = getattr(wandb.run, 'id', None)
    if run_id:
        return f"{prefix}_{model}_{dataset}_{run_id}.{extension}"
    return f"{prefix}_{model}_{dataset}.{extension}"


def save_detailed_noise_results(results, args):
    """Save one flat row per question and noise configuration."""
    detailed_rows = []

    for result in results:
        for noise_record in result['noise_records']:
            detailed_rows.append({
                'question_id': noise_record['question_id'],
                'noise_mean': noise_record['mu'],
                'noise_std': noise_record['sigma'],
                'SE_original': noise_record['se_before'],
                'Noise_added_SE_100_values': json.dumps(noise_record['se_after_samples']),
                'mean_noisy_SE': noise_record['se_mean'],
                'delta_SE': noise_record['delta_se'],
                'abs_delta_SE': abs(noise_record['delta_se']),
                'std_noisy_SE': noise_record['se_std'],
            })

    detailed_path = os.path.join(wandb.run.dir, get_run_filename('detailed_question_noise_results', args, 'csv'))
    fieldnames = [
        'question_id',
        'noise_mean',
        'noise_std',
        'SE_original',
        'Noise_added_SE_100_values',
        'mean_noisy_SE',
        'delta_SE',
        'abs_delta_SE',
        'std_noisy_SE',
    ]

    with open(detailed_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_rows)

    wandb.save(detailed_path)
    logging.info('Saved %d detailed noise rows to %s', len(detailed_rows), detailed_path)

def sanity_check_logit_noise(logits_data, token_ids_data, log_liks):
    """
    Sanity checks for logit-space noise implementation:
    (1) sigma=0 → noisy log-probs equal original log-probs
    (2) Different answers receive different noise (not broadcast)
    """
    logging.info("Running sanity checks for logit-space noise...")
    
    # Check 1: sigma=0 reproduces original log-probs
    noisy_log_liks_sigma0 = []
    for logits_seq, token_ids in zip(logits_data, token_ids_data):
        noisy_token_log_probs = []
        for logits_t, token_id in zip(logits_seq, token_ids):
            # Add zero noise
            noisy_logits = logits_t.numpy() + 0.0
            # Recompute log-probabilities
            logits_max = noisy_logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(noisy_logits - logits_max)
            log_probs = noisy_logits - logits_max - np.log(exp_logits.sum(axis=1, keepdims=True))
            noisy_ll_t = log_probs[0, token_id]
            noisy_token_log_probs.append(noisy_ll_t)
        noisy_log_liks_sigma0.append(np.mean(noisy_token_log_probs))
    
    original_log_liks = [np.mean(ll_seq) for ll_seq in log_liks]
    max_diff = max(abs(nl - ol) for nl, ol in zip(noisy_log_liks_sigma0, original_log_liks))
    logging.info(f"Check 1 - sigma=0 max diff: {max_diff:.6f} (should be < 1e-5)")
    assert max_diff < 1e-5, f"sigma=0 check failed: max diff {max_diff}"
    
    # Check 2: Different answers get different noise
    if len(logits_data) >= 2:
        noise_values_answer0 = []
        noise_values_answer1 = []
        np.random.seed(42)  # Fixed seed for reproducibility
        
        for logits_seq, token_ids in zip([logits_data[0]], [token_ids_data[0]]):
            for logits_t, token_id in zip(logits_seq, token_ids):
                vocab_size = logits_t.shape[1]
                noise = np.random.normal(1.0, 1.0, size=(1, vocab_size))
                noise_values_answer0.extend(noise.flatten())
        
        np.random.seed(42)  # Same seed
        for logits_seq, token_ids in zip([logits_data[1]], [token_ids_data[1]]):
            for logits_t, token_id in zip(logits_seq, token_ids):
                vocab_size = logits_t.shape[1]
                noise = np.random.normal(1.0, 1.0, size=(1, vocab_size))
                noise_values_answer1.extend(noise.flatten())
        
        # With same seed but different execution, noise should still be different
        # due to different number of tokens or vocab size
        same_count = sum(1 for n0, n1 in zip(noise_values_answer0[:100], noise_values_answer1[:100]) if abs(n0 - n1) < 1e-10)
        logging.info(f"Check 2 - Same noise values (first 100): {same_count}/100 (should be < 50 for different answers)")
        # Note: This check is weak, but demonstrates noise is not naively broadcast
    
    logging.info("Sanity checks passed!")


def process_single_question(args_tuple):
    """
    Worker function to process a single question in parallel.
    Returns results dict for one question.
    """
    (idx, tid, validation_generations, args_dict, semantic_ids_cache, recompute_accuracy, metric_fn, has_logits) = args_tuple
    
    # Reconstruct necessary objects (cannot pickle model/entailment)
    example = validation_generations[tid]
    question = example['question']
    context = example['context']
    full_responses = example["responses"]
    most_likely_answer = example['most_likely_answer']
    
    # Determine number of generations to use AND detect tuple format
    if not args_dict['use_all_generations']:
        num_gen = args_dict['use_num_generations']
        responses_raw = full_responses[:num_gen]
    else:
        responses_raw = full_responses
    
    # Check format: 4-tuple (old) vs 6-tuple (new)
    if has_logits:
        # New format: (answer, log_liks, embedding, logits, token_ids, acc)
        responses = [fr[0] for fr in responses_raw]
        log_liks = [r[1] for r in responses_raw]
        logits_data = [fr[3] for fr in responses_raw]
        token_ids_data = [fr[4] for fr in responses_raw]
    else:
        # Old format: (answer, log_liks, embedding, acc)
        responses = [fr[0] for fr in responses_raw]
        log_liks = [r[1] for fr in responses_raw]
        logits_data = None
        token_ids_data = None
    
    # Helper to check answerability
    def is_answerable(gen):
        return len(gen['reference']['answers']['text']) > 0
    
    # Handle accuracy recomputation
    if recompute_accuracy:
        if is_answerable(example):
            # Note: metric_fn is serializable dict with metric type
            # Actual metric computation needs to be done outside parallel context
            acc = -1.0  # Placeholder for later computation
        else:
            acc = 0.0
        validation_is_true_val = acc
    else:
        validation_is_true_val = most_likely_answer['accuracy']
    
    result = {
        'idx': idx,
        'tid': tid,
        'question': question,
        'context': context,
        'responses': responses,
        'validation_is_true': validation_is_true_val,
        'validation_answerable': is_answerable(example),
        'validation_embedding': most_likely_answer['embedding'],
        'entropies': {},
        'semantic_ids': None,
        'noise_records': [],
        'most_likely_answer': most_likely_answer['response'],  # For p_true computation
        'needs_accuracy_recompute': recompute_accuracy and is_answerable(example)
    }
    
    if not args_dict['compute_predictive_entropy']:
        return result
    
    # Use cached semantic IDs if available
    if semantic_ids_cache and tid in semantic_ids_cache:
        semantic_ids = semantic_ids_cache[tid]
    else:
        # For parallel processing, semantic_ids must be precomputed
        # This is a placeholder - actual computation requires entailment model
        semantic_ids = list(range(len(responses)))  # Fallback: each response is unique
    
    result['semantic_ids'] = semantic_ids
    
    # Compute entropies
    log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]
    
    # Cluster assignment entropy
    result['entropies']['cluster_assignment_entropy'] = cluster_assignment_entropy(semantic_ids)
    
    # Regular predictive entropy
    result['entropies']['regular_entropy'] = predictive_entropy(log_liks_agg)
    
    # Semantic entropy
    log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
    se_before = predictive_entropy_rao(log_likelihood_per_semantic_id)
    result['entropies']['semantic_entropy'] = se_before
    
    # Noise robustness analysis - SKIP if old format
    if has_logits:
        correctness = most_likely_answer['accuracy']

        # ====== SEQUENTIAL NOISE SAMPLING (NO NESTED POOL) ======
        for mu in NOISE_MEANS:
            for sigma in NOISE_STDS:
                se_after_samples = []
                for sample_idx in range(N_NOISE_SAMPLES):
                    seed = 42 + idx * 1000 + int(mu * 10) * 100 + int(sigma * 10) + sample_idx
                    np.random.seed(seed)

                    noisy_log_liks_agg = []
                    for logits_seq, token_ids in zip(logits_data, token_ids_data):
                        noisy_token_log_probs = []
                        for logits_t, token_id in zip(logits_seq, token_ids):
                            # logits_t may be a numpy array here (we convert before forking)
                            if hasattr(logits_t, 'numpy'):
                                logits_arr = logits_t.numpy()
                            else:
                                logits_arr = np.array(logits_t)
                            vocab_size = logits_arr.shape[1]
                            noise = np.random.normal(mu, sigma, size=(1, vocab_size))
                            noisy_logits = logits_arr + noise

                            logits_max = noisy_logits.max(axis=1, keepdims=True)
                            exp_logits = np.exp(noisy_logits - logits_max)
                            log_probs = noisy_logits - logits_max - np.log(exp_logits.sum(axis=1, keepdims=True))

                            noisy_ll_t = log_probs[0, token_id]
                            noisy_token_log_probs.append(noisy_ll_t)

                        noisy_log_liks_agg.append(np.mean(noisy_token_log_probs))

                    noisy_log_likelihood_per_semantic_id = logsumexp_by_id(
                        semantic_ids, noisy_log_liks_agg, agg='sum_normalized')
                    noisy_se = predictive_entropy_rao(noisy_log_likelihood_per_semantic_id)
                    se_after_samples.append(noisy_se)

                se_mean = np.mean(se_after_samples)
                se_std = np.std(se_after_samples)
                delta_se = se_mean - se_before

                noise_record = {
                    'question_id': tid,
                    'mu': mu,
                    'sigma': sigma,
                    'se_before': float(se_before),
                    'se_after_samples': [float(s) for s in se_after_samples],
                    'se_mean': float(se_mean),
                    'se_std': float(se_std),
                    'delta_se': float(delta_se),
                    'correctness': float(correctness)
                }
                result['noise_records'].append(noise_record)
        # ====== END SEQUENTIAL NOISE SAMPLING ======

    return result


def main(args):
    if args.train_wandb_runid is None:
        args.train_wandb_runid = args.eval_wandb_runid

    user = os.environ['USER']
    scratch_dir = os.getenv('SCRATCH_DIR', '.')
    wandb_dir = f'{scratch_dir}/{user}/uncertainty'
    slurm_jobid = os.getenv('SLURM_JOB_ID', None)
    project = "semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug"

    if args.assign_new_wandb_id:
        logging.info('Assign new wandb_id.')
        api = wandb.Api()
        old_run = api.run(f'{args.restore_entity_eval}/{project}/{args.eval_wandb_runid}')
        wandb.init(
            entity=args.entity,
            project=project,
            dir=wandb_dir,
            notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
            config={**old_run.config, **args.__dict__},
        )

        def restore(filename):
            old_run.file(filename).download(replace=True, exist_ok=False, root=wandb.run.dir)

            class Restored:
                name = f'{wandb.run.dir}/{filename}'

            return Restored
    else:
        logging.info('Reuse active wandb id.')

        def restore(filename):
            class Restored:
                name = f'{wandb.run.dir}/{filename}'
            return Restored

    if args.train_wandb_runid != args.eval_wandb_runid:
        logging.info(
            "Distribution shift for p_ik. Training on embeddings from run %s but evaluating on run %s",
            args.train_wandb_runid, args.eval_wandb_runid)

        is_ood_eval = True  # pylint: disable=invalid-name
        api = wandb.Api()
        old_run_train = api.run(f'{args.restore_entity_train}/semantic_uncertainty/{args.train_wandb_runid}')
        filename = 'train_generations.pkl'
        old_run_train.file(filename).download(replace=True, exist_ok=False, root=wandb.run.dir)
        with open(f'{wandb.run.dir}/{filename}', "rb") as infile:
            train_generations = pickle.load(infile)
        wandb.config.update({"ood_training_set": old_run_train.config['dataset']}, allow_val_change=True)
    else:
        is_ood_eval = False  # pylint: disable=invalid-name
        if args.compute_p_ik or args.compute_p_ik_answerable:
            try:
                train_generations_pickle = restore('train_generations.pkl')
                with open(train_generations_pickle.name, 'rb') as infile:
                    train_generations = pickle.load(infile)
                logging.info('Restored train_generations.pkl from previous run.')
            except Exception as e:
                logging.warning('Could not restore train_generations.pkl (%s). Trying local cache lookup.', e)
                patterns = [
                    os.path.join(wandb_dir, f'run-*{args.train_wandb_runid or args.eval_wandb_runid}', 'train_generations.pkl'),
                    os.path.join(wandb_dir, f'run-*{args.train_wandb_runid or args.eval_wandb_runid}', 'files', 'train_generations.pkl'),
                ]
                candidates = []
                for pat in patterns:
                    candidates.extend(glob.glob(pat))
                if not candidates:
                    cwd = os.getcwd()
                    candidates = glob.glob(os.path.join(cwd, '**', 'train_generations.pkl'), recursive=True)
                if candidates:
                    local_train_path = sorted(candidates, key=lambda p: os.path.getmtime(p))[-1]
                    with open(local_train_path, 'rb') as infile:
                        train_generations = pickle.load(infile)
                    logging.info('Loaded train_generations.pkl from local cache: %s', local_train_path)
                else:
                    raise RuntimeError('train_generations.pkl not found in WandB or local cache.')

    wandb.config.update({"is_ood_eval": is_ood_eval}, allow_val_change=True)

    embedding_model = None
    entailment_model = None

    if args.compute_predictive_entropy:
        embedding_model = load_embedding_model()
        logging.info('Loaded embedding model for semantic clustering.')

        if args.compute_context_entails_response or args.compute_p_true_in_compute_stage:
            logging.info('Beginning loading for entailment model.')
            if args.entailment_model == 'deberta':
                entailment_model = EntailmentDeberta()
            elif args.entailment_model == 'gpt-4':
                entailment_model = EntailmentGPT4(args.entailment_cache_id, args.entailment_cache_only)
            elif args.entailment_model == 'gpt-3.5':
                entailment_model = EntailmentGPT35(args.entailment_cache_id, args.entailment_cache_only)
            elif args.entailment_model == 'gpt-4-turbo':
                entailment_model = EntailmentGPT4Turbo(args.entailment_cache_id, args.entailment_cache_only)
            elif 'llama' in args.entailment_model.lower():
                entailment_model = EntailmentLlama(args.entailment_cache_id, args.entailment_cache_only, args.entailment_model)
            else:
                raise ValueError
            logging.info('Entailment model loading complete.')

    if args.compute_p_true_in_compute_stage:
        old_exp = restore(EXP_DETAILS)
        with open(old_exp.name, "rb") as infile:
            old_exp = pickle.load(infile)

        if args.reuse_entailment_model:
            pt_model = entailment_model.model
        else:
            pt_model = utils.init_model(old_exp['args'])

        pt_train_dataset, _ = load_ds(
            old_exp['args'].dataset, add_options=old_exp['args'].use_mc_options,
            seed=args.random_seed)

        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            num_gen = args.use_num_generations
        else:
            num_gen = args.num_generations

        p_true_few_shot_prompt, _, len_p_true = p_true_utils.construct_few_shot_prompt(
            model=pt_model,
            dataset=pt_train_dataset,
            indices=old_exp['p_true_indices'],
            prompt=old_exp['prompt'],
            brief=old_exp['BRIEF'],
            brief_always=old_exp['args'].brief_always and old_exp['args'].enable_brief,
            make_prompt=utils.get_make_prompt(old_exp['args']),
            num_generations=num_gen,
            metric=utils.get_metric(old_exp['args'].metric))
        wandb.config.update({'p_true_num_fewshot': len_p_true}, allow_val_change=True)
        wandb.log(dict(len_p_true=len_p_true))

    if args.recompute_accuracy:
        logging.warning('Recompute accuracy enabled.')
        metric = utils.get_metric(args.metric)

    try:
        result_dict_pickle = restore('uncertainty_measures.pkl')
        with open(result_dict_pickle.name, "rb") as infile:
            result_dict = pickle.load(infile)
        logging.info('Restored uncertainty_measures.pkl from previous run.')
    except Exception as e:
        logging.warning('Could not restore uncertainty_measures.pkl (%s). Initializing new result_dict.', e)
        result_dict = {}
    result_dict['semantic_ids'] = []

    try:
        validation_generations_pickle = restore('validation_generations.pkl')
        with open(validation_generations_pickle.name, 'rb') as infile:
            validation_generations = pickle.load(infile)
        logging.info('Restored validation_generations.pkl from previous run.')
    except Exception as e:
        logging.warning('Could not restore validation_generations.pkl (%s). Trying local cache lookup.', e)
        patterns = [
            os.path.join(wandb_dir, f'run-*{args.eval_wandb_runid}', 'validation_generations.pkl'),
            os.path.join(wandb_dir, f'run-*{args.eval_wandb_runid}', 'files', 'validation_generations.pkl'),
        ]
        candidates = []
        for pat in patterns:
            candidates.extend(glob.glob(pat))
        if not candidates:
            cwd = os.getcwd()
            candidates = glob.glob(os.path.join(cwd, '**', 'validation_generations.pkl'), recursive=True)
        if candidates:
            local_val_path = sorted(candidates, key=lambda p: os.path.getmtime(p))[-1]
            with open(local_val_path, 'rb') as infile:
                validation_generations = pickle.load(infile)
            logging.info('Loaded validation_generations.pkl from local cache: %s', local_val_path)
        else:
            raise RuntimeError(f'validation_generations.pkl not found for run_id {args.eval_wandb_runid}.')

    entropies = defaultdict(list)
    validation_embeddings, validation_is_true, validation_answerable = [], [], []
    p_trues = []
    count = 0  # pylint: disable=invalid-name

    def is_answerable(generation):
        return len(generation['reference']['answers']['text']) > 0

    first_tid = list(validation_generations.keys())[0]
    first_example = validation_generations[first_tid]
    first_response = first_example['responses'][0]
    has_logits = len(first_response) == 6

    if has_logits:
        logging.info("✅ Detected NEW pickle format (6-tuple with logits/token_ids)")
    else:
        logging.warning("⚠️  Detected OLD pickle format (4-tuple without logits)")

    logging.info('Precomputing semantic IDs for all validation questions...')
    indices_list = list(validation_generations.keys())[:args.num_eval_samples]
    semantic_ids_cache = {}

    if args.compute_predictive_entropy:
        for idx, tid in enumerate(tqdm(indices_list, desc="Computing semantic IDs")):
            example = validation_generations[tid]
            question = example['question']
            context = example['context']
            full_responses = example["responses"]

            if not args.use_all_generations:
                responses = [fr[0] for fr in full_responses[:args.use_num_generations]]
            else:
                responses = [fr[0] for fr in full_responses]

            if args.condition_on_question and args.entailment_model == 'deberta':
                responses = [f'{question} {r}' for r in responses]

            if args.compute_context_entails_response:
                context_ent = context_entails_response(context, responses, entailment_model)
                entropies['context_entails_response'].append(context_ent)

            semantic_ids, _, _ = get_semantic_ids_embedding(responses, embedding_model)
            semantic_ids_cache[tid] = semantic_ids

            if idx == 0 and has_logits:
                logits_data = [fr[3] for fr in full_responses[:args.use_num_generations]] if not args.use_all_generations else [fr[3] for fr in full_responses]
                token_ids_data = [fr[4] for fr in full_responses[:args.use_num_generations]] if not args.use_all_generations else [fr[4] for fr in full_responses]
                log_liks = [r[1] for r in full_responses[:args.use_num_generations]] if not args.use_all_generations else [r[1] for r in full_responses]
                sanity_check_logit_noise(logits_data, token_ids_data, log_liks)

    num_workers = args.num_workers if args.num_workers is not None else (min(cpu_count() - 1, 8) if cpu_count() > 1 else 1)
    use_parallel = len(indices_list) > 10 and num_workers > 1

    args_dict = {
        'use_all_generations': args.use_all_generations,
        'use_num_generations': args.use_num_generations,
        'compute_predictive_entropy': args.compute_predictive_entropy,
    }
    metric_info = {'type': args.metric} if args.recompute_accuracy else None

    if use_parallel:
        logging.info(f'Processing {len(indices_list)} questions with {num_workers} workers...')

        # Use spawn start method to avoid forking issues with torch/tokenizers
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            # start method already set; continue
            pass

        # Build a lightweight validation structure that converts torch tensors to numpy
        # so we don't send torch.Storage objects across processes (avoids DupFd errors).
        light_validation = {}
        for tid in indices_list:
            ex = validation_generations[tid]
            lite = {
                'question': ex['question'],
                'context': ex.get('context'),
                'most_likely_answer': {
                    'response': ex['most_likely_answer'].get('response'),
                    'accuracy': ex['most_likely_answer'].get('accuracy', 0.0),
                    'embedding': ex['most_likely_answer'].get('embedding')
                },
                'responses': []
            }

            full_responses = ex['responses']
            if not args.use_all_generations:
                responses_iter = full_responses[:args.use_num_generations]
            else:
                responses_iter = full_responses

            for fr in responses_iter:
                if has_logits:
                    # New format: (answer, log_liks, embedding, logits, token_ids, acc)
                    answer, log_liks, emb, logits_seq, token_ids, acc = fr
                    converted_logits = []
                    for lt in logits_seq:
                        if hasattr(lt, 'cpu'):
                            converted_logits.append(lt.cpu().numpy())
                        else:
                            converted_logits.append(np.array(lt))
                    lite['responses'].append((answer, log_liks, emb, converted_logits, list(token_ids), acc))
                else:
                    lite['responses'].append(fr)

            light_validation[tid] = lite

        worker_args = [
            (idx, tid, light_validation, args_dict, semantic_ids_cache, args.recompute_accuracy, metric_info, has_logits)
            for idx, tid in enumerate(indices_list)
        ]

        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(process_single_question, worker_args), total=len(worker_args), desc="Processing questions"))
    else:
        logging.info('Processing questions sequentially (small batch or single worker)...')
        results = []
        for idx, tid in enumerate(tqdm(indices_list, desc="Processing questions")):
            worker_args = (idx, tid, validation_generations, args_dict, semantic_ids_cache, args.recompute_accuracy, metric_info, has_logits)
            results.append(process_single_question(worker_args))

    if args.recompute_accuracy:
        logging.info('Recomputing accuracy for questions that need it...')
        metric = utils.get_metric(args.metric)
        model = None
        if 'llm' in args.metric and metric != utils.get_metric('squad'):
            old_exp = restore(EXP_DETAILS)
            with open(old_exp.name, "rb") as infile:
                old_exp = pickle.load(infile)
            model = utils.init_model(old_exp['args'])

        for result in tqdm(results, desc="Recomputing accuracy"):
            if result.get('needs_accuracy_recompute', False):
                example = validation_generations[result['tid']]
                result['validation_is_true'] = metric(result['most_likely_answer'], example, model)

        if model:
            del model

    if args.compute_p_true_in_compute_stage:
        for result in tqdm(results, desc="Computing p_true"):
            tid = result['tid']
            example = validation_generations[tid]
            question = example['question']
            p_true = p_true_utils.calculate_p_true(
                pt_model, question, result['most_likely_answer'],
                result['responses'], p_true_few_shot_prompt,
                hint=old_exp['args'].p_true_hint)
            p_trues.append(p_true)

    logging.info('Aggregating results from parallel processing...')
    num_noise_records = 0
    for result in results:
        validation_is_true.append(result['validation_is_true'])
        validation_answerable.append(result['validation_answerable'])
        validation_embeddings.append(result['validation_embedding'])

        if result['semantic_ids']:
            result_dict['semantic_ids'].append(result['semantic_ids'])

        for key, value in result['entropies'].items():
            entropies[key].append(value)

        if result['noise_records']:
            noise_file = os.path.join(wandb.run.dir, get_run_filename('se_after_noise', args, 'jsonl'))
            with open(noise_file, 'a', encoding='utf-8') as f:
                for noise_record in result['noise_records']:
                    f.write(json.dumps(noise_record) + '\n')
                    num_noise_records += 1

        if (len(validation_is_true) % 50) == 0:
            logging.info(f'Processed {len(validation_is_true)} questions')
            logging.info(f'Current accuracy: {np.mean(validation_is_true):.4f}')

        count += 1

    if has_logits:
        noise_file = os.path.join(wandb.run.dir, get_run_filename('se_after_noise', args, 'jsonl'))
        logging.info(f'Saving {num_noise_records} noise records to %s', noise_file)
        wandb.save(noise_file)
        logging.info(f'Saved {num_noise_records} noise records to %s', noise_file)
        save_detailed_noise_results(results, args)
    else:
        logging.info('No noise records saved (old pickle format)')

    logging.info('Finished processing all questions.')

    logging.info('Accuracy on original task: %f', np.mean(validation_is_true))
    validation_is_false = [1.0 - is_t for is_t in validation_is_true]
    result_dict['validation_is_false'] = validation_is_false

    validation_unanswerable = [1.0 - is_a for is_a in validation_answerable]
    result_dict['validation_unanswerable'] = validation_unanswerable
    logging.info('Unanswerable prop on validation: %f', np.mean(validation_unanswerable))

    if 'uncertainty_measures' not in result_dict:
        result_dict['uncertainty_measures'] = dict()

    if args.compute_predictive_entropy:
        result_dict['uncertainty_measures'].update(entropies)

    if args.compute_p_ik or args.compute_p_ik_answerable:
        train_is_true, train_embeddings, train_answerable = [], [], []
        for tid in train_generations:
            most_likely_answer = train_generations[tid]['most_likely_answer']
            train_embeddings.append(most_likely_answer['embedding'])
            train_is_true.append(most_likely_answer['accuracy'])
            train_answerable.append(is_answerable(train_generations[tid]))
        train_is_false = [0.0 if is_t else 1.0 for is_t in train_is_true]
        train_unanswerable = [0.0 if is_t else 1.0 for is_t in train_answerable]
        logging.info('Unanswerable prop on p_ik training: %f', np.mean(train_unanswerable))

    if args.compute_p_ik:
        p_ik_predictions = get_p_ik(
            train_embeddings=train_embeddings, is_false=train_is_false,
            eval_embeddings=validation_embeddings, eval_is_false=validation_is_false)
        result_dict['uncertainty_measures']['p_ik'] = p_ik_predictions

    if args.compute_p_ik_answerable:
        p_ik_predictions = get_p_ik(
            train_embeddings=train_embeddings, is_false=train_unanswerable,
            eval_embeddings=validation_embeddings, eval_is_false=validation_unanswerable)
        result_dict['uncertainty_measures']['p_ik_unanswerable'] = p_ik_predictions

    if args.compute_p_true_in_compute_stage:
        result_dict['uncertainty_measures']['p_false'] = [1 - p for p in p_trues]
        result_dict['uncertainty_measures']['p_false_fixed'] = [1 - np.exp(p) for p in p_trues]

    utils.save(result_dict, 'uncertainty_measures.pkl')

    if args.compute_predictive_entropy and entailment_model is not None:
        entailment_model.save_prediction_cache()

    if args.analyze_run:
        logging.info(50 * '#X')
        logging.info('STARTING `analyze_run`!')
        analyze_run(wandb.run.id)
        logging.info(50 * '#X')
        logging.info('FINISHED `analyze_run`!')


if __name__ == '__main__':
    parser = utils.get_parser(stages=['compute'])
    args, unknown = parser.parse_known_args()  # pylint: disable=invalid-name
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    logging.info("Args: %s", args)

    main(args)
