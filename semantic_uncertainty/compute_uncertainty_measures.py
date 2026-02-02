"""Compute uncertainty measures after generating answers."""
from collections import defaultdict
import json
import logging
import os
import pickle
import numpy as np
import wandb
import glob

from analyze_results import analyze_run
from uncertainty.data.data_utils import load_ds
from uncertainty.uncertainty_measures.p_ik import get_p_ik
from uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids
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
            # For convenience, keep any 'generate_answers' configs from old run,
            # but overwrite the rest!
            # NOTE: This means any special configs affecting this script must be
            # called again when calling this script!
            config={**old_run.config, **args.__dict__},
        )

        def restore(filename):
            old_run.file(filename).download(
                replace=True, exist_ok=False, root=wandb.run.dir)

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
        old_run_train.file(filename).download(
            replace=True, exist_ok=False, root=wandb.run.dir)
        with open(f'{wandb.run.dir}/{filename}', "rb") as infile:
            train_generations = pickle.load(infile)
        wandb.config.update(
            {"ood_training_set": old_run_train.config['dataset']}, allow_val_change=True)
    else:
        is_ood_eval = False  # pylint: disable=invalid-name
        if args.compute_p_ik or args.compute_p_ik_answerable:
            # Try restoring from WandB, fallback to local cache
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
                    raise RuntimeError('train_generations.pkl not found in WandB or local cache. Ensure generate_answers.py produced training generations for p_ik.')

    wandb.config.update({"is_ood_eval": is_ood_eval}, allow_val_change=True)

    # Load entailment model.
    if args.compute_predictive_entropy:
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
        # This is usually not called.
        old_exp = restore(EXP_DETAILS)
        with open(old_exp.name, "rb") as infile:
            old_exp = pickle.load(infile)

        if args.reuse_entailment_model:
            pt_model = entailment_model.model
        else:
            pt_model = utils.init_model(old_exp['args'])

        pt_train_dataset, pt_validation_dataset = load_ds(
            old_exp['args'].dataset, add_options=old_exp['args'].use_mc_options,
            seed=args.random_seed)
        del pt_validation_dataset

        # Reduce num generations used in p_true if needed!
        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            num_gen = args.use_num_generations
        else:
            num_gen = args.num_generations

        p_true_few_shot_prompt, p_true_responses, len_p_true = p_true_utils.construct_few_shot_prompt(
            model=pt_model,
            dataset=pt_train_dataset,
            indices=old_exp['p_true_indices'],
            prompt=old_exp['prompt'],
            brief=old_exp['BRIEF'],
            brief_always=old_exp['args'].brief_always and old_exp['args'].enable_brief,
            make_prompt=utils.get_make_prompt(old_exp['args']),
            num_generations=num_gen,
            metric=utils.get_metric(old_exp['args'].metric))
        del p_true_responses
        wandb.config.update(
            {'p_true_num_fewshot': len_p_true}, allow_val_change=True)
        wandb.log(dict(len_p_true=len_p_true))

        logging.info('Generated few-shot prompt for p_true.')
        logging.info(80*'#')
        logging.info('p_true_few_shot_prompt: %s', p_true_few_shot_prompt)
        logging.info(80*'#')

    if args.recompute_accuracy:
        # This is usually not enabled.
        logging.warning('Recompute accuracy enabled. This does not apply to precomputed p_true!')
        metric = utils.get_metric(args.metric)

    # Restore outputs from `generate_answers.py` run.
    # Fallback: initialize fresh result_dict if file is not present in the old run.
    try:
        result_dict_pickle = restore('uncertainty_measures.pkl')
        with open(result_dict_pickle.name, "rb") as infile:
            result_dict = pickle.load(infile)
        logging.info('Restored uncertainty_measures.pkl from previous run.')
    except Exception as e:  # wandb.errors.CommError on 404
        logging.warning('Could not restore uncertainty_measures.pkl (%s). Initializing new result_dict.', e)
        result_dict = {}
    result_dict['semantic_ids'] = []

    # Restore validation generations; fallback to local cache if missing in WandB
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
            # Fallback: search project results and current working directory recursively
            cwd = os.getcwd()
            candidates = glob.glob(os.path.join(cwd, '**', 'validation_generations.pkl'), recursive=True)
        if candidates:
            # Pick the most recent file
            local_val_path = sorted(candidates, key=lambda p: os.path.getmtime(p))[-1]
            with open(local_val_path, 'rb') as infile:
                validation_generations = pickle.load(infile)
            logging.info('Loaded validation_generations.pkl from local cache: %s', local_val_path)
        else:
            raise RuntimeError(f'validation_generations.pkl not found in WandB or local cache for run_id {args.eval_wandb_runid}. Please run generate_answers.py first.')

    entropies = defaultdict(list)
    validation_embeddings, validation_is_true, validation_answerable = [], [], []
    p_trues = []
    count = 0  # pylint: disable=invalid-name

    def is_answerable(generation):
        return len(generation['reference']['answers']['text']) > 0

    # Loop over datapoints and compute validation embeddings and entropies.
    for idx, tid in enumerate(validation_generations):

        example = validation_generations[tid]
        question = example['question']
        context = example['context']
        full_responses = example["responses"]
        most_likely_answer = example['most_likely_answer']

        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            responses = [fr[0] for fr in full_responses[:args.use_num_generations]]
        else:
            responses = [fr[0] for fr in full_responses]

        if args.recompute_accuracy:
            logging.info('Recomputing accuracy!')
            if is_answerable(example):
                acc = metric(most_likely_answer['response'], example, None)
            else:
                acc = 0.0  # pylint: disable=invalid-name
            validation_is_true.append(acc)
            logging.info('Recomputed accuracy!')

        else:
            validation_is_true.append(most_likely_answer['accuracy'])

        validation_answerable.append(is_answerable(example))
        validation_embeddings.append(most_likely_answer['embedding'])
        logging.info('validation_is_true: %f', validation_is_true[-1])

        if args.compute_predictive_entropy:
            # Token log likelihoods. Shape = (n_sample, n_tokens)
            if not args.use_all_generations:
                log_liks = [r[1] for r in full_responses[:args.use_num_generations]]
            else:
                log_liks = [r[1] for r in full_responses]

            for i in log_liks:
                assert i

            if args.compute_context_entails_response:
                # Compute context entails answer baseline.
                entropies['context_entails_response'].append(context_entails_response(
                    context, responses, entailment_model))

            if args.condition_on_question and args.entailment_model == 'deberta':
                responses = [f'{question} {r}' for r in responses]

            # Compute semantic ids.
            semantic_ids = get_semantic_ids(
                responses, model=entailment_model,
                strict_entailment=args.strict_entailment, example=example)

            result_dict['semantic_ids'].append(semantic_ids)

            # Compute entropy from frequencies of cluster assignments.
            entropies['cluster_assignment_entropy'].append(cluster_assignment_entropy(semantic_ids))

            # Length normalization of generation probabilities.
            log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]

            # Compute naive entropy.
            entropies['regular_entropy'].append(predictive_entropy(log_liks_agg))

            # Compute semantic entropy.
            log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
            pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
            entropies['semantic_entropy'].append(pe)

            # ====== NOISE ROBUSTNESS ANALYSIS ======
            # Compute SE_after under Gaussian noise perturbations (PER-TOKEN)
            se_before = pe
            correctness = most_likely_answer['accuracy']
            
            for mu in NOISE_MEANS:
                for sigma in NOISE_STDS:
                    se_after_samples = []
                    all_noise_values = []  # Track all noise added across samples
                    
                    for _ in range(N_NOISE_SAMPLES):
                        # Add Gaussian noise to PER-TOKEN log-likelihoods, then aggregate
                        noisy_log_liks_agg = []
                        
                        for log_lik_seq in log_liks:  # For each answer's token sequence
                            # Generate noise for each token in this answer
                            token_noise = np.random.normal(mu, sigma, len(log_lik_seq))
                            all_noise_values.extend(token_noise)  # Track all token-level noise
                            
                            # Add noise to each token's log-prob
                            noisy_log_lik_seq = [ll + n for ll, n in zip(log_lik_seq, token_noise)]
                            
                            # Aggregate: mean of noisy per-token log-probs
                            noisy_log_liks_agg.append(np.mean(noisy_log_lik_seq))
                        
                        # Recompute semantic entropy with noisy aggregated log-likelihoods
                        noisy_log_likelihood_per_semantic_id = logsumexp_by_id(
                            semantic_ids, noisy_log_liks_agg, agg='sum_normalized')
                        noisy_se = predictive_entropy_rao(noisy_log_likelihood_per_semantic_id)
                        se_after_samples.append(noisy_se)
                    
                    # Compute statistics
                    se_mean = np.mean(se_after_samples)
                    se_std = np.std(se_after_samples)
                    delta_se = se_mean - se_before
                    mean_noise = np.mean(all_noise_values)  # Empirical mean of ALL token-level noise
                    
                    # Store noise results
                    noise_record = {
                        'question_id': tid,
                        'question': question,
                        'generated_answers': responses,  # List of 10 generated answers
                        'mu': mu,
                        'sigma': sigma,
                        'se_before': float(se_before),
                        'se_after_samples': [float(s) for s in se_after_samples],
                        'se_mean': float(se_mean),
                        'se_std': float(se_std),
                        'delta_se': float(delta_se),
                        'mean_noise': float(mean_noise),
                        'correctness': float(correctness)
                    }
                    
                    # Save to JSONL file
                    noise_file = os.path.join(wandb.run.dir, 'se_after_noise.jsonl')
                    with open(noise_file, 'a') as f:
                        f.write(json.dumps(noise_record) + '\n')
            # ====== END NOISE ROBUSTNESS ANALYSIS ======

            # pylint: disable=invalid-name
            log_str = 'semantic_ids: %s, avg_token_log_likelihoods: %s, entropies: %s'
            entropies_fmt = ', '.join([f'{i}:{j[-1]:.2f}' for i, j in entropies.items()])
            # pylint: enable=invalid-name
            logging.info(80*'#')
            logging.info('NEW ITEM %d at id=`%s`.', idx, tid)
            logging.info('Context:')
            logging.info(example['context'])
            logging.info('Question:')
            logging.info(question)
            logging.info('True Answers:')
            logging.info(example['reference'])
            logging.info('Low Temperature Generation:')
            logging.info(most_likely_answer['response'])
            logging.info('Low Temperature Generation Accuracy:')
            logging.info(most_likely_answer['accuracy'])
            logging.info('High Temp Generation:')
            logging.info([r[0] for r in full_responses])
            logging.info('High Temp Generation:')
            logging.info(log_str, semantic_ids, log_liks_agg, entropies_fmt)

        if args.compute_p_true_in_compute_stage:
            p_true = p_true_utils.calculate_p_true(
                pt_model, question, most_likely_answer['response'],
                responses, p_true_few_shot_prompt,
                hint=old_exp['args'].p_true_hint)
            p_trues.append(p_true)
            logging.info('p_true: %s', np.exp(p_true))

        count += 1
        if count >= args.num_eval_samples:
            logging.info('Breaking out of main loop.')
            break

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
        # Assemble training data for embedding classification.
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
        logging.info('Starting training p_ik on train embeddings.')
        # Train classifier of correct/incorrect from embeddings.
        p_ik_predictions = get_p_ik(
            train_embeddings=train_embeddings, is_false=train_is_false,
            eval_embeddings=validation_embeddings, eval_is_false=validation_is_false)
        result_dict['uncertainty_measures']['p_ik'] = p_ik_predictions
        logging.info('Finished training p_ik on train embeddings.')

    if args.compute_p_ik_answerable:
        # Train classifier of answerable/unanswerable.
        p_ik_predictions = get_p_ik(
            train_embeddings=train_embeddings, is_false=train_unanswerable,
            eval_embeddings=validation_embeddings, eval_is_false=validation_unanswerable)
        result_dict['uncertainty_measures']['p_ik_unanswerable'] = p_ik_predictions

    if args.compute_p_true_in_compute_stage:
        result_dict['uncertainty_measures']['p_false'] = [1 - p for p in p_trues]
        result_dict['uncertainty_measures']['p_false_fixed'] = [1 - np.exp(p) for p in p_trues]

    utils.save(result_dict, 'uncertainty_measures.pkl')

    if args.compute_predictive_entropy:
        entailment_model.save_prediction_cache()

    if args.analyze_run:
        # Follow up with computation of aggregate performance metrics.
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
