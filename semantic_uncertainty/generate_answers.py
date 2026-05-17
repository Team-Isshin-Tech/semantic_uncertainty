"""Sample answers from LLMs on QA task."""
import gc
import os
import logging
import random
import json
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import shutil
import wandb

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

from uncertainty.data.data_utils import load_ds
from uncertainty.utils import utils
from uncertainty.uncertainty_measures import p_true as p_true_utils
from compute_uncertainty_measures import main as main_compute


def get_git_commit_hash():
    """Get current git commit hash. Returns 'unknown' if git not available."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return 'unknown'


def setup_run_output_directory(args):
    """Create run output directory structure and configure logging.
    
    Returns:
        run_dir: Path to the run-specific output directory
    """
    args.output_dir = os.path.abspath(args.output_dir)


    # Create run_id from timestamp + model_name + dataset + num_samples + num_generations
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_short = args.model_name.replace('/', '_').replace('-', '_')[:30]
    run_id = f"{timestamp}_{model_short}_{args.dataset}_{args.num_samples}_{args.num_generations}"
    
    # Create directory structure
    run_dir = os.path.join(args.output_dir, run_id)
    logs_dir = os.path.join(run_dir, 'logs')
    artifacts_dir = os.path.join(run_dir, 'artifacts')
    
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Setup logging to file and console
    log_file = os.path.join(logs_dir, 'run.log')
    logger = logging.getLogger()
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)

    # Avoid adding multiple handlers if already present
    log_file_abs = os.path.abspath(log_file)

    if not any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == log_file_abs
        for h in logger.handlers
    ):
        logger.addHandler(file_handler)
    
    # Save run metadata
    metadata = {
        'run_id': run_id,
        'hostname': socket.gethostname(),
        'username': os.environ.get('USER', 'unknown'),
        'start_time': datetime.now().isoformat(),
        'git_commit': get_git_commit_hash(),
        'args': vars(args),
    }
    
    metadata_file = os.path.join(run_dir, 'run_info.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info('='*80)
    logging.info(f'Run output directory: {run_dir}')
    logging.info(f'Run ID: {run_id}')
    logging.info(f'Logs: {logs_dir}')
    logging.info(f'Artifacts: {artifacts_dir}')
    logging.info(f'Metadata: {metadata_file}')
    logging.info('='*80)
    
    # return run_dir
    return run_dir, artifacts_dir



utils.setup_logger()


def main(args):
    logging.info("CWD: %s", os.getcwd())
    # ====== NEW CODE: Setup output directory and logging ======
    # run_dir = setup_run_output_directory(args)
    run_dir, artifacts_dir = setup_run_output_directory(args)
    # ====== END NEW CODE ======


    # Setup run.
    if args.dataset == 'svamp':
        if not args.use_context:
            logging.info('Forcing `use_context=True` for svamp dataset.')
            args.use_context = True
    elif args.dataset == 'squad':
        if not args.answerable_only:
            logging.info('Forcing `answerable_only=True` for squad dataset.')
            args.answerable_only = True

    experiment_details = {'args': args}
    random.seed(args.random_seed)
    user = os.environ['USER']
    slurm_jobid = os.getenv('SLURM_JOB_ID', None)
    scratch_dir = os.getenv('SCRATCH_DIR', '.')
    if not os.path.exists(f"{scratch_dir}/{user}/uncertainty"):
        os.makedirs(f"{scratch_dir}/{user}/uncertainty")

    wandb.init(
        entity=args.entity,
        project="semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug",
        dir=f"{scratch_dir}/{user}/uncertainty",
        config=args,
        notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
    )
    logging.info('Finished wandb init.')

    # Get accuracy metric.
    metric = utils.get_metric(args.metric)

    def save_and_copy(obj, filename):
        """Save object to wandb and copy to artifacts directory."""
        try:
            # save using repo's utils (goes to wandb.run.dir)
            utils.save(obj, filename)
            logging.info(f"✅ Saved {filename} to wandb.run.dir: {wandb.run.dir}")
        except Exception as e:
            logging.error(f"❌ Failed to save {filename} to wandb: {e}")
            raise

        try:
            # copy into results/<run_id>/artifacts
            src = os.path.join(wandb.run.dir, filename)
            dst = os.path.join(artifacts_dir, filename)
            
            if os.path.exists(src):
                shutil.copy2(src, dst)
                src_size = os.path.getsize(src) / (1024**2)
                dst_size = os.path.getsize(dst) / (1024**2)
                logging.info(f"✅ Copied {filename} to artifacts/")
                logging.info(f"   Source: {src} ({src_size:.2f} MB)")
                logging.info(f"   Dest:   {dst} ({dst_size:.2f} MB)")
            else:
                logging.error(f"❌ Source file not found at {src}")
                raise FileNotFoundError(f"Source pickle file not found: {src}")
        except Exception as e:
            logging.error(f"❌ Failed to copy {filename} to artifacts: {e}")
            raise


    # Load dataset.
    train_dataset, validation_dataset = load_ds(
        args.dataset, add_options=args.use_mc_options, seed=args.random_seed)
    if args.ood_train_dataset is not None:
        logging.warning(
            'Using OOD dataset %s to construct few-shot prompts and train p_ik.',
            args.ood_train_dataset)
        # Get indices of answerable and unanswerable questions and construct prompt.
        train_dataset, _ = load_ds(args.ood_train_dataset, add_options=args.use_mc_options)
    if not isinstance(train_dataset, list):
        logging.info('Train dataset: %s', train_dataset)

    # Get indices of answerable and unanswerable questions and construct prompt.
    answerable_indices, unanswerable_indices = utils.split_dataset(train_dataset)

    if args.answerable_only:
        unanswerable_indices = []
        val_answerable, val_unanswerable = utils.split_dataset(validation_dataset)
        del val_unanswerable
        validation_dataset = [validation_dataset[i] for i in val_answerable]

    prompt_indices = random.sample(answerable_indices, args.num_few_shot)
    experiment_details['prompt_indices'] = prompt_indices
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))

    # Create Few-Shot prompt.
    make_prompt = utils.get_make_prompt(args)
    BRIEF = utils.BRIEF_PROMPTS[args.brief_prompt]
    arg = args.brief_always if args.enable_brief else True
    prompt = utils.construct_fewshot_prompt_from_indices(
        train_dataset, prompt_indices, BRIEF, arg, make_prompt)
    experiment_details['prompt'] = prompt
    experiment_details['BRIEF'] = BRIEF
    logging.info('Prompt is: %s', prompt)

    # Initialize model.
    model = utils.init_model(args)

    # Initialize prompt for p_true baseline.
    if args.compute_p_true:
        logging.info(80*'#')
        logging.info('Constructing few-shot prompt for p_true.')

        p_true_indices = random.sample(answerable_indices, args.p_true_num_fewshot)
        remaining_answerable = list(set(remaining_answerable) - set(p_true_indices))
        p_true_few_shot_prompt, p_true_responses, len_p_true = p_true_utils.construct_few_shot_prompt(
            model=model, dataset=train_dataset, indices=p_true_indices,
            prompt=prompt, brief=BRIEF,
            brief_always=args.brief_always and args.enable_brief,
            make_prompt=make_prompt, num_generations=args.num_generations,
            metric=metric)
        wandb.config.update(
            {'p_true_num_fewshot': len_p_true}, allow_val_change=True)
        wandb.log(dict(len_p_true=len_p_true))
        experiment_details['p_true_indices'] = p_true_indices
        experiment_details['p_true_responses'] = p_true_responses
        experiment_details['p_true_few_shot_prompt'] = p_true_few_shot_prompt
        logging.info('Finished constructing few-shot prompt for p_true.')
        logging.info(80*'#')
        logging.info('p_true_few_shot_prompt: %s', p_true_few_shot_prompt)
        logging.info(80*'#')

    # Start answer generation.
    logging.info(80 * '=')
    logging.info('Generating answers: ')
    logging.info(80 * '=')
    for dataset_split in ['train', 'validation']:
        logging.info(80 * 'x')
        logging.info('Starting with dataset_split %s.', dataset_split)
        logging.info(80 * 'x')

        # This will store all input data and model predictions.
        accuracies, generations, results_dict, p_trues = [], {}, {}, []
        records = []  # Track SE_before per question

        if dataset_split == 'train':
            if not args.get_training_set_generations:
                logging.info('Skip training data.')
                continue
            dataset = train_dataset
            possible_indices = list(set(remaining_answerable) | set(unanswerable_indices))
        else:
            dataset = validation_dataset
            possible_indices = range(0, len(dataset))

        # Evaluate over random subset of the datasets.
        indices = random.sample(possible_indices, min(args.num_samples, len(dataset)))
        experiment_details[dataset_split] = {'indices': indices}

        if args.num_samples > len(dataset):
            logging.warning('Not enough samples in dataset. Using all %d samples.', len(dataset))

        it = 0
        for index in tqdm(indices):
            if (it + 1 % 10) == 0:
                gc.collect()
                torch.cuda.empty_cache()
            it += 1

            # Grab example at index.
            example = dataset[index]
            question, context = example["question"], example['context']
            generations[example['id']] = {'question': question, 'context': context}
            correct_answer = example['answers']['text']

            current_input = make_prompt(
                context, question, None, BRIEF, args.brief_always and args.enable_brief)
            local_prompt = prompt + current_input

            logging.info('Current input: '.ljust(15) + current_input)

            full_responses = []

            # We sample one low temperature answer on which we will compute the
            # accuracy and args.num_generation high temperature answers which will
            # be used to estimate the entropy variants.

            if dataset_split == 'train' and args.get_training_set_generations_most_likely_only:
                num_generations = 1
            else:
                num_generations = args.num_generations + 1

            for i in range(num_generations):

                # Temperature for first generation is always `0.1`.
                temperature = 0.1 if i == 0 else args.temperature

                predicted_answer, token_log_likelihoods, embedding, logits_per_token, generated_token_ids = model.predict(
                    local_prompt, temperature)
                embedding = embedding.cpu() if embedding is not None else None

                # Only compute accuracy if question is answerable.
                compute_acc = args.compute_accuracy_at_all_temps or (i == 0)
                if correct_answer and compute_acc:
                    acc = metric(predicted_answer, example, model)
                else:
                    acc = 0.0  # pylint: disable=invalid-name

                if i == 0:
                    logging.info('Iteration ' + str(it) + ':  ' + 80*'#')
                    if args.use_context:
                        logging.info('context: '.ljust(15) + str(context))
                    logging.info('question: '.ljust(15) + question)
                    logging.info('low-t prediction: '.ljust(15) + predicted_answer)
                    logging.info('correct answer: '.ljust(15) + str(correct_answer))
                    logging.info('accuracy: '.ljust(15) + str(acc))

                    accuracies.append(acc)
                    most_likely_answer_dict = {
                        'response': predicted_answer,
                        'token_log_likelihoods': token_log_likelihoods,
                        'embedding': embedding,
                        'accuracy': acc}
                    generations[example['id']].update({
                        'most_likely_answer': most_likely_answer_dict,
                        'reference': utils.get_reference(example)})

                else:
                    logging.info('high-t prediction '.ljust(15) + str(i) + ' : ' + predicted_answer)
                    # Aggregate predictions over num_generations.
                    full_responses.append(
                        (predicted_answer, token_log_likelihoods, embedding, logits_per_token, generated_token_ids, acc))

            # Append all predictions for this example to `generations`.
            generations[example['id']]['responses'] = full_responses

            # ====== NEW CODE: Compute SE_before (token-level predictive entropy) ======
            if dataset_split == 'validation':
                from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy
                
                se_before = None
                if full_responses:
                    # Get log likelihoods from all high-T generations
                    all_token_lls = []
                    for response_tuple in full_responses:
                        _, token_ll, _, _, _, _ = response_tuple
                        if token_ll:
                            all_token_lls.extend(token_ll)
                    if all_token_lls:
                        se_before = predictive_entropy(np.array(all_token_lls))
                
                record = {
                    'question_id': example['id'],
                    'se_before': se_before,
                    'correctness': most_likely_answer_dict['accuracy'],
                }
                records.append(record)
            # ====== END NEW CODE ======

            if args.compute_p_true and dataset_split == 'validation':
                # Already compute p_true here. Avoid cost of generations in compute_uncertainty script.
                p_true = p_true_utils.calculate_p_true(
                    model, question, most_likely_answer_dict['response'],
                    [r[0] for r in full_responses], p_true_few_shot_prompt,
                    hint=args.p_true_hint)
                p_trues.append(p_true)
                logging.info('p_true: %s', p_true)

        # Save generations for that split.
        logging.info(f"Saving {dataset_split}_generations.pkl...")
        try:
            save_and_copy(generations, f'{dataset_split}_generations.pkl')
            logging.info(f"✅ Successfully saved and copied {dataset_split}_generations.pkl")
        except Exception as e:
            logging.error(f"❌ CRITICAL: Failed to save {dataset_split}_generations.pkl: {e}")
            raise

        # Log overall accuracy.
        accuracy = np.mean(accuracies)
        print(f"Overall {dataset_split} split accuracy: {accuracy}")
        wandb.log({f"{dataset_split}_accuracy": accuracy})

        if dataset_split == 'validation':
            if args.compute_p_true:
                results_dict['uncertainty_measures'] = {
                    'p_false':  [1 - p for p in p_trues],
                    'p_false_fixed':  [1 - np.exp(p) for p in p_trues],
                }
            logging.info("Saving validation uncertainty_measures.pkl...")
            try:
                save_and_copy(results_dict, 'uncertainty_measures.pkl')
                logging.info("✅ Successfully saved and copied uncertainty_measures.pkl")
            except Exception as e:
                logging.error(f"❌ CRITICAL: Failed to save uncertainty_measures.pkl: {e}")
                raise

    logging.info("Saving experiment_details.pkl...")
    try:
        utils.save(experiment_details, 'experiment_details.pkl')
        src = os.path.join(wandb.run.dir, 'experiment_details.pkl')
        dst = os.path.join(artifacts_dir, 'experiment_details.pkl')
        if os.path.exists(src):
            shutil.copy2(src, dst)
            logging.info("✅ Successfully saved experiment_details.pkl")
        else:
            logging.error(f"❌ experiment_details.pkl not found at {src}")
    except Exception as e:
        logging.error(f"❌ CRITICAL: Failed to save experiment_details.pkl: {e}")
        raise

    logging.info('Run complete.')
    del model


if __name__ == '__main__':

    parser = utils.get_parser()
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    if args.compute_uncertainties:
        args.assign_new_wandb_id = False

    # First sample generations from LLM.
    logging.info('STARTING `generate_answers`!')
    main(args)
    logging.info('FINISHED `generate_answers`!')

    if args.compute_uncertainties:
        # Follow with uncertainty calculation script by default.
        args.assign_new_wandb_id = False
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(50 * '#X')
        logging.info('STARTING `compute_uncertainty_measures`!')
        main_compute(args)
        logging.info('FINISHED `compute_uncertainty_measures`!')
