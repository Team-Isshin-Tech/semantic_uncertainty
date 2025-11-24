import math
import json
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def chosen_logprobs_from_noisy_logits(step_logits: List[torch.Tensor],
                                      chosen_ids: List[int],
                                      mu: float,
                                      sigma: float,
                                      rng: np.random.Generator) -> List[float]:
    """
    step_logits: list[T] of Tensor[vocab] on CPU float32
    chosen_ids:  list[T] of ints
    Returns list of chosen-token log-probs (float) for each step.
    """
    out = []
    for t, logits in enumerate(step_logits):
        # logits: torch.Tensor on CPU float32
        # Draw noise as numpy array and convert to torch
        eps_np = rng.normal(loc=mu, scale=sigma, size=(logits.shape[0],))
        eps = torch.from_numpy(eps_np).to(dtype=torch.float32)
        noisy = logits + eps
        logp = F.log_softmax(noisy, dim=-1)
        cid = int(chosen_ids[t])
        out.append(float(logp[cid].item()))
    return out


def cluster_probs_from_scores(cluster_members: List[List[int]], seq_logps: List[float]) -> List[float]:
    """Compute cluster probabilities from per-sequence log-scores.
    Stable log-sum-exp over members and softmax across clusters.
    """
    logs = []
    for members in cluster_members:
        if not members:
            logs.append(-1e9)
        else:
            m = max(seq_logps[m] for m in members)
            s = sum(math.exp(seq_logps[m] - m) for m in members)
            logs.append(m + math.log(s + 1e-12))
    m = max(logs)
    exps = [math.exp(x - m) for x in logs]
    Z = sum(exps) + 1e-12
    probs = [e / Z for e in exps]
    return probs


def entropy(probs: List[float]) -> float:
    arr = np.asarray(probs, dtype=np.float64)
    arr = arr[arr > 0]
    return float(-(arr * np.log(arr)).sum())


def se_run_from_logits(step_logits_list: List[List[torch.Tensor]],
                       step_ids_list: List[List[int]],
                       members: List[List[int]],
                       mu: float,
                       sigma: float,
                       rng: np.random.Generator) -> float:
    """
    Run one SE computation from raw logits for a single question.

    step_logits_list: List over answers (K) -> list of per-step Tensor[vocab]
    step_ids_list: List over answers -> list of chosen ids
    members: cluster member lists as returned by clustering on original texts
    """
    seq_logps = []
    K = len(step_logits_list)
    for k in range(K):
        step_logits = step_logits_list[k]
        step_ids = step_ids_list[k]
        if len(step_logits) == 0:
            # No generated tokens: assign very low score
            seq_logps.append(-1e9)
            continue
        chosen = chosen_logprobs_from_noisy_logits(step_logits, step_ids, mu, sigma, rng)
        seq_logps.append(float(np.mean(chosen)) if len(chosen) > 0 else -1e9)

    probs = cluster_probs_from_scores(members, seq_logps)
    return entropy(probs)


def compute_baseline_se(answers: List[str], token_logprobs: List[List[float]], members: List[List[int]] = None) -> Tuple[float, List[List[int]], List[float]]:
    """
    Compute baseline SE from clean token log-probs.
    If `members` is provided, reuse clustering; else do a trivial clustering by exact string match.
    Returns SE, members, seq_logps
    """
    # seq logps as mean token log-prob
    seq_logps = [float(np.mean(x)) if len(x) > 0 else -1e9 for x in token_logprobs]
    if members is None:
        # simple clustering by exact equality of answers (fallback)
        rep = {}
        members_map = {}
        for idx, a in enumerate(answers):
            if a not in rep:
                rep[a] = len(rep)
                members_map[rep[a]] = []
            members_map[rep[a]].append(idx)
        members = [members_map[i] for i in range(len(members_map))]
    probs = cluster_probs_from_scores(members, seq_logps)
    return entropy(probs), members, seq_logps
