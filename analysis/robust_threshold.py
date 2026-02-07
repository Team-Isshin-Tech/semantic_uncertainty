import argparse
import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def safe_divide(numer: float, denom: float) -> float:
    if denom == 0:
        return 0.0
    return numer / denom


def compute_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    # Positive class is confabulation (label=1).
    preds = (scores >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    fpr = safe_divide(fp, fp + tn)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def f_beta(precision: float, recall: float, beta: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta2 = beta * beta
    return (1 + beta2) * (precision * recall) / (beta2 * precision + recall)


def select_threshold(
    scores_before: np.ndarray,
    scores_mean: np.ndarray,
    labels: np.ndarray,
    recall_target: float,
    fpr_max: float,
    flip_max: float | None,
    beta: float = 2.0,
    selection: str = "constraints",
) -> Tuple[float, Dict[str, float]]:
    # Sweep candidates over the combined range to enforce consistency across both metrics.
    combined = np.concatenate([scores_before, scores_mean])
    candidates = np.unique(combined)

    best = None
    best_stats = None

    for t in candidates:
        metrics_before = compute_metrics(scores_before, labels, t)
        metrics_mean = compute_metrics(scores_mean, labels, t)

        recall_min = min(metrics_before["recall"], metrics_mean["recall"])
        fpr_max_observed = max(metrics_before["fpr"], metrics_mean["fpr"])
        avg_precision = (metrics_before["precision"] + metrics_mean["precision"]) / 2.0
        avg_f1 = (metrics_before["f1"] + metrics_mean["f1"]) / 2.0
        avg_fbeta = (
            f_beta(metrics_before["precision"], metrics_before["recall"], beta)
            + f_beta(metrics_mean["precision"], metrics_mean["recall"], beta)
        ) / 2.0
        flip_rate = float(
            np.mean((scores_before >= t).astype(int) != (scores_mean >= t).astype(int))
        )

        if selection == "recall-only":
            # Pick the highest threshold that still meets recall, to reduce false positives.
            if recall_min >= recall_target:
                score = (float(t), avg_precision, avg_f1)
                if best is None or score > best:
                    best = score
                    best_stats = {
                        "threshold": float(t),
                        "avg_precision": avg_precision,
                        "avg_recall_min": recall_min,
                        "avg_f1": avg_f1,
                        "avg_fbeta": avg_fbeta,
                        "avg_fpr_max": fpr_max_observed,
                        "flip_rate": flip_rate,
                        "metrics_before": metrics_before,
                        "metrics_mean": metrics_mean,
                        "used_fallback": False,
                    }
            else:
                if best_stats is None:
                    # Fallback if recall target not met: maximize recall, then precision.
                    score = (recall_min, avg_precision, avg_f1)
                    if best is None or score > best:
                        best = score
                        best_stats = {
                            "threshold": float(t),
                            "avg_precision": avg_precision,
                            "avg_recall_min": recall_min,
                            "avg_f1": avg_f1,
                            "avg_fbeta": avg_fbeta,
                            "avg_fpr_max": fpr_max_observed,
                            "flip_rate": flip_rate,
                            "metrics_before": metrics_before,
                            "metrics_mean": metrics_mean,
                            "used_fallback": True,
                        }
        else:
            meets_recall = recall_min >= recall_target
            meets_fpr = fpr_max_observed <= fpr_max
            meets_flip = True if flip_max is None else flip_rate <= flip_max

            if meets_recall and meets_fpr and meets_flip:
                # Most stable = lowest flip rate, then highest precision/F1.
                score = (-flip_rate, avg_precision, avg_f1, recall_min)
                if best is None or score > best:
                    best = score
                    best_stats = {
                        "threshold": float(t),
                        "avg_precision": avg_precision,
                        "avg_recall_min": recall_min,
                        "avg_f1": avg_f1,
                        "avg_fbeta": avg_fbeta,
                        "avg_fpr_max": fpr_max_observed,
                        "flip_rate": flip_rate,
                        "metrics_before": metrics_before,
                        "metrics_mean": metrics_mean,
                        "used_fallback": False,
                    }
            else:
                if best_stats is None:
                    # Fallback if no threshold meets constraints: pick best F-beta.
                    # This still prioritizes recall while ensuring some usable threshold.
                    if best is None or avg_fbeta > best[0]:
                        best = (avg_fbeta, avg_precision, recall_min)
                        best_stats = {
                            "threshold": float(t),
                            "avg_precision": avg_precision,
                            "avg_recall_min": recall_min,
                            "avg_f1": avg_f1,
                            "avg_fbeta": avg_fbeta,
                            "avg_fpr_max": fpr_max_observed,
                            "flip_rate": flip_rate,
                            "metrics_before": metrics_before,
                            "metrics_mean": metrics_mean,
                            "used_fallback": True,
                        }

    return best_stats["threshold"], best_stats


def compute_uncertainty_margin(
    scores_before: np.ndarray,
    scores_mean: np.ndarray,
    percentile: float,
) -> float:
    # Delta captures expected max perturbation due to numerical noise.
    delta = np.abs(scores_mean - scores_before)
    return float(np.percentile(delta, percentile))


def robust_decision(scores: np.ndarray, threshold: float, margin: float) -> np.ndarray:
    # Three-way decision to prevent small perturbations from flipping outcomes.
    decisions = np.full(scores.shape[0], -1, dtype=int)
    decisions[scores >= threshold + margin] = 1
    decisions[scores <= threshold - margin] = 0
    return decisions


def decision_flip_rate(
    scores_before: np.ndarray,
    scores_mean: np.ndarray,
    threshold: float,
    margin: float,
) -> float:
    decisions_before = robust_decision(scores_before, threshold, margin)
    decisions_mean = robust_decision(scores_mean, threshold, margin)
    return float(np.mean(decisions_before != decisions_mean))


def report_results(
    threshold: float,
    margin: float,
    scores_before: np.ndarray,
    scores_mean: np.ndarray,
    labels: np.ndarray,
    stats: Dict[str, float],
) -> None:
    metrics_before = compute_metrics(scores_before, labels, threshold)
    metrics_mean = compute_metrics(scores_mean, labels, threshold)

    decisions_before = robust_decision(scores_before, threshold, margin)
    decisions_mean = robust_decision(scores_mean, threshold, margin)

    uncertain_rate_before = float(np.mean(decisions_before == -1))
    uncertain_rate_mean = float(np.mean(decisions_mean == -1))
    flip_rate = decision_flip_rate(scores_before, scores_mean, threshold, margin)

    print("Robust Threshold Report")
    print("=======================")
    print(f"t* (base threshold): {threshold:.6f}")
    print(f"Delta (uncertainty margin): {margin:.6f}")

    if stats.get("used_fallback"):
        print("NOTE: Recall target not met; fallback to best F-beta.")

    print("\nConfabulation Detection (label=1)")
    print(f"Recall (se_before):   {metrics_before['recall']:.4f}")
    print(f"Precision (se_before):{metrics_before['precision']:.4f}")
    print(f"FPR (se_before):      {metrics_before['fpr']:.4f}")
    print(f"Recall (se_mean):     {metrics_mean['recall']:.4f}")
    print(f"Precision (se_mean):  {metrics_mean['precision']:.4f}")
    print(f"FPR (se_mean):        {metrics_mean['fpr']:.4f}")

    print("\nUncertainty Band")
    print(f"Uncertain rate (se_before): {uncertain_rate_before:.4f}")
    print(f"Uncertain rate (se_mean):   {uncertain_rate_mean:.4f}")
    print(f"Decision flip rate:         {flip_rate:.4f}")

    print("\nRationale")
    print("- Recall is prioritized to minimize missed confabulations in high-stakes use.")
    print("- The margin Delta reduces flip risk from small noise without repeated queries.")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    # Accept either "label" or "correctness" to avoid extra preprocessing.
    if "label" not in df.columns and "correctness" in df.columns:
        df = df.rename(columns={"correctness": "label"})

    required = {"se_before", "se_mean", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust confabulation thresholding")
    parser.add_argument(
        "--data",
        type=str,
        default="results/se_after_noise_complete.jsonl",
        help="Path to JSONL with se_before, se_mean, label",
    )
    parser.add_argument(
        "--recall-target",
        type=float,
        default=0.95,
        help="Minimum recall target for confabulations",
    )
    parser.add_argument(
        "--fpr-max",
        type=float,
        default=0.30,
        help="Maximum false positive rate on correct answers",
    )
    parser.add_argument(
        "--flip-max",
        type=float,
        default=None,
        help="Maximum allowed flip rate between se_before and se_mean",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="constraints",
        choices=["constraints", "recall-only"],
        help="Threshold selection strategy",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Percentile for uncertainty margin Delta",
    )
    args = parser.parse_args()

    df = load_data(args.data)
    scores_before = df["se_before"].to_numpy(dtype=float)
    scores_mean = df["se_mean"].to_numpy(dtype=float)
    labels = df["label"].to_numpy(dtype=int)

    threshold, stats = select_threshold(
        scores_before,
        scores_mean,
        labels,
        recall_target=args.recall_target,
        fpr_max=args.fpr_max,
        flip_max=args.flip_max,
        selection=args.selection,
    )
    margin = compute_uncertainty_margin(scores_before, scores_mean, args.percentile)

    report_results(threshold, margin, scores_before, scores_mean, labels, stats)


if __name__ == "__main__":
    main()
