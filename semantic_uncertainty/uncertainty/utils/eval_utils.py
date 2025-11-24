"""Functions for performance evaluation, mainly used in analyze_results.py.

This module contains defensive wrappers around bootstrap and common
metrics so that analysis on tiny or degenerate datasets returns
well-formed, nan-safe values instead of raising exceptions or
producing runtime warnings that propagate to the caller.
"""
import warnings
import math
import numpy as np
import scipy
from sklearn import metrics


# pylint: disable=missing-function-docstring


def bootstrap(function, rng, n_resamples=1000):
    def inner(data):
        try:
            # Catch runtime warnings (e.g. invalid value encountered) and
            # convert them into a controlled fallback below.
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                bs = scipy.stats.bootstrap(
                    (data, ), function, n_resamples=n_resamples, confidence_level=0.9,
                    random_state=rng)

            # Validate results: if any returned value is NaN or infinite,
            # return a nan-safe structure instead of propagating odd values.
            std_err = float(bs.standard_error)
            low = float(bs.confidence_interval.low)
            high = float(bs.confidence_interval.high)
            if any(map(lambda x: not np.isfinite(x), [std_err, low, high])):
                return {'std_err': math.nan, 'low': math.nan, 'high': math.nan}

            return {'std_err': std_err, 'low': low, 'high': high}
        except Exception:
            # Anything that went wrong during bootstrap (too few samples,
            # degenerate inputs, runtime warnings promoted to errors, etc.)
            # is handled here. Return nan-safe summary so downstream code
            # can continue and log the situation.
            return {'std_err': math.nan, 'low': math.nan, 'high': math.nan}

    return inner


def auroc(y_true, y_score):
    # AUROC is undefined when only one class is present in y_true.
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return math.nan

    try:
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        del thresholds
        return metrics.auc(fpr, tpr)
    except Exception:
        return math.nan


def accuracy_at_quantile(accuracies, uncertainties, quantile):
    cutoff = np.quantile(uncertainties, quantile)
    select = uncertainties <= cutoff
    if not np.any(select):
        return math.nan
    return float(np.mean(accuracies[select]))


def area_under_thresholded_accuracy(accuracies, uncertainties):
    quantiles = np.linspace(0.1, 1, 20)
    select_accuracies = np.array([
        accuracy_at_quantile(accuracies, uncertainties, q) for q in quantiles
    ], dtype=float)
    # If all entries are nan, return nan. Otherwise treat nan as zero for
    # area computation (conservative) after logging is performed by caller.
    if np.all(np.isnan(select_accuracies)):
        return math.nan
    # Replace NaNs with 0 for area computation (so partial results remain usable).
    select_accuracies = np.nan_to_num(select_accuracies, nan=0.0)
    dx = float(quantiles[1] - quantiles[0])
    area = float((select_accuracies * dx).sum())
    return area


# Need wrappers because scipy expects 1D data.
def compatible_bootstrap(func, rng):
    def helper(y_true_y_score):
        # this function is called in the bootstrap
        y_true = np.array([i['y_true'] for i in y_true_y_score])
        y_score = np.array([i['y_score'] for i in y_true_y_score])
        out = func(y_true, y_score)
        return out

    def wrap_inputs(y_true, y_score):
        return [{'y_true': i, 'y_score': j} for i, j in zip(y_true, y_score)]

    def converted_func(y_true, y_score):
        y_true_y_score = wrap_inputs(y_true, y_score)
        try:
            return bootstrap(helper, rng=rng)(y_true_y_score)
        except Exception:
            # If bootstrap fails for any reason, return nan-safe structure
            # so callers can continue. The structure mirrors the normal
            # return of bootstrap(...) above.
            return {'std_err': math.nan, 'low': math.nan, 'high': math.nan}
    return converted_func
