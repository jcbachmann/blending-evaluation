"""Metrics of the predictions of one objective. Both objectives are minimized, so the best solutions have the lowest values."""

import numpy as np
from scipy.stats import spearmanr

TAIL_FRACTION = 0.1
PAIR_COUNT = 20_000


def get_metrics(y_true: np.ndarray, y_pred: np.ndarray, noise_sd: float | None = None, repeats: int = 1, seed: int = 0) -> dict[str, float]:
    """Errors overall and in the tail of the best solutions, and how well the order of the solutions is kept.

    `noise_sd` is the standard deviation of a single simulation of the same input. If it is given, `nrmse` is the rmse divided by it: 1 is
    the noise of one simulation. As the labels are the mean of `repeats` simulations, a perfect model has an `nrmse` of `1 / sqrt(repeats)`.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    total_variance = float(np.sum((y_true - y_true.mean()) ** 2))
    metrics = {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": 1 - float(np.sum(error**2)) / total_variance if total_variance > 0 else float("nan"),
        "negative_rate": float(np.mean(y_pred < 0)),
    }
    if noise_sd:
        metrics["nrmse"] = metrics["rmse"] / noise_sd

    tail = y_true <= np.quantile(y_true, TAIL_FRACTION)
    metrics["tail_rmse"] = float(np.sqrt(np.mean(error[tail] ** 2)))
    metrics["tail_bias"] = float(np.mean(error[tail]))

    metrics["spearman"] = float(spearmanr(y_true, y_pred).statistic) if np.ptp(y_true) > 0 and np.ptp(y_pred) > 0 else float("nan")
    min_gap = 2 * noise_sd / np.sqrt(repeats) if noise_sd else 0.0
    metrics["pairwise_accuracy"] = get_pairwise_accuracy(y_true, y_pred, min_gap, seed)
    return metrics


def get_pairwise_accuracy(y_true: np.ndarray, y_pred: np.ndarray, min_gap: float = 0.0, seed: int = 0) -> float:
    """The share of random pairs of inputs that the model puts in the right order. Pairs that the noise could have swapped are left out."""
    rng = np.random.default_rng(seed)
    first, second = rng.integers(0, len(y_true), (2, PAIR_COUNT))
    true_difference = y_true[first] - y_true[second]
    decided = np.abs(true_difference) > min_gap
    if not decided.any():
        return float("nan")
    return float(np.mean(np.sign(true_difference[decided]) == np.sign((y_pred[first] - y_pred[second])[decided])))
