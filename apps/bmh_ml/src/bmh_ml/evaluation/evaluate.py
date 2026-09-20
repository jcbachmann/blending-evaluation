"""Evaluates a model on the datasets of a bundle."""

import time
from dataclasses import dataclass

import numpy as np

from bmh_ml.datasets.manifest import Dataset
from bmh_ml.evaluation.metrics import get_metrics
from bmh_ml.evaluation.noise import OBJECTIVES, get_noise_summary
from bmh_ml.models.base import Model

THROUGHPUT_BATCH_SIZES = (100, 10_000)


@dataclass
class SetResult:
    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: dict[str, float]  # keys like "F1/rmse"


def evaluate_dataset(model: Model, dataset: Dataset, scope: str) -> SetResult:
    y_pred = model.predict(dataset.features(scope))
    metrics = {}
    noise = get_noise_summary(dataset) if dataset.y_noise_sd is not None else None
    for i, objective in enumerate(OBJECTIVES):
        noise_sd = noise[objective]["noise_sd"] if noise else None
        for name, value in get_metrics(dataset.y[:, i], y_pred[:, i], noise_sd, dataset.repeats).items():
            metrics[f"{objective}/{name}"] = value
        if noise:
            metrics[f"{objective}/r2_ceiling"] = noise[objective]["r2_ceiling_labels"]
            metrics[f"{objective}/noise_sd"] = noise_sd
    return SetResult(dataset.y, y_pred, metrics)


def evaluate_datasets(model: Model, datasets: dict[str, Dataset], scope: str) -> dict[str, SetResult]:
    return {name: evaluate_dataset(model, dataset, scope) for name, dataset in datasets.items()}


def measure_throughput(model: Model, x: np.ndarray, batch_sizes: tuple[int, ...] = THROUGHPUT_BATCH_SIZES, min_seconds: float = 0.5) -> dict[str, float]:
    """Predictions per second for batches of the given sizes, which is what an optimizer calling the model in every generation experiences."""
    result = {}
    for batch_size in batch_sizes:
        batch = x[np.arange(batch_size) % len(x)]
        model.predict(batch)  # the first call includes the set up
        calls, start = 0, time.perf_counter()
        while time.perf_counter() - start < min_seconds:
            model.predict(batch)
            calls += 1
        result[f"throughput/batch_{batch_size}"] = calls * batch_size / (time.perf_counter() - start)
    return result
