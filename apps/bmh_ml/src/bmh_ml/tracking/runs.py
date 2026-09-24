"""Thin wrapper around MLflow: where the runs are stored and how they are named."""

import logging
import math
import os

from bmh_ml.tracking.store import get_artifact_root, get_tracking_uri

EXPERIMENT_NAMES = {"S1": "S1-fixed-material", "S2": "S2-general-material"}


def get_experiment_name(scope: str) -> str:
    return EXPERIMENT_NAMES[scope]


def configure_mlflow():
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # a hint for coding agents that MLflow logs on import
    import mlflow

    mlflow.set_tracking_uri(get_tracking_uri())
    return mlflow


def get_experiment_id(name: str) -> str:
    mlflow = configure_mlflow()
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        return mlflow.create_experiment(name, artifact_location=get_artifact_root())
    return experiment.experiment_id


def log_metrics_to_run(run_id: str, metrics: dict[str, float]) -> None:
    """Adds metrics to a run by its id, without reopening it (reopening a finished run moves its end time)."""
    import time

    from mlflow.entities import Metric

    client = configure_mlflow().MlflowClient()
    timestamp = int(time.time() * 1000)
    entries = [Metric(key, value, timestamp, 0) for key, value in get_finite_metrics(metrics).items()]
    for start in range(0, len(entries), 1000):  # the tracking API accepts at most 1000 metrics per batch
        client.log_batch(run_id, metrics=entries[start : start + 1000])


def get_finite_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Metrics without NaN or infinity, which undefined metrics (e.g. R2 of constant labels) or a broken model would produce."""
    finite = {name: float(value) for name, value in metrics.items() if math.isfinite(value)}
    dropped = sorted(set(metrics) - set(finite))
    if dropped:
        logging.warning(f"Not logged as they are not finite: {dropped}")
    return finite
