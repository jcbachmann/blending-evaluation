import argparse
import logging
import tempfile
from pathlib import Path

from bmh_ml.datasets.store import load_bundle, load_dataset
from bmh_ml.evaluation.evaluate import evaluate_datasets
from bmh_ml.models.registry import load_model
from bmh_ml.tracking.runs import configure_mlflow, get_finite_metrics
from bmh_ml.train import get_set_metrics


def evaluate_run(run_id: str, bundle_name: str) -> dict[str, float]:
    """Evaluates the model of a run on the validation and test sets of another bundle and adds the metrics to the run as `<bundle>/<set>/...`.

    A bundle with the same scope, e.g. one with new test sets, or a test set made later. The training bundle is evaluated by the training.
    """
    mlflow = configure_mlflow()
    run = mlflow.get_run(run_id)
    if run.data.params["bundle"] == bundle_name:
        raise ValueError(f"Run {run_id} was trained on bundle {bundle_name} and evaluated on it, choose another bundle")
    bundle = load_bundle(bundle_name)
    if bundle.scope != run.data.params["scope"]:
        raise ValueError(f"Bundle {bundle_name} has scope {bundle.scope}, the run has scope {run.data.params['scope']}")

    with tempfile.TemporaryDirectory() as directory:
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model", dst_path=directory)
        model = load_model(Path(directory) / "model")
    datasets = {"val": load_dataset(bundle.val), **{name: load_dataset(dataset_id) for name, dataset_id in bundle.tests.items()}}
    results = evaluate_datasets(model, datasets, bundle.scope)
    metrics = get_set_metrics(results, prefix=f"{bundle_name}/")
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(get_finite_metrics(metrics))
    return metrics


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Evaluates the model of a run on another bundle")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bundle", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    metrics = evaluate_run(args.run_id, args.bundle)
    for name, value in sorted(metrics.items()):
        if name.endswith(("/nrmse", "/r2")):
            print(f"{name:<24}{value:>10.4f}")


if __name__ == "__main__":
    main()
