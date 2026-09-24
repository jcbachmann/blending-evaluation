import argparse
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from bmh_ml.datasets.manifest import Dataset
from bmh_ml.datasets.store import Bundle, get_evaluation_sets, load_bundle, load_dataset, load_training_dataset
from bmh_ml.evaluation.evaluate import SetResult, evaluate_datasets, measure_throughput
from bmh_ml.evaluation.plots import close_figure, plot_predictions
from bmh_ml.evaluation.transfer import TransferConfig, log_transfer, run_transfer_test
from bmh_ml.models.registry import MODELS, create_model
from bmh_ml.tracking.environment import get_code_version, get_hardware, get_lock_hash
from bmh_ml.tracking.runs import configure_mlflow, get_experiment_id, get_experiment_name, get_finite_metrics

VALIDATION = "val"


@dataclass
class TrainingResult:
    run_id: str
    metrics: dict[str, float]


def load_bundle_datasets(bundle: Bundle) -> tuple[Dataset, dict[str, Dataset]]:
    """The training data and the sets the model is evaluated on, the validation set first."""
    return load_training_dataset(bundle), {name: load_dataset(dataset_id) for name, dataset_id in get_evaluation_sets(bundle).items()}


def get_set_metrics(results: dict[str, SetResult], prefix: str = "") -> dict[str, float]:
    return {f"{prefix}{name}/{metric}": value for name, result in results.items() for metric, value in result.metrics.items()}


def log_predictions(results: dict[str, SetResult], with_plots: bool = True):
    """The predictions and the plots of every evaluated set as artifacts of the current run."""
    mlflow = configure_mlflow()
    with tempfile.TemporaryDirectory() as directory:
        for name, result in results.items():
            file = Path(directory) / f"{name}.npz"
            np.savez_compressed(file, y_true=result.y_true, y_pred=result.y_pred)
            mlflow.log_artifact(str(file), "predictions")
            if with_plots:
                figure = plot_predictions(name, result.y_true, result.y_pred, result.metrics)
                mlflow.log_figure(figure, f"plots/{name}.png")
                close_figure(figure)


def run_training(
    bundle_name: str,
    model_name: str,
    params: dict,
    seed: int = 1,
    run_name: str | None = None,
    with_plots: bool = True,
    nested: bool = False,
    tags: dict[str, str] | None = None,
    transfer: TransferConfig | None = None,
) -> TrainingResult:
    """Trains a model on the training data of a bundle, evaluates it on the validation data and the test sets and logs everything to MLflow.

    `nested` makes the run a child of the active run, e.g. a trial of a sweep or a round of the refinement loop. `transfer` adds the
    transfer test (see `evaluation.transfer`), which needs the reference test set (T2) in the bundle.
    """
    bundle = load_bundle(bundle_name)
    train, evaluation = load_bundle_datasets(bundle)
    validation = evaluation[VALIDATION]
    x_train, x_val = train.features(bundle.scope), validation.features(bundle.scope)
    model = create_model(model_name, **params)

    logging.info(f"Training {model_name} on {bundle_name} ({len(train)} rows, scope {bundle.scope})")
    start = time.perf_counter()
    info = model.fit(x_train, train.y, x_val, validation.y, seed)
    fit_seconds = time.perf_counter() - start
    logging.info(f"Trained in {fit_seconds:.1f} s")

    results = evaluate_datasets(model, evaluation, bundle.scope)
    metrics = {
        **get_set_metrics(results),
        **{f"train/{key}": value for key, value in info.items()},
        "train/seconds": fit_seconds,
        "train/rows": float(len(train)),
        **measure_throughput(model, x_val),
    }
    transfer_result = None
    if transfer:
        if transfer.reference_set not in evaluation:
            raise ValueError(f"The transfer test needs the set {transfer.reference_set}, bundle {bundle.name} has {sorted(evaluation)}")
        transfer_result = run_transfer_test(model, bundle.scope, evaluation[transfer.reference_set], transfer)
        metrics.update({f"transfer/{name}": value for name, value in transfer_result.metrics.items()})

    mlflow = configure_mlflow()
    with mlflow.start_run(experiment_id=get_experiment_id(get_experiment_name(bundle.scope)), run_name=run_name, nested=nested) as run:
        mlflow.log_params(
            {
                **model.describe(),
                "bundle": bundle.name,
                "scope": bundle.scope,
                "seed": seed,
                "train_dataset": bundle.train,
                **({"train_extra_datasets": ",".join(bundle.train_extra)} if bundle.train_extra else {}),
                **{f"{name}_dataset": dataset_id for name, dataset_id in get_evaluation_sets(bundle).items()},
                "train_repeats": train.repeats,
            }
        )
        mlflow.set_tags(
            {
                "code_version": get_code_version(),
                "lock_hash": get_lock_hash(),
                **{f"hardware.{k}": str(v) for k, v in get_hardware().items()},
                **(tags or {}),
            }
        )
        mlflow.log_metrics(get_finite_metrics(metrics))
        with tempfile.TemporaryDirectory() as directory:
            model.save(Path(directory) / "model")
            mlflow.log_artifacts(str(Path(directory) / "model"), "model")
        log_predictions(results, with_plots)
        if transfer_result:
            log_transfer(transfer_result, transfer, with_plots)
        logging.info(f"Run {run.info.run_id} logged in experiment {get_experiment_name(bundle.scope)}")
        return TrainingResult(run.info.run_id, metrics)


def parse_parameters(assignments: list[str]) -> dict:
    """`key=value` pairs, the values are read as YAML: numbers, true/false, strings and lists."""
    params = {}
    for assignment in assignments:
        key, separator, value = assignment.partition("=")
        if not separator or not key:
            raise argparse.ArgumentTypeError(f"'{assignment}' is not of the form key=value")
        params[key] = yaml.safe_load(value)
    return params


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trains a model on a bundle and logs the run with its evaluation")
    parser.add_argument("--bundle", required=True, help="Name of the bundle, see build_bundle")
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--param", nargs="*", default=[], metavar="KEY=VALUE", help="Parameters of the model, e.g. width=512 depth=4")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-name", help="Name of the run in the UI")
    parser.add_argument("--transfer", action="store_true", help="Also run the transfer test with its default settings (python -m bmh_ml.transfer)")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    args.params = parse_parameters(args.param)
    return args


def main(argv: list[str] | None = None):
    args = get_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    transfer = TransferConfig() if args.transfer else None
    result = run_training(args.bundle, args.model, args.params, args.seed, args.run_name, not args.no_plots, transfer=transfer)
    shown = {name: value for name, value in result.metrics.items() if name.split("/")[-1] in ("nrmse", "r2") and name.split("/")[0] in (VALIDATION, "T1", "T2")}
    shown.update({name: value for name, value in result.metrics.items() if name in ("transfer/hv_ratio", "transfer/igd_plus")})
    for name, value in sorted(shown.items()):
        print(f"{name:<16}{value:>10.4f}")
    print(f"Run {result.run_id}")


if __name__ == "__main__":
    main()
