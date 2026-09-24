import argparse
import logging
from pathlib import Path

import numpy as np

from bmh_ml.datasets.generators import find_result_files
from bmh_ml.datasets.store import load_bundle, load_dataset
from bmh_ml.evaluate_run import load_run_model
from bmh_ml.evaluation.chevron import get_chevron_objectives
from bmh_ml.evaluation.transfer import (
    TransferConfig,
    TransferResult,
    get_reference_front,
    get_simulator_run_metrics,
    get_transfer_metrics,
    log_transfer,
    run_transfer_test,
)
from bmh_ml.tracking.annotate import annotate_run
from bmh_ml.tracking.runs import configure_mlflow


def transfer_run(run_id: str, config: TransferConfig, with_plots: bool = True) -> TransferResult:
    """Runs the transfer test on the model of a training run and adds the results to the run."""
    mlflow = configure_mlflow()
    run = mlflow.get_run(run_id)
    bundle = load_bundle(run.data.params["bundle"])
    if config.reference_set not in bundle.tests:
        raise ValueError(f"The transfer test needs the test set {config.reference_set}, bundle {bundle.name} has {sorted(bundle.tests)}")
    result = run_transfer_test(load_run_model(run_id), bundle.scope, load_dataset(bundle.tests[config.reference_set]), config)
    log_transfer(run_id, result, config, with_plots)
    annotate_run(run_id)
    return result


def recompute_transfer(run_id: str, with_plots: bool = True) -> dict[str, float]:
    """Computes the metrics of an earlier transfer test again from its stored solutions, without optimizing or simulating: for metrics
    added after the test ran. Keeps the timing metrics of the test."""
    import tempfile

    mlflow = configure_mlflow()
    run = mlflow.get_run(run_id)
    bundle = load_bundle(run.data.params["bundle"])
    reference_dataset = load_dataset(bundle.tests[run.data.params.get("transfer.reference_set", "T2")])
    with tempfile.TemporaryDirectory() as directory:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="transfer/solutions.npz", dst_path=directory)
        with np.load(path) as stored:
            arrays = {name: stored[name] for name in stored.files}
    chevron = get_chevron_objectives(reference_dataset.material[:1])
    reference = get_reference_front(reference_dataset)
    metrics = get_transfer_metrics(arrays["seed"], arrays["predicted"], arrays["simulated"], reference, chevron)
    result = TransferResult(
        {
            **metrics,
            **{key.removeprefix("transfer/"): value for key, value in run.data.metrics.items() if key.endswith("_seconds") and key.startswith("transfer/")},
        },
        arrays["deposition"],
        arrays["seed"],
        arrays["predicted"],
        arrays["simulated"],
        arrays.get("simulated_sd"),
        reference,
        chevron,
    )
    config = TransferConfig(**parse_transfer_params(run.data.params))
    log_transfer(run_id, result, config, with_plots)
    annotate_run(run_id)
    return result.metrics


def parse_transfer_params(params: dict[str, str]) -> dict:
    """The settings of a transfer test as the run logged them (`transfer.<name>`)."""
    import ast

    values = {key.removeprefix("transfer."): value for key, value in params.items() if key.startswith("transfer.")}
    parsed = {}
    for key, value in values.items():
        try:
            parsed[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed[key] = value
    return parsed


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = TransferConfig()
    parser = argparse.ArgumentParser(
        description="Transfer test: optimizes the model of a run, simulates the solutions found and compares them with the best front of the simulator"
    )
    parser.add_argument("--run-id", nargs="+", default=[], help="Training runs, the results are added to them as transfer/...")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(defaults.seeds), help="One NSGA-III run per seed")
    parser.add_argument("--population-size", type=int, default=defaults.population_size)
    parser.add_argument("--evaluations", type=int, default=defaults.evaluations, help="Evaluations of the model per NSGA-III run")
    parser.add_argument("--repeats", type=int, default=defaults.repeats, help="Simulations of every solution found")
    parser.add_argument("--reference-set", default=defaults.reference_set, help="Test set whose non-dominated solutions are the reference front")
    parser.add_argument("--n-jobs", type=int, help="Processes for the simulation (default: all cores)")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--simulator-runs",
        type=Path,
        help="Instead: the hypervolume ratio of each optimization run on the simulator in this directory (the yardstick), needs --bundle",
    )
    parser.add_argument("--bundle", help="Bundle of the reference set for --simulator-runs")
    parser.add_argument("--recompute", action="store_true", help="Compute the metrics of the earlier tests of --run-id again from their stored solutions")
    args = parser.parse_args(argv)
    if not args.run_id and not args.simulator_runs:
        parser.error("give --run-id or --simulator-runs")
    if args.simulator_runs and not args.bundle:
        parser.error("--simulator-runs needs --bundle")
    return args


def main(argv: list[str] | None = None):
    args = get_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.simulator_runs:
        reference = load_dataset(load_bundle(args.bundle).tests[args.reference_set])
        chevron = get_chevron_objectives(reference.material[:1], n_jobs=args.n_jobs)
        metrics = get_simulator_run_metrics(find_result_files(args.simulator_runs), reference, chevron)
        print(f"{len(metrics['hv_ratio'])} runs on the simulator, Chevron F1 {chevron[0]:.4f} F2 {chevron[1]:.3f}:")
        for name, values in metrics.items():
            print(f"  {name:<22} mean {values.mean():.4f}, min {values.min():.4f}, max {values.max():.4f}")
    if args.recompute:
        for run_id in args.run_id:
            metrics = recompute_transfer(run_id, not args.no_plots)
            print(f"{run_id}: better than Chevron {metrics['chevron_beaten_rate']:.1%}, Chevron hypervolume {metrics['chevron_hv']:.4f}")
        return

    config = TransferConfig(tuple(args.seeds), args.population_size, args.evaluations, args.repeats, args.reference_set, args.n_jobs)
    for run_id in args.run_id:
        metrics = transfer_run(run_id, config, not args.no_plots).metrics
        print(
            f"{run_id}: hypervolume ratio {metrics['hv_ratio']:.4f} (per run {metrics['hv_ratio_run_mean']:.4f}), IGD+ {metrics['igd_plus']:.4f}, "
            f"predicted {metrics['predicted_hv_ratio']:.4f}, bias F1 {metrics['F1/bias']:.4f} F2 {metrics['F2/bias']:.3f}, "
            f"negative {metrics['negative_rate']:.1%}"
        )


if __name__ == "__main__":
    main()
