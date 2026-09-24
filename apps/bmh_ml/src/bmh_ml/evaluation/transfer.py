"""The transfer test: is a model useful for optimization? Optimize the model, simulate the solutions it finds, and compare them with the best
front known from optimizing the simulator itself.

The reference front is the non-dominated part of the test set T2 (the solutions of the optimization runs on the simulator, labeled with the
mean of repeated simulations). The solutions found on the model are simulated with as many repeats, so both are judged without the noise of
the simulator, which would otherwise favor whichever solutions were lucky once.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bmh_ml.datasets.manifest import Dataset, make_features
from bmh_ml.datasets.simulate import simulate
from bmh_ml.evaluation.noise import OBJECTIVES
from bmh_ml.models.base import Model
from bmh_ml.settings import DEPOSITION_LENGTH, X_MAX, X_MIN

REFERENCE_POINT = 1.1  # in objectives normalized to the reference front (0 its best, 1 its worst value), so its extremes count as well


@dataclass
class TransferConfig:
    """A fixed budget, so the results of different models are comparable: one optimization per seed, the fronts simulated `repeats` times."""

    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    population_size: int = 100
    evaluations: int = 20_000
    repeats: int = 16
    reference_set: str = "T2"
    n_jobs: int | None = None

    def params(self) -> dict[str, str]:
        return {f"transfer.{key}": str(value) for key, value in self.__dict__.items() if key != "n_jobs"}


@dataclass
class TransferResult:
    metrics: dict[str, float]
    deposition: np.ndarray  # the solutions found on the model, one row each
    seed: np.ndarray  # the seed of the optimization that found each solution
    predicted: np.ndarray  # objectives the model predicted for them
    simulated: np.ndarray  # objectives of the simulator, mean of the repeats
    simulated_sd: np.ndarray | None
    reference: np.ndarray  # the reference front


def get_problem_class():
    from pymoo.core.problem import Problem

    class ModelProblem(Problem):
        """Optimizes the deposition for a fixed material with the objectives predicted by a model of any scope."""

        def __init__(self, model: Model, scope: str, material: np.ndarray):
            self.model, self.scope, self.material = model, scope, material
            super().__init__(n_var=DEPOSITION_LENGTH, n_obj=2, xl=np.full(DEPOSITION_LENGTH, X_MIN), xu=np.full(DEPOSITION_LENGTH, X_MAX))

        def _evaluate(self, x, out, *_args, **_kwargs):
            out["F"] = self.model.predict(make_features(self.scope, self.material, x))

    return ModelProblem


def optimize_model(
    model: Model, scope: str, material: np.ndarray, seeds: tuple[int, ...], population_size: int, evaluations: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The final fronts of one NSGA-III run per seed on the model: depositions, predicted objectives and the seed of each solution."""
    from bmh_ml.experiment import optimize

    problem = get_problem_class()(model, scope, np.atleast_2d(material))
    depositions, predictions, run_seeds = [], [], []
    for seed in seeds:
        objectives, variables = optimize(problem, population_size, evaluations, seed)
        depositions.append(np.atleast_2d(variables))
        predictions.append(np.atleast_2d(objectives))
        run_seeds.append(np.full(len(depositions[-1]), seed))
    return np.vstack(depositions), np.vstack(predictions), np.concatenate(run_seeds)


def get_nondominated(objectives: np.ndarray) -> np.ndarray:
    """The rows of the non-dominated solutions (both objectives are minimized)."""
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    return np.sort(NonDominatedSorting().do(objectives, only_non_dominated_front=True))


def get_reference_front(dataset: Dataset) -> np.ndarray:
    return dataset.y[get_nondominated(dataset.y)]


def normalize(objectives: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ideal, nadir = reference.min(axis=0), reference.max(axis=0)
    return (objectives - ideal) / np.where(nadir > ideal, nadir - ideal, 1.0)


def get_hypervolume(objectives: np.ndarray, reference: np.ndarray) -> float:
    """Hypervolume of the solutions in objectives normalized to the reference front, solutions beyond the reference point add nothing."""
    from pymoo.indicators.hv import HV

    points = normalize(objectives, reference)
    points = points[np.all(points < REFERENCE_POINT, axis=1)]
    if len(points) == 0:
        return 0.0
    return float(HV(ref_point=np.full(2, REFERENCE_POINT))(points))


def get_front_metrics(objectives: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """`hv_ratio`: hypervolume of the solutions divided by that of the reference front, 1 is as good, above 1 better.
    `igd_plus`: distance of the reference front to the solutions (normalized objectives, 0 means the reference front is reached)."""
    from pymoo.indicators.igd_plus import IGDPlus

    return {
        "hv_ratio": get_hypervolume(objectives, reference) / get_hypervolume(reference, reference),
        "igd_plus": float(IGDPlus(normalize(reference, reference))(normalize(objectives, reference))),
    }


def run_transfer_test(model: Model, scope: str, reference_dataset: Dataset, config: TransferConfig) -> TransferResult:
    """Optimizes the model for the material of the reference dataset, simulates the solutions found and compares them with the reference front."""
    material = reference_dataset.material[:1]
    reference = get_reference_front(reference_dataset)

    start = time.perf_counter()
    deposition, predicted, seeds = optimize_model(model, scope, material, config.seeds, config.population_size, config.evaluations)
    optimize_seconds = time.perf_counter() - start
    start = time.perf_counter()
    simulated, simulated_sd = simulate(material, deposition, config.repeats, config.n_jobs)
    simulate_seconds = time.perf_counter() - start

    metrics = get_front_metrics(simulated, reference)
    per_run = [get_front_metrics(simulated[seeds == seed], reference)["hv_ratio"] for seed in config.seeds]
    metrics.update(
        {
            "hv_ratio_run_mean": float(np.mean(per_run)),
            "hv_ratio_run_min": float(np.min(per_run)),
            "predicted_hv_ratio": get_front_metrics(predicted, reference)["hv_ratio"],
            "negative_rate": float(np.mean(np.any(predicted < 0, axis=1))),
            "solutions": float(len(deposition)),
            "optimize_seconds": optimize_seconds,
            "simulate_seconds": simulate_seconds,
        }
    )
    for i, objective in enumerate(OBJECTIVES):
        error = predicted[:, i] - simulated[:, i]
        metrics[f"{objective}/bias"] = float(np.mean(error))  # negative: the model promises better values than the simulator gives
        metrics[f"{objective}/rmse"] = float(np.sqrt(np.mean(error**2)))
    return TransferResult(metrics, deposition, seeds, predicted, simulated, simulated_sd, reference)


def get_simulator_run_ratios(result_files: list[Path], reference_dataset: Dataset) -> np.ndarray:
    """The hypervolume ratio of each optimization run on the simulator, the yardstick for `hv_ratio_run_mean`.

    Its solutions are judged by their labels in the reference dataset (which contains the solutions of these runs), not by the single
    simulation the run saw, so the ratio is free of the luck of the noise.
    """
    labels = {deposition.tobytes(): y for deposition, y in zip(reference_dataset.deposition, reference_dataset.y, strict=True)}
    reference = get_reference_front(reference_dataset)
    ratios = []
    for file in result_files:
        variables = np.array(json.loads(file.read_text())["variables"], dtype=float)[:, -DEPOSITION_LENGTH:]
        missing = [row for row in variables if row.tobytes() not in labels]
        if missing:
            raise ValueError(f"{len(missing)} solutions of {file.name} are not in the reference dataset {reference_dataset.name}")
        ratios.append(get_front_metrics(np.array([labels[row.tobytes()] for row in variables]), reference)["hv_ratio"])
    return np.array(ratios)


def plot_transfer(result: TransferResult):
    """The solutions found on the model, as the model predicted them and as the simulator evaluates them, with the reference front."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    order = np.argsort(result.reference[:, 0])
    axis.plot(result.reference[order, 0], result.reference[order, 1], color="black", marker="o", markersize=3, label="reference front (simulator)")
    axis.scatter(result.predicted[:, 0], result.predicted[:, 1], s=6, alpha=0.4, label="found on the model, predicted")
    axis.scatter(result.simulated[:, 0], result.simulated[:, 1], s=6, alpha=0.6, label="found on the model, simulated")
    metrics = result.metrics
    axis.set(
        xlabel="F1",
        ylabel="F2",
        title=f"Transfer: hypervolume ratio {metrics['hv_ratio']:.3f} (per run {metrics['hv_ratio_run_mean']:.3f}), IGD+ {metrics['igd_plus']:.3f}",
    )
    axis.legend()
    return figure


def log_transfer(result: TransferResult, config: TransferConfig, with_plots: bool = True) -> None:
    """Logs the metrics (`transfer/...`), the settings and the solutions of a transfer test to the active MLflow run."""
    import tempfile

    from bmh_ml.evaluation.plots import close_figure
    from bmh_ml.tracking.runs import configure_mlflow, get_finite_metrics

    mlflow = configure_mlflow()
    mlflow.log_params(config.params())
    mlflow.log_metrics(get_finite_metrics({f"transfer/{name}": value for name, value in result.metrics.items()}))
    with tempfile.TemporaryDirectory() as directory:
        file = Path(directory) / "solutions.npz"
        arrays = {"deposition": result.deposition, "seed": result.seed, "predicted": result.predicted, "simulated": result.simulated}
        if result.simulated_sd is not None:
            arrays["simulated_sd"] = result.simulated_sd
        np.savez_compressed(file, reference=result.reference, **arrays)
        mlflow.log_artifact(str(file), "transfer")
    if with_plots:
        figure = plot_transfer(result)
        mlflow.log_figure(figure, "transfer/front.png")
        close_figure(figure)
