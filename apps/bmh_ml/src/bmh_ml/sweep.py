"""Hyperparameter sweeps with Optuna: every trial is a training run, nested in a parent run of the sweep.

The objective is a mean of validation metrics: test sets are for reporting, choosing models on them would overfit them unnoticed. The study
is stored in the store (`optuna.db`), so running the same sweep name again continues it with more trials.
"""

import argparse
import logging
from dataclasses import dataclass, field

import numpy as np

from bmh_ml.datasets.store import get_evaluation_sets, load_bundle
from bmh_ml.models.registry import MODELS
from bmh_ml.tracking.runs import configure_mlflow, get_experiment_id, get_experiment_name
from bmh_ml.tracking.store import get_store
from bmh_ml.train import parse_parameters, run_training

# name: ("float" | "int", low, high, log) or ("categorical", choices)
SEARCH_SPACES: dict[str, dict[str, tuple]] = {
    "ridge": {"alpha": ("float", 1e-3, 1e3, True)},
    "lightgbm": {
        "learning_rate": ("float", 0.01, 0.2, True),
        "num_leaves": ("int", 15, 511, True),
        "min_child_samples": ("int", 5, 200, True),
        "subsample": ("float", 0.5, 1.0, False),
        "colsample_bytree": ("float", 0.5, 1.0, False),
    },
    "mlp": {
        "width": ("categorical", [64, 128, 256, 512, 1024]),
        "depth": ("int", 2, 6, False),
        "learning_rate": ("float", 1e-4, 3e-3, True),
        "batch_size": ("categorical", [128, 256, 512, 1024]),
        "activation": ("categorical", ["relu", "gelu", "silu"]),
        "dropout": ("float", 0.0, 0.2, False),
    },
}
OBJECTIVE_METRIC = "nrmse"


@dataclass
class SweepConfig:
    name: str
    bundle: str
    model: str
    trials: int = 20
    objective: list[str] = field(default_factory=list)  # empty: the nrmse of F1 and F2 on every validation set of the bundle
    fixed: dict = field(default_factory=dict)  # parameters that are not searched
    seed: int = 1
    timeout: float | None = None


def suggest(trial, space: dict[str, tuple], fixed: dict) -> dict:
    """The parameters of a trial: the fixed ones and a suggestion for each other one of the search space."""
    params = dict(fixed)
    for name, (kind, *args) in space.items():
        if name in fixed:
            continue
        if kind == "categorical":
            params[name] = trial.suggest_categorical(name, args[0])
        elif kind == "float":
            params[name] = trial.suggest_float(name, args[0], args[1], log=args[2])
        elif kind == "int":
            params[name] = trial.suggest_int(name, args[0], args[1], log=args[2])
        else:
            raise ValueError(f"Unknown kind '{kind}' of parameter {name}")
    return params


def get_objective_metrics(config: SweepConfig, validation_sets: list[str]) -> list[str]:
    if not config.objective:
        return [f"{name}/{objective}/{OBJECTIVE_METRIC}" for name in validation_sets for objective in ("F1", "F2")]
    not_validation = [metric for metric in config.objective if metric.split("/")[0] not in validation_sets]
    if not_validation:
        raise ValueError(f"The objective may only use the validation sets {validation_sets}, not {not_validation}: test sets are for reporting")
    return list(config.objective)


def run_sweep(config: SweepConfig):
    """Runs the trials and returns the Optuna study. The parent run is tagged with the run of the best trial."""
    import optuna

    if config.model not in SEARCH_SPACES:
        raise ValueError(f"No search space for model '{config.model}', known: {sorted(SEARCH_SPACES)}")
    bundle = load_bundle(config.bundle)
    validation_sets = [name for name in get_evaluation_sets(bundle) if name not in bundle.tests]
    metrics = get_objective_metrics(config, validation_sets)
    storage = f"sqlite:///{(get_store() / 'optuna.db').as_posix()}"
    sampler = optuna.samplers.TPESampler(seed=config.seed)
    study = optuna.create_study(study_name=config.name, storage=storage, direction="minimize", sampler=sampler, load_if_exists=True)

    mlflow = configure_mlflow()
    with mlflow.start_run(experiment_id=get_experiment_id(get_experiment_name(bundle.scope)), run_name=f"sweep-{config.name}"):
        mlflow.log_params({"sweep.model": config.model, "sweep.bundle": config.bundle, "sweep.trials": config.trials, "sweep.seed": config.seed})
        mlflow.log_params({"sweep.objective": ",".join(metrics), **{f"sweep.fixed.{key}": value for key, value in config.fixed.items()}})
        mlflow.set_tags({"sweep": config.name})

        def objective(trial) -> float:
            params = suggest(trial, SEARCH_SPACES[config.model], config.fixed)
            tags = {"sweep": config.name, "trial": str(trial.number)}
            result = run_training(config.bundle, config.model, params, config.seed, f"{config.name}-t{trial.number}", with_plots=False, nested=True, tags=tags)
            value = float(np.mean([result.metrics[metric] for metric in metrics]))
            trial.set_user_attr("run_id", result.run_id)
            mlflow.log_metric("sweep/objective", value, step=trial.number)
            return value

        study.optimize(objective, n_trials=config.trials, timeout=config.timeout)
        best = study.best_trial
        mlflow.log_metric("sweep/best_objective", best.value)
        mlflow.log_dict({"value": best.value, "params": best.params, "trial": best.number}, "best_trial.json")
        mlflow.set_tags({"best_run": best.user_attrs["run_id"], "best_trial": str(best.number)})
    return study


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter sweep with Optuna, every trial is a nested training run")
    parser.add_argument("--name", required=True, help="Name of the sweep, the same name continues it")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--model", required=True, choices=sorted(set(SEARCH_SPACES) & set(MODELS)))
    parser.add_argument("--trials", type=int, default=20, help="Trials to add in this call")
    parser.add_argument("--objective", nargs="*", default=[], help="Validation metrics to minimize, their mean (default: val*/F1|F2/nrmse)")
    parser.add_argument("--param", nargs="*", default=[], metavar="KEY=VALUE", help="Fixed parameters, not searched")
    parser.add_argument("--seed", type=int, default=1, help="Seed of the sampler and of every training")
    parser.add_argument("--timeout", type=float, help="Stop starting new trials after this many seconds")
    args = parser.parse_args(argv)
    args.fixed = parse_parameters(args.param)
    return args


def main(argv: list[str] | None = None):
    args = get_args(argv)
    logging.basicConfig(level=logging.INFO)
    config = SweepConfig(args.name, args.bundle, args.model, args.trials, args.objective, args.fixed, args.seed, args.timeout)
    study = run_sweep(config)
    best = study.best_trial
    print(f"Best of {len(study.trials)} trials: {best.value:.4f} (trial {best.number}, run {best.user_attrs['run_id']})")
    for name, value in sorted(best.params.items()):
        print(f"  {name} = {value}")


if __name__ == "__main__":
    main()
