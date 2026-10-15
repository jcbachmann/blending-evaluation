import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np

from bmh_ml.datasets.generators import find_result_files
from bmh_ml.datasets.manifest import Dataset, concatenate_datasets
from bmh_ml.datasets.store import get_bundle_simulator, load_bundle, load_dataset, use_bundle_simulator
from bmh_ml.evaluate_run import load_run_model
from bmh_ml.evaluation.chevron import get_chevron_objectives
from bmh_ml.evaluation.transfer import (
    TransferConfig,
    TransferResult,
    combine_results,
    get_reference_front,
    get_simulator_run_metrics,
    get_transfer_metrics,
    log_transfer,
    run_simulator_baseline,
    run_transfer_test,
    split_by_material,
)
from bmh_ml.settings import use_simulator_settings
from bmh_ml.tracking.annotate import annotate_run
from bmh_ml.tracking.runs import configure_mlflow


def load_references(specs: list[str]) -> tuple[Dataset, str]:
    """The reference sets `BUNDLE:SET` of other bundles as one dataset (the materials of all of them), with their simulator settings
    activated: the found solutions and Chevron are simulated at the level of the references. Returns the dataset and the scope."""
    parts, levels, scopes = [], set(), set()
    for spec in specs:
        bundle_name, separator, set_name = spec.partition(":")
        if not separator:
            raise ValueError(f"'{spec}' is not of the form BUNDLE:SET")
        bundle = load_bundle(bundle_name)
        parts.append(load_dataset(bundle.tests[set_name]))
        levels.add(get_bundle_simulator(bundle))
        scopes.add(bundle.scope)
    if len(levels) > 1 or len(scopes) > 1:
        raise ValueError(f"The references {specs} mix simulator settings or scopes: {levels}, {scopes}")
    use_simulator_settings(levels.pop())
    return concatenate_datasets("+".join(specs), parts), scopes.pop()


def transfer_run_against(run_id: str, references: list[str], label: str, config: TransferConfig, with_plots: bool = True) -> TransferResult:
    """The transfer test of a run's model against the reference fronts of other bundles, at their level (a model trained at one level, its
    solutions simulated at another; or more materials than the run's own bundle has). Logged as `transfer_<label>/...`."""
    reference, scope = load_references(references)
    result = run_transfer_test(load_run_model(run_id), scope, reference, config)
    prefix = f"transfer_{label}"
    configure_mlflow().MlflowClient().log_param(run_id, f"{prefix}.references", " ".join(references))
    log_transfer(run_id, result, config, with_plots, prefix)
    return result


def transfer_run(run_id: str, config: TransferConfig, with_plots: bool = True) -> TransferResult:
    """Runs the transfer test on the model of a training run and adds the results to the run."""
    mlflow = configure_mlflow()
    run = mlflow.get_run(run_id)
    bundle = load_bundle(run.data.params["bundle"])
    use_bundle_simulator(bundle)
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
    use_bundle_simulator(bundle)  # Chevron at the level of the test
    reference_dataset = load_dataset(bundle.tests[run.data.params.get("transfer.reference_set", "T2")])
    with tempfile.TemporaryDirectory() as directory:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="transfer/solutions.npz", dst_path=directory)
        with np.load(path) as stored:
            arrays = {name: stored[name] for name in stored.files}
    parts = split_by_material(reference_dataset)
    index = arrays.get("material_index", np.zeros(len(arrays["seed"]), dtype=int))
    results = []
    for i, (material, part) in enumerate(parts):
        rows = index == i
        chevron = get_chevron_objectives(material)
        reference = get_reference_front(part)
        metrics = get_transfer_metrics(arrays["seed"][rows], arrays["predicted"][rows], arrays["simulated"][rows], reference, chevron)
        sd = arrays["simulated_sd"][rows] if "simulated_sd" in arrays else None
        results.append(
            TransferResult(
                metrics, arrays["deposition"][rows], arrays["seed"][rows], arrays["predicted"][rows], arrays["simulated"][rows], sd, reference, chevron
            )
        )
    result = combine_results(results)
    times = {key.removeprefix("transfer/"): value for key, value in run.data.metrics.items() if key.endswith("_seconds") and key.startswith("transfer/")}
    result.metrics.update(times)
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


def load_run_fronts(run_id: str, label: str = "") -> dict[bytes, np.ndarray]:
    """The solutions a run's model found in its transfer test `transfer_<label>` (`transfer` without a label), by material (its bytes): those
    the model predicts as non-dominated among all its optimization runs of that material. Only the model's predictions select them."""
    import tempfile

    from bmh_ml.evaluation.transfer import get_nondominated
    from bmh_ml.settings import get_simulator_settings

    mlflow = configure_mlflow()
    params = mlflow.get_run(run_id).data.params
    prefix = f"transfer_{label}" if label else "transfer"
    if f"{prefix}.references" in params:
        level = get_simulator_settings()
        reference, _ = load_references(params[f"{prefix}.references"].split())
        use_simulator_settings(level)  # only the materials are needed here, the level stays
    else:
        reference = load_dataset(load_bundle(params["bundle"]).tests[params.get(f"{prefix}.reference_set", "T2")])
    with tempfile.TemporaryDirectory() as directory:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=f"{prefix}/solutions.npz", dst_path=directory)
        with np.load(path) as stored:
            deposition, predicted = stored["deposition"], stored["predicted"]
            index = stored["material_index"] if "material_index" in stored.files else np.zeros(len(deposition), dtype=int)
    fronts = {}
    for i, (material, _) in enumerate(split_by_material(reference)):
        rows = np.flatnonzero(index == i)
        fronts[material[0].tobytes()] = deposition[rows][get_nondominated(predicted[rows])]
    return fronts


def load_initial_populations(specs: list[str]) -> dict[bytes, np.ndarray]:
    """The starting populations of seeded optimizations by material (its bytes): the depositions of the sets `BUNDLE:SET`, or with
    `run:<run id>[:<label>]` the solutions a model found in its transfer test (`load_run_fronts`)."""
    initial: dict[bytes, list[np.ndarray]] = {}
    for spec in specs:
        if spec.startswith("run:"):
            _, run_id, label = [*spec.split(":"), ""][:3]
            for key, deposition in load_run_fronts(run_id, label).items():
                initial.setdefault(key, []).append(deposition)
            continue
        bundle_name, _, set_name = spec.partition(":")
        for material, part in split_by_material(load_dataset(load_bundle(bundle_name).tests[set_name])):
            initial.setdefault(material[0].tobytes(), []).append(part.deposition)
    return {key: np.vstack(value) for key, value in initial.items()}


def save_prescreen_models(specs: list[str], scope: str, candidates: int, simulated: int, directory: Path):
    """The models of the runs saved into `directory` (worker processes load them from there) as a `prescreen.Prescreen`. `specs`: one entry
    for both objectives or two (F1, then F2), each a run id or several, comma-separated, whose models are averaged (an ensemble)."""
    from bmh_ml.prescreen import Prescreen

    if not 1 <= len(specs) <= 2:
        raise ValueError("--prescreen takes one entry (both objectives) or two (F1, then F2)")
    paths: dict[str, str] = {}
    for run_id in dict.fromkeys(run_id for spec in specs for run_id in spec.split(",")):
        path = directory / f"model-{len(paths)}"
        load_run_model(run_id).save(path)
        paths[run_id] = str(path)
    f1, f2 = (tuple(paths[run_id] for run_id in spec.split(",")) for spec in (specs[0], specs[-1]))
    return Prescreen(f1, f2, scope, candidates, simulated)


def get_seeded_suffix(seed_population: list[str]) -> str:
    """The run name suffix of a seeded baseline: `-seeded`, and the names of the runs whose model fronts seed it."""
    mlflow = configure_mlflow()
    sources = [mlflow.get_run(spec.split(":")[1]).info.run_name for spec in seed_population if spec.startswith("run:")]
    return "-seeded" + (f"-from-{'+'.join(sources)}" if sources else "")


def simulator_baseline(
    bundle_name: str | None,
    config: TransferConfig,
    with_plots: bool = True,
    references: list[str] | None = None,
    label: str = "",
    seed_population: list[str] | None = None,
    prescreen: list[str] | None = None,
    candidates: int = 300,
    simulated: int = 100,
) -> str:
    """NSGA-III directly on the simulator with the budget `config.evaluations` per run, logged as a run with the transfer test's metrics:
    for the reference set of a bundle, or for the reference sets `references` of several bundles (at their level, run name with `label`).
    `seed_population`: sets `BUNDLE:SET` whose solutions of each material start its runs (e.g. the fronts found at another level), or
    `run:<run id>[:<label>]`, the fronts a model found (`load_run_fronts`). `prescreen`: the training runs whose models rate the offspring
    (one for both objectives, or one for F1 and one for F2); every generation makes `candidates` offspring and simulates the `simulated`
    best (`prescreen.optimize_prescreened`)."""
    from bmh_ml.tracking.annotate import NOTE
    from bmh_ml.tracking.environment import get_code_version
    from bmh_ml.tracking.runs import get_experiment_id, get_experiment_name

    if references:
        reference, scope = load_references(references)
        params = {"references": " ".join(references), "scope": scope}
    else:
        bundle = load_bundle(bundle_name)
        use_bundle_simulator(bundle)
        reference, scope = load_dataset(bundle.tests[config.reference_set]), bundle.scope
        params = {"bundle": bundle.name, "scope": scope}
    initial = load_initial_populations(seed_population) if seed_population else None
    if initial:
        missing = [material for material, _ in split_by_material(reference) if material[0].tobytes() not in initial]
        if missing:
            raise ValueError(f"{len(missing)} materials of the references have no starting population in {seed_population}")
        params["seed_population"] = " ".join(seed_population)
    with tempfile.TemporaryDirectory() as directory:
        screen = None
        if prescreen:
            screen = save_prescreen_models(prescreen, scope, candidates, simulated, Path(directory))
            params |= {"prescreen": " ".join(prescreen), "prescreen_candidates": candidates, "prescreen_simulated": simulated}
        result = run_simulator_baseline(reference, config, initial, screen)
    mlflow = configure_mlflow()
    name = f"simulator-nsga3-{config.evaluations}" + (f"-{label}" if label else "") + (get_seeded_suffix(seed_population) if initial else "")
    if prescreen:
        name += f"-prescreened-{candidates}to{simulated}"
    with mlflow.start_run(experiment_id=get_experiment_id(get_experiment_name(scope)), run_name=name) as run:
        mlflow.log_params({"optimizer": "NSGA-III on the simulator", **params})
        mlflow.log_metrics({"budget/simulations": float(config.evaluations), "budget/simulations_all_runs": float(config.evaluations * len(config.seeds))})
        m = result.metrics
        mlflow.set_tags(
            {
                "code_version": get_code_version(),
                "purpose": "simulator baseline",
                NOTE: f"{name} | transfer hv {m['hv_ratio']:.3f}; per run {m['hv_ratio_run_mean']:.3f}; "
                f"{100 * m['chevron_beaten_rate']:.0f} % beat Chevron\n\nThe yardstick of the transfer test: NSGA-III directly on the simulator, "
                f"{len(config.seeds)} runs (seeds {list(config.seeds)}) of {config.evaluations:,} simulations each (population {config.population_size}), "
                f"the final fronts simulated {config.repeats} times and scored like the transfer test of a model. `transfer/predicted_*` and the "
                "bias are the single noisy simulations the optimizer saw against the repeated ones.",
            }
        )
    log_transfer(run.info.run_id, result, config, with_plots, f"transfer_{label}" if label else "transfer")  # labeled like the models it is compared with
    return run.info.run_id


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
    parser.add_argument(
        "--simulator-baseline", nargs="+", type=int, metavar="SIMULATIONS", help="Instead: NSGA-III on the simulator with these budgets per run, needs --bundle"
    )
    parser.add_argument("--recompute", action="store_true", help="Compute the metrics of the earlier tests of --run-id again from their stored solutions")
    parser.add_argument(
        "--references", nargs="+", metavar="BUNDLE:SET", help="Reference sets of other bundles instead of the run's own (their level, all their materials)"
    )
    parser.add_argument("--label", help="With --references: name of the test, the metrics are logged as transfer_<label>/...")
    parser.add_argument(
        "--seed-population",
        nargs="+",
        metavar="BUNDLE:SET",
        help="With --simulator-baseline: start each material's runs from its solutions in these sets, or with run:<run id>[:<label>] from the "
        "solutions a model predicts as non-dominated in that run's transfer test",
    )
    parser.add_argument(
        "--prescreen",
        nargs="+",
        metavar="RUN_IDS",
        help="With --simulator-baseline: models of these runs choose which offspring are simulated; one entry for both objectives or two "
        "(F1, then F2), each a run id or several comma-separated ones, whose predictions are averaged",
    )
    parser.add_argument("--candidates", type=int, default=300, help="With --prescreen: offspring made per generation")
    parser.add_argument("--simulated", type=int, default=100, help="With --prescreen: offspring simulated per generation, the models' best")
    args = parser.parse_args(argv)
    if args.references and not args.label:
        parser.error("--references needs --label")
    if not args.run_id and not args.simulator_runs and not args.simulator_baseline:
        parser.error("give --run-id or --simulator-runs")
    if args.simulator_runs and not args.bundle:
        parser.error("--simulator-runs needs --bundle")
    return args


def main(argv: list[str] | None = None):
    args = get_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.simulator_runs:
        bundle = load_bundle(args.bundle)
        use_bundle_simulator(bundle)
        reference = load_dataset(bundle.tests[args.reference_set])
        chevron = get_chevron_objectives(reference.material[:1], n_jobs=args.n_jobs)
        metrics = get_simulator_run_metrics(find_result_files(args.simulator_runs), reference, chevron)
        print(f"{len(metrics['hv_ratio'])} runs on the simulator, Chevron F1 {chevron[0]:.4f} F2 {chevron[1]:.3f}:")
        for name, values in metrics.items():
            print(f"  {name:<22} mean {values.mean():.4f}, min {values.min():.4f}, max {values.max():.4f}")
    if args.simulator_baseline:
        if not args.bundle and not args.references:
            raise SystemExit("--simulator-baseline needs --bundle or --references")
        for budget in args.simulator_baseline:
            config = TransferConfig(tuple(args.seeds), args.population_size, budget, args.repeats, args.reference_set, args.n_jobs)
            run_id = simulator_baseline(
                args.bundle, config, not args.no_plots, args.references, args.label or "", args.seed_population, args.prescreen, args.candidates, args.simulated
            )
            print(f"simulator NSGA-III, {budget} simulations per run: run {run_id}")
        return
    if args.recompute:
        for run_id in args.run_id:
            metrics = recompute_transfer(run_id, not args.no_plots)
            print(f"{run_id}: better than Chevron {metrics['chevron_beaten_rate']:.1%}, Chevron hypervolume {metrics['chevron_hv']:.4f}")
        return

    config = TransferConfig(tuple(args.seeds), args.population_size, args.evaluations, args.repeats, args.reference_set, args.n_jobs)
    for run_id in args.run_id:
        if args.references:
            metrics = transfer_run_against(run_id, args.references, args.label, config, not args.no_plots).metrics
            share, per_run = metrics["chevron_hv_ratio"], metrics["chevron_hv_ratio_run_mean"]
            print(f"{run_id}: {args.label}: share of the improvement over Chevron {share:.4f} (per run {per_run:.4f})")
            continue
        metrics = transfer_run(run_id, config, not args.no_plots).metrics
        print(
            f"{run_id}: hypervolume ratio {metrics['hv_ratio']:.4f} (per run {metrics['hv_ratio_run_mean']:.4f}), IGD+ {metrics['igd_plus']:.4f}, "
            f"predicted {metrics['predicted_hv_ratio']:.4f}, bias F1 {metrics['F1/bias']:.4f} F2 {metrics['F2/bias']:.3f}, "
            f"negative {metrics['negative_rate']:.1%}"
        )


if __name__ == "__main__":
    main()
