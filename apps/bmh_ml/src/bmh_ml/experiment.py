"""NSGA-III runs over a Latin hypercube design of algorithm settings, shared by the optimization on the surrogate and on the simulation.

All runs optimize the deposition for the same fixed material (the first row of the training data). This is deliberate: the runs are
replicates. They repeat the optimization with different random seeds to cover the randomness of the algorithm (and, for the simulation,
the randomness of the simulation itself), instead of varying the problem. Every result file stores the material it was made for.
"""

import argparse
import json
import logging
import os
from collections import Counter

import numpy as np
from pyDOE3 import lhs
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.reference_direction import UniformReferenceDirectionFactory

from bmh_ml.settings import OUTPUT_DIR, TRAINING_DATA_FILE

DESIGN_SEED = 42


def add_experiment_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument("--runs", type=int, default=30, help="Number of runs, each with settings from the Latin hypercube design")
    parser.add_argument("--evaluations", type=int, nargs="+", default=[20000, 100000], help="Numbers of function evaluations to choose from")
    parser.add_argument("--population-sizes", type=int, nargs="+", default=[50, 100, 200], help="Population sizes to choose from")
    parser.add_argument("--training-data", default=TRAINING_DATA_FILE, help="CSV file, its first row provides the fixed material of all runs")


def get_design(runs: int, evaluations: list[int], population_sizes: list[int]) -> list[tuple[int, int, int]]:
    """Latin hypercube design of the settings of every run: the number of evaluations, the population size and the replicate.

    The replicate counts the runs with the same settings (0 for the first of them). It only tells the runs apart, the randomness of a run
    comes from its seed. With the Latin hypercube the number of replicates of a setting varies.
    """
    # The design has a third dimension that used to be the "instance" of the run. It is drawn although it is not used, because leaving it out
    # would change the settings assigned to the runs, which are the settings of the experiments made so far.
    samples = lhs(n=3, samples=runs, seed=DESIGN_SEED)
    settings = [
        (int(evaluations[int(samples[run, 0] * len(evaluations))]), int(population_sizes[int(samples[run, 1] * len(population_sizes))])) for run in range(runs)
    ]

    replicates: Counter[tuple[int, int]] = Counter()
    design = []
    for setting in settings:
        design.append((*setting, replicates[setting]))
        replicates[setting] += 1
    return design


def get_initial_population(initial: np.ndarray, population_size: int, problem: Problem, seed: int) -> np.ndarray:
    """Up to `population_size` of the given solutions, the rest uniformly random within the bounds of the problem."""
    rng = np.random.default_rng(seed)
    given = np.unique(np.asarray(initial, dtype=float), axis=0)
    if len(given) > population_size:
        given = given[rng.choice(len(given), population_size, replace=False)]
    random = rng.uniform(problem.xl, problem.xu, (population_size - len(given), problem.n_var))
    return np.vstack([given, random])


def make_nsga3(problem: Problem, population_size: int, seed: int, initial: np.ndarray | None = None, n_offsprings: int | None = None) -> NSGA3:
    """NSGA-III as every optimization of bmh_ml runs it: one reference direction per solution of the population (two objectives), SBX
    crossover (probability 0.9, eta 15), polynomial mutation (eta 20), no duplicates. The first population is random, or starts from the
    solutions `initial` (`get_initial_population`). `n_offsprings`: offspring per generation, default the population size."""
    sampling = FloatRandomSampling() if initial is None else get_initial_population(initial, population_size, problem, seed)
    return NSGA3(
        pop_size=population_size,
        n_offsprings=n_offsprings,
        ref_dirs=UniformReferenceDirectionFactory(2, n_points=population_size).do(),
        sampling=sampling,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )


def optimize(
    problem: Problem, population_size: int, evaluations: int, seed: int, initial: np.ndarray | None = None, callback=None
) -> tuple[np.ndarray, np.ndarray]:
    """NSGA-III with `evaluations` evaluations in all (the first generation included). `initial`: solutions to start from, e.g. the front
    found at another detail level, instead of a random population. `callback`: a pymoo callback, called after every generation."""
    generations = evaluations // population_size
    if generations < 1:
        raise ValueError(f"{evaluations} evaluations are not enough for one generation of {population_size} solutions")
    algorithm = make_nsga3(problem, population_size, seed, initial)
    options = {} if callback is None else {"callback": callback}  # None would replace pymoo's default callback
    result = minimize(problem, algorithm, ("n_gen", generations), seed=seed, verbose=False, **options)
    return result.F, result.X


def run_experiments(
    problem: Problem, model_name: str, args: argparse.Namespace, material_variables: np.ndarray, extra_parameters: dict | None = None
) -> list[str]:
    """Runs all runs of the design and writes the resulting fronts to output/<model name>, returns the paths of the files.

    The seed of a run is the design seed plus its run id, so a run can be repeated exactly. The optimization on the surrogate then gives the same
    result again, the simulation is random itself and gives a slightly different one.
    """
    output_directory = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_directory, exist_ok=True)

    paths = []
    for run_id, (evaluations, population_size, replicate) in enumerate(get_design(args.runs, args.evaluations, args.population_sizes)):
        logging.info(f"Running run {run_id + 1} (replicate {replicate}), evals={evaluations}, pop={population_size}")

        objectives, variables = optimize(problem, population_size, evaluations, seed=DESIGN_SEED + run_id)

        results_data = {
            "model": model_name,
            "material_variables": [float(value) for value in material_variables],
            "objectives": objectives.tolist(),
            "variables": variables.tolist(),
            "parameters": {
                "run_id": run_id,
                "replicate": replicate,
                "population_size": population_size,
                "n_evaluations": evaluations,
                **(extra_parameters or {}),
            },
        }
        path = os.path.join(output_directory, f"run_{run_id:02d}_evals{evaluations}_pop{population_size}_replicate{replicate:02d}.json")
        with open(path, "w") as f:
            json.dump(results_data, f, indent=4)

        logging.info(f"Data saved to {path}")
        paths.append(path)
    return paths
