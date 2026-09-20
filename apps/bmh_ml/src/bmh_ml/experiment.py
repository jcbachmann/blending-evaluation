"""NSGA-III runs over a Latin hypercube design of algorithm settings, shared by the optimization on the surrogate and on the simulation."""

import argparse
import json
import logging
import os

import numpy as np
from pyDOE3 import lhs
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.reference_direction import UniformReferenceDirectionFactory

from bmh_ml.settings import OUTPUT_DIR

DESIGN_SEED = 42


def add_experiment_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument("--runs", type=int, default=30, help="Number of runs, each with settings from the Latin hypercube design")
    parser.add_argument("--evaluations", type=int, nargs="+", default=[20000, 100000], help="Numbers of function evaluations to choose from")
    parser.add_argument("--population-sizes", type=int, nargs="+", default=[50, 100, 200], help="Population sizes to choose from")


def get_design(runs: int, evaluations: list[int], population_sizes: list[int]) -> list[tuple[int, int, int]]:
    """Latin hypercube design, the number of evaluations, the population size and the instance id of every run."""
    samples = lhs(n=3, samples=runs, seed=DESIGN_SEED)
    instance_ids = np.arange(runs)
    return [
        (
            int(evaluations[int(samples[run, 0] * len(evaluations))]),
            int(population_sizes[int(samples[run, 1] * len(population_sizes))]),
            int(instance_ids[int(samples[run, 2] * len(instance_ids))]),
        )
        for run in range(runs)
    ]


def optimize(problem: Problem, population_size: int, evaluations: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generations = evaluations // population_size
    if generations < 1:
        raise ValueError(f"{evaluations} evaluations are not enough for one generation of {population_size} solutions")
    algorithm = NSGA3(
        pop_size=population_size,
        ref_dirs=UniformReferenceDirectionFactory(2, n_points=population_size).do(),
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    result = minimize(problem, algorithm, ("n_gen", generations), seed=seed, verbose=False)
    return result.F, result.X


def run_experiments(problem: Problem, model_name: str, args: argparse.Namespace) -> list[str]:
    """Runs all runs of the design and writes the resulting fronts to output/<model name>, returns the paths of the files."""
    output_directory = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_directory, exist_ok=True)

    paths = []
    for run_id, (evaluations, population_size, instance_id) in enumerate(get_design(args.runs, args.evaluations, args.population_sizes)):
        logging.info(f"Running instance {instance_id}, run {run_id + 1}, evals={evaluations}, pop={population_size}")

        objectives, variables = optimize(problem, population_size, evaluations, seed=DESIGN_SEED + run_id)

        results_data = {
            "model": model_name,
            "objectives": objectives.tolist(),
            "variables": variables.tolist(),
            "parameters": {
                "instance_id": instance_id,
                "run_id": run_id,
                "population_size": population_size,
                "n_evaluations": evaluations,
            },
        }
        path = os.path.join(output_directory, f"instance_{instance_id:02d}_run_{run_id:02d}_pop{population_size}.json")
        with open(path, "w") as f:
            json.dump(results_data, f, indent=4)

        logging.info(f"Data saved to {path}")
        paths.append(path)
    return paths
