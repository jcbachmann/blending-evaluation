import argparse
import logging
from multiprocessing import Pool
from multiprocessing.pool import Pool as PoolType

import numpy as np
from pymoo.core.problem import Problem

from bmh_ml.experiment import add_experiment_arguments, run_experiments
from bmh_ml.settings import BED_SIZE_X, BED_SIZE_Z, DEPOSITION_LENGTH, TOTAL_VOLUME, X_MAX, X_MIN
from bmh_ml.simulation import evaluate_sim
from bmh_ml.training_data import load_fixed_material_variables


def evaluate_deposition(material_variables: np.ndarray, deposition_variables: np.ndarray) -> tuple[float, float]:
    return evaluate_sim(
        material_variables=material_variables,
        deposition_variables=deposition_variables,
        bed_size_x=BED_SIZE_X,
        bed_size_z=BED_SIZE_Z,
        total_volume=TOTAL_VOLUME,
    )


class HomogenizationProblemSimulation(Problem):
    """Optimizes the deposition variables for a fixed material with the objectives of the simulation, evaluated in parallel by the given pool."""

    def __init__(self, material_variables: np.ndarray, pool: PoolType):
        self.material_variables = material_variables
        self.pool = pool

        super().__init__(n_var=DEPOSITION_LENGTH, n_obj=2, n_constr=0, xl=np.full(DEPOSITION_LENGTH, X_MIN), xu=np.full(DEPOSITION_LENGTH, X_MAX))

    def _evaluate(self, x, out, *_args, **_kwargs):
        out["F"] = np.array(self.pool.starmap(evaluate_deposition, [(self.material_variables, x_i) for x_i in x]))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_experiment_arguments(parser)
    return parser.parse_args()


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    material_variables = load_fixed_material_variables(args.training_data)
    logging.info(f"Optimizing for the material of the first row of {args.training_data}")

    # Starting the worker processes takes longer than evaluating a generation, so one pool is used for all generations and runs
    with Pool() as pool:
        problem = HomogenizationProblemSimulation(material_variables=material_variables, pool=pool)
        run_experiments(problem, "simulation", args, material_variables)


if __name__ == "__main__":
    main(get_args())
