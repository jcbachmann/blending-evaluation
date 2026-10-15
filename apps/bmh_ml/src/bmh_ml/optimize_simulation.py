import argparse
import logging

import numpy as np
from pymoo.core.problem import Problem

from bmh_ml.experiment import add_experiment_arguments, run_experiments
from bmh_ml.parallel import get_cpu_count
from bmh_ml.settings import BED_SIZE_X, BED_SIZE_Z, DEPOSITION_LENGTH, TOTAL_VOLUME, X_MAX, X_MIN
from bmh_ml.simulation import evaluate_sims_with_profiles
from bmh_ml.training_data import load_fixed_material_variables


class HomogenizationProblemSimulation(Problem):
    """Optimizes the deposition variables for a fixed material with the objectives of the simulation. A generation goes to the simulator in
    one call, `threads` simulations at the same time (0 for one per core): no worker processes, nothing to pickle."""

    def __init__(self, material_variables: np.ndarray, threads: int = 0):
        self.material_variables = np.asarray(material_variables, dtype=float)
        self.threads = threads

        super().__init__(n_var=DEPOSITION_LENGTH, n_obj=2, n_constr=0, xl=np.full(DEPOSITION_LENGTH, X_MIN), xu=np.full(DEPOSITION_LENGTH, X_MAX))

    def _evaluate(self, x, out, *_args, **_kwargs):
        materials = np.broadcast_to(self.material_variables, (len(x), len(self.material_variables)))
        out["F"] = evaluate_sims_with_profiles(materials, x, BED_SIZE_X, BED_SIZE_Z, TOTAL_VOLUME, threads=self.threads)[0]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_experiment_arguments(parser)
    return parser.parse_args()


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    material_variables = load_fixed_material_variables(args.training_data)
    logging.info(f"Optimizing for the material of the first row of {args.training_data}")

    problem = HomogenizationProblemSimulation(material_variables=material_variables, threads=get_cpu_count())
    run_experiments(problem, "simulation", args, material_variables)


if __name__ == "__main__":
    main(get_args())
