import argparse
import logging

import numpy as np
from pymoo.core.problem import Problem

from bmh_ml.experiment import add_experiment_arguments, run_experiments
from bmh_ml.settings import DEPOSITION_LENGTH, MATERIAL_LENGTH, X_MAX, X_MIN
from bmh_ml.surrogate import Surrogate
from bmh_ml.training_data import load_fixed_material_variables


class HomogenizationProblemMl(Problem):
    """Optimizes the deposition variables for a fixed material with the objectives predicted by the surrogate."""

    def __init__(self, surrogate: Surrogate, material_variables: np.ndarray):
        self.surrogate = surrogate
        self.material_variables = material_variables

        # The material variables are part of the model input, but fixed by equal bounds
        xl = np.concatenate((material_variables, np.full(DEPOSITION_LENGTH, X_MIN)))
        xu = np.concatenate((material_variables, np.full(DEPOSITION_LENGTH, X_MAX)))

        super().__init__(n_var=MATERIAL_LENGTH + DEPOSITION_LENGTH, n_obj=2, n_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *_args, **_kwargs):
        out["F"] = self.surrogate.predict(x)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_experiment_arguments(parser)
    return parser.parse_args()


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    problem = HomogenizationProblemMl(surrogate=Surrogate.load(), material_variables=load_fixed_material_variables())
    run_experiments(problem, "lstm_model", args)


if __name__ == "__main__":
    main(get_args())
