import argparse
import json
import logging
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from pyDOE3 import lhs
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.reference_direction import UniformReferenceDirectionFactory

from bmh_ml.simulation import evaluate_sim

# Simulation Configuration
MATERIAL_LENGTH: int = 50  # Length of material variables array
DEPOSITION_LENGTH: int = 20  # Length of deposition variables array
BED_SIZE_X: int = 59  # Bed size in X dimension
BED_SIZE_Z: int = 20  # Bed size in Z dimension
TOTAL_VOLUME: int = 2500  # Total volume for simulation


def evaluate_simulation(
    material_variables: np.ndarray,
    deposition_variables: np.ndarray,
    requires_scaling: bool,
) -> tuple[float, float]:
    if requires_scaling:
        # Convert deposition_variables from range 0-1 to blending bed positions
        x_min = BED_SIZE_Z * 0.5
        x_max = BED_SIZE_X - x_min
        scaled_deposition_variables = np.array(deposition_variables) * (x_max - x_min) + x_min
    else:
        scaled_deposition_variables = deposition_variables

    return evaluate_sim(
        material_variables=material_variables,
        deposition_variables=scaled_deposition_variables,
        bed_size_x=BED_SIZE_X,
        bed_size_z=BED_SIZE_Z,
        total_volume=TOTAL_VOLUME,
    )


class HomogenizationProblemSimulation(Problem):
    def __init__(self, material_variables):
        self.material_variables = material_variables

        x_min = BED_SIZE_Z * 0.5
        x_max = BED_SIZE_X - x_min

        n_var = DEPOSITION_LENGTH
        xl = np.full(DEPOSITION_LENGTH, x_min)
        xu = np.full(DEPOSITION_LENGTH, x_max)

        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *_args, **_kwargs):
        n_samples = x.shape[0]
        out["F"] = np.zeros((n_samples, 2))

        eval_params = [(self.material_variables, x_i, False) for x_i in x]

        with Pool() as pool:
            results = pool.starmap(evaluate_simulation, eval_params)

        out["F"] = np.array(results)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    logging.info("Loading training data")
    training_data = pd.read_csv("data/training_data.csv")
    logging.info("Training data loaded")

    example_row = training_data.iloc[0]
    material_variables = example_row.iloc[2 : 2 + MATERIAL_LENGTH].to_numpy()

    problem = HomogenizationProblemSimulation(material_variables=material_variables)

    os.makedirs("output/simulation", exist_ok=True)

    lhs_samples = lhs(n=3, samples=30, seed=42)
    function_evals = np.array([20000, 100000])
    pop_sizes = np.array([50, 100, 200])
    instance_ids = np.arange(30)

    lhs_combinations = []
    for i in range(30):
        feval_idx = int(lhs_samples[i, 0] * len(function_evals))
        pop_idx = int(lhs_samples[i, 1] * len(pop_sizes))
        inst_idx = int(lhs_samples[i, 2] * len(instance_ids))
        lhs_combinations.append((function_evals[feval_idx], pop_sizes[pop_idx], inst_idx))

    for run_id, (n_eval, pop_size, instance_id) in enumerate(lhs_combinations):
        logging.info(f"Running instance {instance_id}, run {run_id + 1}, evals={n_eval}, pop={pop_size}")

        ref_dirs = UniformReferenceDirectionFactory(2, n_points=pop_size).do()

        algorithm = NSGA3(
            pop_size=pop_size,
            ref_dirs=ref_dirs,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )

        n_gen = n_eval // pop_size

        res = minimize(
            problem,
            algorithm,
            ("n_gen", n_gen),
            seed=42 + run_id,
            verbose=False,
        )

        results_data = {
            "model": "simulation",
            "objectives": res.F.tolist(),
            "variables": res.X.tolist(),
            "parameters": {
                "instance_id": instance_id,
                "run_id": run_id,
                "population_size": int(pop_size),
                "n_evaluations": int(n_eval),
            },
        }
        output_path = f"output/simulation/instance_{instance_id:02d}_run_{run_id:02d}_pop{pop_size}.json"
        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=4)

        logging.info(f"Data saved to {output_path}")


if __name__ == "__main__":
    main(get_args())
