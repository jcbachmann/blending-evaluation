import argparse
import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from pyDOE3 import lhs
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.reference_direction import UniformReferenceDirectionFactory

MATERIAL_LENGTH: int = 50  # Length of material variables array
DEPOSITION_LENGTH: int = 20  # Length of deposition variables array


class HomogenizationProblemMl(Problem):
    def __init__(self, scaler, model_f1, model_f2, material_variables):
        self.material_variables = material_variables
        self.scaler = scaler
        self.model_f1 = model_f1
        self.model_f2 = model_f2

        n_var = 70
        xl = np.concatenate((material_variables, np.full(20, 10)))
        xu = np.concatenate((material_variables, np.full(20, 49)))

        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *_args, **_kwargs):
        x_scaled = self.scaler.transform(x)
        x_lstm = x_scaled.reshape((x_scaled.shape[0], 1, x_scaled.shape[1]))
        f1 = self.model_f1.predict(x_lstm, verbose=0).flatten()
        f2 = self.model_f2.predict(x_lstm[:, :, MATERIAL_LENGTH:], verbose=0).flatten()  # F2 only depends on the deposition
        out["F"] = np.column_stack([f1, f2])


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

    lstm_model_f1 = tf.keras.models.load_model("data/lstm_model_f1_random_training_data.keras")
    lstm_model_f2 = tf.keras.models.load_model("data/lstm_model_f2_random_training_data.keras")

    with open("data/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)  # noqa: S301 - written by train_lstm_model.py

    problem = HomogenizationProblemMl(
        scaler=scaler,
        model_f1=lstm_model_f1,
        model_f2=lstm_model_f2,
        material_variables=material_variables,
    )

    os.makedirs("output/lstm_model", exist_ok=True)

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
            "model": "lstm_model",
            "objectives": res.F.tolist(),
            "variables": res.X.tolist(),
            "parameters": {
                "instance_id": instance_id,
                "run_id": run_id,
                "population_size": int(pop_size),
                "n_evaluations": int(n_eval),
            },
        }
        output_path = f"output/lstm_model/instance_{instance_id:02d}_run_{run_id:02d}_pop{pop_size}.json"
        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=4)

        logging.info(f"Data saved to {output_path}")


if __name__ == "__main__":
    main(get_args())
