import argparse
import logging

from bmh_ml.datasets.manifest import SCOPE_GENERAL_MATERIAL
from bmh_ml.experiment import add_experiment_arguments, run_experiments
from bmh_ml.problems import ModelProblem
from bmh_ml.settings import add_model_set_argument
from bmh_ml.surrogate import Surrogate
from bmh_ml.training_data import load_fixed_material_variables


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_experiment_arguments(parser)
    add_model_set_argument(parser)
    return parser.parse_args()


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    material_variables = load_fixed_material_variables(args.training_data)
    logging.info(f"Optimizing with model set '{args.model_set}' for the material of the first row of {args.training_data}")
    # the surrogate reads the material followed by the deposition, like the models of scope S2
    problem = ModelProblem(Surrogate.load(args.model_set), SCOPE_GENERAL_MATERIAL, material_variables)
    run_experiments(problem, "lstm_model", args, material_variables, extra_parameters={"model_set": args.model_set})


if __name__ == "__main__":
    main(get_args())
