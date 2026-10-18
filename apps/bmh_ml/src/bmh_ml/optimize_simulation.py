import argparse
import logging

from bmh_ml.experiment import add_experiment_arguments, run_experiments
from bmh_ml.parallel import get_cpu_count
from bmh_ml.problems import SimulatorProblem
from bmh_ml.training_data import load_fixed_material_variables


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_experiment_arguments(parser)
    return parser.parse_args()


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    material_variables = load_fixed_material_variables(args.training_data)
    logging.info(f"Optimizing for the material of the first row of {args.training_data}")

    problem = SimulatorProblem(material_variables, threads=get_cpu_count())
    run_experiments(problem, "simulation", args, material_variables)


if __name__ == "__main__":
    main(get_args())
