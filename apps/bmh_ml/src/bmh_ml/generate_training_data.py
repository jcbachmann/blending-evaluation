import argparse
import json
import logging
import os

import pandas as pd
from pqdm.processes import pqdm

from bmh_ml.settings import (
    BED_SIZE_X,
    BED_SIZE_Z,
    DEPOSITION_LENGTH,
    MATERIAL_LENGTH,
    MATERIAL_MAX,
    MATERIAL_MIN,
    TOTAL_VOLUME,
    TRAINING_DATA_FILE,
)
from bmh_ml.simulation import evaluate_sim
from bmh_ml.variables import generate_deposition_variables, generate_material_variables


def generate_row(
    material_variables: list[float],
    deposition_length: int,
    bed_size_x: float,
    bed_size_z: float,
    total_volume: float,
) -> pd.DataFrame:
    x_min = bed_size_z * 0.5
    x_max = bed_size_x - x_min

    deposition_variables = generate_deposition_variables(
        deposition_length=deposition_length,
        x_min=x_min,
        x_max=x_max,
    )

    f1, f2 = evaluate_sim(
        material_variables=material_variables,
        deposition_variables=deposition_variables,
        bed_size_x=bed_size_x,
        bed_size_z=bed_size_z,
        total_volume=total_volume,
    )

    return pd.DataFrame(
        {
            "f1": [f1],
            "f2": [f2],
            **{f"m{i + 1}": [m] for i, m in enumerate(material_variables)},
            **{f"d{i + 1}": [d] for i, d in enumerate(deposition_variables)},
        }
    )


def generate(args):
    material_variables = generate_material_variables(
        material_length=args.material_length,
        material_min=args.material_min,
        material_max=args.material_max,
    )

    return [
        generate_row(
            material_variables=material_variables,
            deposition_length=args.deposition_length,
            bed_size_x=args.bed_size_x,
            bed_size_z=args.bed_size_z,
            total_volume=args.total_volume,
        )
        for _ in range(args.depositions_per_material)
    ]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default=TRAINING_DATA_FILE)
    parser.add_argument("--n_jobs", type=int, default=16)
    parser.add_argument("--materials", type=int, default=5000)
    parser.add_argument("--depositions_per_material", type=int, default=50)
    parser.add_argument("--deposition_length", type=int, default=DEPOSITION_LENGTH)
    parser.add_argument("--material_length", type=int, default=MATERIAL_LENGTH)
    parser.add_argument("--bed_size_x", type=float, default=BED_SIZE_X)
    parser.add_argument("--bed_size_z", type=float, default=BED_SIZE_Z)
    parser.add_argument("--total_volume", type=float, default=TOTAL_VOLUME)
    parser.add_argument("--material_min", type=float, default=MATERIAL_MIN)
    parser.add_argument("--material_max", type=float, default=MATERIAL_MAX)
    return parser.parse_args()


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO)

    rows = args.materials * args.depositions_per_material
    logging.info(f"Generating {rows} rows")
    datasets = pqdm([args] * args.materials, generate, n_jobs=args.n_jobs)
    flattened_datasets = [item for sublist in datasets for item in sublist]
    logging.info("Building dataframe")
    combined_data = pd.concat(flattened_datasets, ignore_index=True)
    logging.info(f"Writing to {args.output_file}")
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    combined_data.to_csv(args.output_file, index=False)

    params_file = args.output_file.rsplit(".", 1)[0] + "_params.json"
    with open(params_file, "w") as f:
        json.dump(vars(args), f, indent=4)
    logging.info(f"Parameters saved to {params_file}")
    logging.info("Done")


if __name__ == "__main__":
    main(get_args())
