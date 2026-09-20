import argparse
import json
import logging
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px


def plot_optimization_results(file_paths: list[str]) -> None:
    """Load and plot optimization results from JSON files."""
    all_data = []

    for file_path in file_paths:
        with open(file_path) as f:
            data = json.load(f)
            objectives = np.array(data["objectives"])
            run_df = pd.DataFrame(objectives, columns=["F1", "F2"])
            run_df["Population"] = data["parameters"]["population_size"]
            run_df["Evaluations"] = data["parameters"]["n_evaluations"]
            run_df["File"] = Path(file_path).name
            all_data.append(run_df)

    combined_df = pd.concat(all_data, ignore_index=True)

    # Sort unique values for facets
    populations = sorted(combined_df["Population"].unique())
    evaluations = sorted(combined_df["Evaluations"].unique())

    fig = px.scatter(
        combined_df,
        x="F1",
        y="F2",
        color="File",
        facet_col="Population",
        facet_row="Evaluations",
        title="Optimization Results",
        category_orders={"Population": populations, "Evaluations": evaluations},
    )
    fig.update_layout(
        title_x=0.5,
        showlegend=True,
    )
    fig.show()


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # Expand file patterns
    files = []
    for pattern in args.files:
        # Skip the cache files written by transform_optimization_results, they are not optimization results
        files.extend(file for file in glob(pattern) if "_recomputed_" not in Path(file).name)

    if not files:
        logging.error("No files found matching the provided patterns")
        return

    plot_optimization_results(files)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument("files", type=str, nargs="+", help="paths to JSON files containing optimization results (glob patterns supported)")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
