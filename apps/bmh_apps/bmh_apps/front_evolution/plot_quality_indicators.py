import argparse
import logging
import os
from glob import glob
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def read_quality_indicators(file: str) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO)

    fig = go.Figure()
    for directory_arg in args.directories:
        for quality_indicators_file in glob(os.path.join(directory_arg, "quality_indicators.csv")):
            directory = Path(quality_indicators_file).parent.name
            try:
                quality_indicators_df = read_quality_indicators(quality_indicators_file)
            except FileNotFoundError:
                logging.warning(f"No quality indicators found in {directory}")
                continue

            # Reduce amount of data points to 10%
            quality_indicators_df = quality_indicators_df.iloc[::10, :]

            assignments = directory.split(",")
            variables = {key: value for key, value in [assignment.split("=") for assignment in assignments]}
            fig.add_trace(
                go.Scattergl(
                    x=quality_indicators_df.index,
                    y=quality_indicators_df["HV"],
                    mode="markers",
                    name=f"Run {variables['+run']}",
                )
            )

    fig.update_layout(
        dict(
            showlegend=False,
            xaxis_range=[0, 333333],
            yaxis_range=[0, 0.6],
        )
    )
    fig.show()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot quality indicators")
    parser.add_argument(
        "directories",
        type=str,
        nargs="+",
        help="Paths result directories",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Show interactive plot",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
