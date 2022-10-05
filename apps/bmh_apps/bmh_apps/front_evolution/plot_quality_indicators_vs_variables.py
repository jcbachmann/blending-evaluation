import argparse
import logging
import os
from glob import glob
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from _plotly_utils.colors import sample_colorscale

from .plot_quality_indicators import read_quality_indicators


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO)

    final_hv_df = pd.DataFrame({'Variables': [], 'Run': [], 'HV': [], 'Generations': []})
    total_generations = 33318
    generation_samples = [
        1000,
        2500,
        5000,
        10000,
        20000,
        total_generations
    ]

    fig = go.Figure()
    for directory_arg in args.directories:
        for quality_indicators_file in glob(os.path.join(directory_arg, 'quality_indicators.csv')):
            directory = Path(quality_indicators_file).parent.name
            try:
                quality_indicators_df = read_quality_indicators(quality_indicators_file)
            except FileNotFoundError:
                logging.warning(f"No quality indicators found in {directory}")
                continue
            assignments = directory.split(',')
            variables = {
                key: value for key, value in [assignment.split('=') for assignment in assignments]
            }
            print(float(variables['optimization.variable_count']))
            fig.add_trace(go.Scattergl(
                x=quality_indicators_df.index,
                y=quality_indicators_df['HV'],
                mode='markers',
                name=variables['optimization.variable_count'],
                marker=dict(
                    color=sample_colorscale(
                        'Viridis',
                        [float(variables['optimization.variable_count']) / 295],
                    )[0],
                )
            ))

            if quality_indicators_df.shape[0] == total_generations:
                final_hv_df = pd.concat([final_hv_df, pd.DataFrame({
                    'Variables': [int(variables['optimization.variable_count']) for _ in generation_samples],
                    'Run': [int(variables['+run']) for _ in generation_samples],
                    'HV': [quality_indicators_df['HV'].values[g - 1] for g in generation_samples],
                    'Generations': generation_samples
                })])

    fig.update_layout(dict(
        showlegend=False
    ))
    fig.show()

    final_hv_df = final_hv_df.sort_values(by=['Variables', 'Generations'])
    fig = px.box(final_hv_df, x='Variables', y='HV', color='Generations')
    fig.show()
    fig = px.box(final_hv_df, x='Generations', y='HV', color='Variables')
    fig.show()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot quality indicators"
    )
    parser.add_argument(
        "directories",
        type=str,
        nargs="+",
        help="Paths result directories",
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Show interactive plot',
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
