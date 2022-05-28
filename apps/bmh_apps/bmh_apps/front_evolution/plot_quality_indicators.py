import argparse
import os

import pandas as pd
import plotly.graph_objects as go
from _plotly_utils.colors import sample_colorscale


def read_quality_indicators(file: str) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df


def main(args: argparse.Namespace):
    fig = go.Figure()
    for directory in args.directories:
        quality_indicators_df = read_quality_indicators(os.path.join(directory, 'quality_indicators.csv'))
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
                    [float(variables['optimization.variable_count']) / 160],
                )[0],
            )
        ))

    fig.update_layout(dict(
        showlegend=False
    ))
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
