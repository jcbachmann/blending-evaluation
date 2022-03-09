import argparse
import logging
import time

import plotly.express as px

from bmh_apps.funvar import fun_var_math
from .fun_var_results import FunVarResults


def plot_fun_2d(results: FunVarResults):
    px.scatter(
        results.df,
        x=results.fun_columns[0],
        y=results.fun_columns[1],
        color='run'
    ).show()


def plot_fun_3d(results: FunVarResults):
    px.scatter_3d(
        results.df,
        x=results.fun_columns[0],
        y=results.fun_columns[1],
        z=results.fun_columns[2],
        color='run'
    ).show()


def plot_fun_4d(results: FunVarResults):
    px.scatter_3d(
        results.df,
        x=results.fun_columns[0],
        y=results.fun_columns[1],
        z=results.fun_columns[2],
        color=results.fun_columns[3],
    ).show()


def plot_fun_5d(results: FunVarResults):
    fig = px.scatter_3d(
        results.df,
        x=results.fun_columns[0],
        y=results.fun_columns[1],
        z=results.fun_columns[2],
        size=results.fun_columns[3],
        color=results.fun_columns[4],
    )
    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ))
    fig.write_html(f"trade-offs-1-{int(time.time())}.html")
    fig.show()

    fig = px.scatter_3d(
        results.df,
        x=results.fun_columns[2],
        y=results.fun_columns[3],
        z=results.fun_columns[4],
        size=results.fun_columns[0],
        color=results.fun_columns[1],
    )
    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ))
    fig.write_html(f"trade-offs-2-{int(time.time())}.html")
    fig.show()


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    results = FunVarResults.from_files(args.filename)

    if args.non_dominated:
        results.df = fun_var_math.filter_efficient_front(results.df, results.fun_columns)

    if len(results.fun_columns) == 2:
        plot_fun_2d(results)
    elif len(results.fun_columns) == 3:
        plot_fun_3d(results)
    elif len(results.fun_columns) == 4:
        plot_fun_4d(results)
    elif len(results.fun_columns) == 5:
        plot_fun_5d(results)
    else:
        raise Exception("Invalid number of columns")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, nargs='+')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose logging')
    parser.add_argument('--non-dominated', action='store_true', default=False, help='Show only non-dominated solutions')
    return parser.parse_args()


if __name__ == '__main__':
    main(get_args())
