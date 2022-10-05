import argparse
import logging
import os
import time
import uuid

import plotly.express as px
import plotly.graph_objects as go

from .fun_var_math import filter_efficient_front
from .fun_var_results import FunVarResults


def plot_fun_1d(results: FunVarResults):
    fig = px.box(
        results.df,
        y=results.fun_columns[0],
    )
    fig.write_html(f"plot-fun-1d-{int(time.time())}.html")
    fig.show()


def plot_fun_2d(results: FunVarResults, label: str):
    fig = px.scatter(
        results.df,
        x=results.fun_columns[0],
        y=results.fun_columns[1],
        color='run',
        title=label
    )

    fig.update_xaxes(
        range=[0, 1.5],
        constrain="domain"
    )
    fig.update_yaxes(
        range=[0, 1.5],
        scaleanchor="x",
        scaleratio=1,
        constrain="domain"
    )
    fig.add_trace(go.Scatter(
        x=[1],
        y=[1],
        mode='markers',
        marker=dict(
            size=20,
            color='rgba(0, 0, 0, 1)',
        ),
        name='Reference Point'
    ))
    fig.write_html(os.path.join(
        os.path.commonpath(results.df['file_path'].to_list()),
        f"plot fun 2d {label} {str(uuid.uuid4())[:4]}.html"
    ))
    fig.show()


def plot_fun_3d(results: FunVarResults, label: str):
    fig = px.scatter_3d(
        results.df,
        x=results.fun_columns[0],
        y=results.fun_columns[1],
        z=results.fun_columns[2],
        color='run',
        title=label
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, 1.5]),
            yaxis=dict(range=[0, 1.5]),
            zaxis=dict(range=[0, 1.5]),
        ),
    )
    fig.add_trace(go.Scatter3d(
        x=[1],
        y=[1],
        z=[1],
        mode='markers',
        marker=dict(
            size=20,
            color='rgba(0, 0, 0, 1)',
        ),
        name='Reference Point'
    ))
    fig.update_layout(scene_aspectmode='cube')
    fig.write_html(os.path.join(
        os.path.commonpath(results.df['file_path'].to_list()),
        f"plot fun 3d {label} {str(uuid.uuid4())[:4]}.html"
    ))
    fig.show()


def plot_fun_4d(results: FunVarResults):
    fig = px.scatter_3d(
        results.df,
        x=results.fun_columns[0],
        y=results.fun_columns[1],
        z=results.fun_columns[2],
        color=results.fun_columns[3],
    )
    fig.write_html(f"plot-fun-4d-{int(time.time())}.html")
    fig.show()


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


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, nargs='+')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose logging')
    parser.add_argument('--non-dominated', action='store_true', default=False, help='Show only non-dominated solutions')
    parser.add_argument('--drop-columns', type=str, nargs='*', default=False, help='Columns/objectives to be dropped')
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    results = FunVarResults.from_files(args.filename, fun_only=True)
    label = results.label

    additional_labels = []

    if args.drop_columns:
        results.drop_columns(args.drop_columns)
        additional_labels.append(f"drop [{', '.join([c.replace('/', '_') for c in args.drop_columns])}]")

    if args.non_dominated:
        results.df = filter_efficient_front(results.df, results.fun_columns)
        additional_labels.append("non-dominated")

    if len(additional_labels) > 0:
        label += " (" + ", ".join(additional_labels) + ")"

    if len(results.fun_columns) == 1:
        plot_fun_1d(results)
    elif len(results.fun_columns) == 2:
        plot_fun_2d(results, label)
    elif len(results.fun_columns) == 3:
        plot_fun_3d(results, label)
    elif len(results.fun_columns) == 4:
        plot_fun_4d(results)
    elif len(results.fun_columns) == 5:
        plot_fun_5d(results)
    else:
        raise Exception("Invalid number of columns")


if __name__ == '__main__':
    main()
