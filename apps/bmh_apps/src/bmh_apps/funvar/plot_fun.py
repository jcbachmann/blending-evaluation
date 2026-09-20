import argparse
import logging
import os
import sys
import urllib.parse
import uuid

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline
import pyperclip

from bmh_apps.funvar.fun_var_math import filter_efficient_front
from bmh_apps.funvar.fun_var_results import FunVarResults

from .objective_resolver import prettify_objective


def export_fig(graph_type: str, fig: go.Figure, results: FunVarResults, label: str):
    output_directory = os.path.commonpath(results.df["file_path"].to_list())
    output_label = f"plot fun {graph_type} {label} {str(uuid.uuid4())[:4]}"
    html_filename = f"{output_label}.html"
    html_filepath = os.path.join(output_directory, html_filename)
    txt_filepath = os.path.join(output_directory, f"{output_label}.txt")
    logging.info(f"Writing plot to {html_filepath}")
    fig.write_html(html_filepath)
    fig.show()

    txt_content = "##### Graph\n"
    txt_content += plotly.offline.plot(fig, include_plotlyjs=False, output_type="div")
    txt_content += "\n\n"
    txt_content += f"Command: `{os.path.basename(sys.argv[0])} {' '.join(sys.argv[1:])}`"
    txt_content += "\n\n"
    txt_content += f"Local file: [{html_filename}](file://{urllib.parse.quote(html_filepath)})"

    with open(f"{txt_filepath}", "w") as f:
        f.write(txt_content)

    try:
        pyperclip.copy(txt_content)
        logging.info("Graph copied to clipboard")
    except pyperclip.PyperclipException as e:
        logging.warning(f"Could not copy graph to clipboard: {e}")


def plot_fun_1d(results: FunVarResults, label: str):
    fig = px.box(
        results.df,
        y=results.fun_columns[0],
    )

    export_fig("1d", fig, results, label)


def plot_fun_2d(results: FunVarResults, df_non_dominated: pd.DataFrame | None, label: str, auto_scale: bool = False):
    fig = px.scatter(
        results.df,
        x=results.fun_columns[0],
        y=results.fun_columns[1],
        color="parameters",
        symbol="run",
        title=label,
    )

    if not auto_scale:
        fig.update_xaxes(range=[0, 1.5], constrain="domain")
        fig.update_yaxes(range=[0, 1.5], scaleanchor="x", scaleratio=1, constrain="domain")
    fig.update_layout(
        xaxis_title=prettify_objective(results.fun_columns[0]),
        yaxis_title=prettify_objective(results.fun_columns[1]),
    )
    fig.add_trace(
        go.Scatter(
            x=[1],
            y=[1],
            mode="markers",
            marker={
                "size": 20,
                "color": "rgba(0, 0, 0, 1)",
            },
            name="Reference Point",
        )
    )
    if df_non_dominated is not None:
        fig.add_trace(
            go.Scatter(
                x=df_non_dominated[results.fun_columns[0]],
                y=df_non_dominated[results.fun_columns[1]],
                mode="markers",
                marker={"size": 10, "color": "rgba(0, 0, 0, 0)", "line": {"width": 1.5, "color": "rgba(0, 0, 0, 1)"}},
                name="Non-dominated",
            )
        )

    export_fig("2d", fig, results, label)


def plot_fun_3d(results: FunVarResults, df_non_dominated: pd.DataFrame | None, label: str, auto_scale: bool = False):
    fig = px.scatter_3d(
        results.df,
        x=results.fun_columns[0],
        y=results.fun_columns[1],
        z=results.fun_columns[2],
        color="parameters",
        symbol="run",
        title=label,
    )
    fig.update_traces(marker_size=5)
    if not auto_scale:
        fig.update_layout(
            scene={
                "xaxis": {"range": [0, 1.5]},
                "yaxis": {"range": [0, 1.5]},
                "zaxis": {"range": [0, 1.5]},
            },
        )
    fig.update_layout(
        scene={
            "xaxis_title": prettify_objective(results.fun_columns[0]),
            "yaxis_title": prettify_objective(results.fun_columns[1]),
            "zaxis_title": prettify_objective(results.fun_columns[2]),
        }
    )
    fig.add_trace(
        go.Scatter3d(
            x=[1],
            y=[1],
            z=[1],
            mode="markers",
            marker={
                "size": 20,
                "color": "rgba(0, 0, 0, 1)",
            },
            name="Reference Point",
        )
    )

    if not auto_scale:
        fig.update_layout(scene_aspectmode="cube")
    if df_non_dominated is not None:
        fig.add_trace(
            go.Scatter3d(
                x=df_non_dominated[results.fun_columns[0]],
                y=df_non_dominated[results.fun_columns[1]],
                z=df_non_dominated[results.fun_columns[2]],
                mode="markers",
                marker={"size": 10, "color": "rgba(0, 0, 0, 0)", "line": {"width": 1.5, "color": "rgba(0, 0, 0, 1)"}},
                name="Non-dominated",
            )
        )

    export_fig("3d", fig, results, label)


def plot_fun_4d(results: FunVarResults, label: str):
    fig = px.scatter_3d(
        results.df,
        x=results.fun_columns[0],
        y=results.fun_columns[1],
        z=results.fun_columns[2],
        color=results.fun_columns[3],
    )
    export_fig("4d", fig, results, label)


def plot_fun_5d(results: FunVarResults, label: str):
    fig = px.scatter_3d(
        results.df,
        x=results.fun_columns[0],
        y=results.fun_columns[1],
        z=results.fun_columns[2],
        size=results.fun_columns[3],
        color=results.fun_columns[4],
    )
    fig.update_layout(
        legend={
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        }
    )
    export_fig("5d trade-offs-1", fig, results, label)

    fig = px.scatter_3d(
        results.df,
        x=results.fun_columns[2],
        y=results.fun_columns[3],
        z=results.fun_columns[4],
        size=results.fun_columns[0],
        color=results.fun_columns[1],
    )
    fig.update_layout(
        legend={
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        }
    )
    export_fig("5d trade-offs-2", fig, results, label)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, nargs="+")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose logging")
    parser.add_argument("--non-dominated", action="store_true", default=False, help="Show only non-dominated solutions")
    parser.add_argument("--drop-columns", type=str, nargs="*", default=False, help="Columns/objectives to be dropped")
    parser.add_argument("--auto-scale", action="store_true", default=False, help="Automatically scale axis ranges")
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
        df_non_dominated = None
    else:
        df_non_dominated = filter_efficient_front(results.df, results.fun_columns)

    if len(additional_labels) > 0:
        label += " (" + ", ".join(additional_labels) + ")"

    if len(results.fun_columns) == 1:
        plot_fun_1d(results, label)
    elif len(results.fun_columns) == 2:
        plot_fun_2d(results, df_non_dominated, label, auto_scale=args.auto_scale)
    elif len(results.fun_columns) == 3:
        plot_fun_3d(results, df_non_dominated, label, auto_scale=args.auto_scale)
    elif len(results.fun_columns) == 4:
        plot_fun_4d(results, label)
    elif len(results.fun_columns) == 5:
        plot_fun_5d(results, label)
    else:
        raise Exception("Invalid number of columns")


if __name__ == "__main__":
    main()
