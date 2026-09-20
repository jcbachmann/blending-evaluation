"""Compare groups of independent optimization runs with empirical attainment functions (EAFs).

Runs are grouped by their varying parameters (e.g. `precondition=true` vs. `precondition=false`), every FUN file is one run.
Two plots are exported for two objectives:

* Attainment surfaces of each group. The best surface (dotted) bounds the region attained by at least one run, the
  median surface (solid) the region attained by half of the runs and the worst surface (dashed) the region attained by all runs.
* The EAF difference of two groups. Colored regions are attained by a different share of the runs of both groups,
  the color tells which group attains the region more often and the intensity by how much.

Objectives are minimized. Only two objectives are supported, use --drop-columns to select two of several objectives.
"""

import argparse
import logging

import moocore
import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go

from bmh_apps.funvar.figure_export import export_fig
from bmh_apps.funvar.fun_var_results import FunVarResults
from bmh_apps.funvar.objective_resolver import prettify_objective

# Categorical colors in their fixed, validated order. The color of a group is its position, never cycled.
CATEGORICAL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NEUTRAL_COLOR = "#f0efec"
SURFACE_COLOR = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"

ALL_RUNS = "all runs"
DEFAULT_PARAMETERS = "defaults"  # runs without varying parameters next to runs with, e.g. a random start population that is the config default

# EAF percentile -> (name, line dash, line width)
PERCENTILES = {0: ("best", "dot", 1.5), 50: ("median", "solid", 2.5), 100: ("worst", "dash", 1.5)}

DEFAULT_RANGE = (0.0, 1.5)


def get_groups(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Groups the runs by their varying parameters."""
    empty_name = ALL_RUNS if (df["parameters"] == "").all() else DEFAULT_PARAMETERS
    names = df["parameters"].where(df["parameters"] != "", empty_name)
    return dict(iter(df.groupby(names, sort=True)))


def select_groups(groups: dict[str, pd.DataFrame], compare: list[str] | None) -> dict[str, pd.DataFrame]:
    if compare is not None:
        unknown = [name for name in compare if name not in groups]
        if unknown or compare[0] == compare[1]:
            raise ValueError(f"--compare needs two different groups, unknown: {unknown or 'none'}, available: {list(groups)}")
        return {name: groups[name] for name in compare}
    if len(groups) > len(CATEGORICAL_COLORS):
        raise ValueError(f"{len(groups)} groups cannot be distinguished by color (maximum {len(CATEGORICAL_COLORS)}), use --compare: {list(groups)}")
    return groups


def get_points_and_sets(group: pd.DataFrame, fun_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Objective values of all runs of a group together with the number of the run (set) of each point."""
    return group[fun_columns].to_numpy(dtype=float), pd.factorize(group["file_path"], sort=True)[0] + 1


def compute_attainment_surfaces(points: np.ndarray, sets: np.ndarray) -> dict[int, np.ndarray]:
    """Attainment surfaces by percentile, each sorted by the first objective."""
    eaf = moocore.eaf(points, sets=sets, percentiles=list(PERCENTILES))
    surfaces = {}
    for percentile in PERCENTILES:
        surface = eaf[eaf[:, -1] == percentile][:, :2]
        surfaces[percentile] = surface[np.lexsort((-surface[:, 1], surface[:, 0]))]
    return surfaces


def keep_first_runs(points: np.ndarray, sets: np.ndarray, runs: int) -> tuple[np.ndarray, np.ndarray]:
    keep = sets <= runs
    return points[keep], sets[keep]


def equalize_run_counts(points_a: np.ndarray, sets_a: np.ndarray, points_b: np.ndarray, sets_b: np.ndarray) -> tuple[np.ndarray, ...]:
    """The difference of attainment counts is only comparable for the same number of runs, so keep the first runs (by path) of both groups."""
    runs = int(min(sets_a.max(), sets_b.max()))
    return (*keep_first_runs(points_a, sets_a, runs), *keep_first_runs(points_b, sets_b, runs))


def to_levels(differences: np.ndarray, runs: int, intervals: int) -> np.ndarray:
    """Level k is a difference of more than (k - 1) / intervals and at most k / intervals of the runs, the sign tells which group attains more."""
    counts = np.rint(differences).astype(int)
    return np.sign(counts) * ((np.abs(counts) * intervals + runs - 1) // runs)


def compute_eaf_difference(points_a: np.ndarray, sets_a: np.ndarray, points_b: np.ndarray, sets_b: np.ndarray, intervals: int) -> np.ndarray:
    """Rectangles (xmin, ymin, xmax, ymax, level), a positive level means that group a attains the rectangle more often than group b."""
    runs = int(sets_a.max())
    if int(sets_b.max()) != runs:
        raise ValueError(f"The EAF difference needs the same number of runs in both groups, got {runs} and {int(sets_b.max())}")
    # The last column is the number of runs of a minus the number of runs of b attaining the rectangle
    rectangles = moocore.eafdiff(np.column_stack([points_a, sets_a]), np.column_stack([points_b, sets_b]), rectangles=True)
    rectangles[:, 4] = to_levels(rectangles[:, 4], runs, intervals)
    return rectangles[rectangles[:, 4] != 0]


def get_axis_ranges(points: np.ndarray, auto_scale: bool) -> tuple[tuple[float, float], tuple[float, float]]:
    if not auto_scale:
        return DEFAULT_RANGE, DEFAULT_RANGE
    lower, upper = points.min(axis=0), points.max(axis=0)
    padding = np.where(upper > lower, 0.05 * (upper - lower), 0.05)
    return (lower[0] - padding[0], upper[0] + padding[0]), (lower[1] - padding[1], upper[1] + padding[1])


def clip_rectangles(rectangles: np.ndarray, ranges: tuple[tuple[float, float], tuple[float, float]]) -> np.ndarray:
    """Clip the unbounded (infinite) rectangles to the visible range and drop those outside of it."""
    (x_min, x_max), (y_min, y_max) = ranges
    clipped = rectangles.copy()
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], x_min, x_max)
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], y_min, y_max)
    return clipped[(clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])]


def mix_colors(low: str, high: str, amount: float) -> str:
    mixed = [
        round(low_channel + (high_channel - low_channel) * amount) for low_channel, high_channel in zip(pc.hex_to_rgb(low), pc.hex_to_rgb(high), strict=True)
    ]
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def get_difference_colors(intervals: int, color_a: str, color_b: str) -> dict[int, str]:
    """Diverging colors, positive levels (group a attains more) blend from the neutral midpoint to color a, negative ones to color b."""
    colors = {0: NEUTRAL_COLOR}
    for level in range(1, intervals + 1):
        amount = 0.2 + 0.8 * (level - 1) / max(intervals - 1, 1)
        colors[level] = mix_colors(NEUTRAL_COLOR, color_a, amount)
        colors[-level] = mix_colors(NEUTRAL_COLOR, color_b, amount)
    return colors


def get_rectangle_outlines(rectangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed outlines of all rectangles in one line, separated by gaps."""
    x_min, y_min, x_max, y_max = rectangles[:, 0], rectangles[:, 1], rectangles[:, 2], rectangles[:, 3]
    gap = np.full(len(rectangles), np.nan)
    return (
        np.column_stack([x_min, x_max, x_max, x_min, x_min, gap]).ravel(),
        np.column_stack([y_min, y_min, y_max, y_max, y_min, gap]).ravel(),
    )


def apply_chart_style(
    fig: go.Figure, *, title: str, label: str, x_title: str, y_title: str, ranges: tuple[tuple[float, float], tuple[float, float]], auto_scale: bool
):
    fig.update_layout(
        template="plotly_white",
        title={"text": f"{title}<br><sup>{label}</sup>" if label else title, "font": {"color": PRIMARY_INK}},
        paper_bgcolor=SURFACE_COLOR,
        plot_bgcolor=SURFACE_COLOR,
        font={"color": SECONDARY_INK},
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    axis_style = {"gridcolor": GRID_COLOR, "linecolor": AXIS_COLOR, "zerolinecolor": AXIS_COLOR, "tickfont": {"color": MUTED_INK}}
    fig.update_xaxes(range=ranges[0], constrain="domain", **axis_style)
    fig.update_yaxes(range=ranges[1], **({} if auto_scale else {"scaleanchor": "x", "scaleratio": 1}), **axis_style)


def add_reference_point(fig: go.Figure, ranges: tuple[tuple[float, float], tuple[float, float]]):
    # Objectives are relative to the reference solution, so the reference point is at (1, 1)
    if ranges[0][0] <= 1 <= ranges[0][1] and ranges[1][0] <= 1 <= ranges[1][1]:
        fig.add_trace(go.Scatter(x=[1], y=[1], mode="markers", marker={"size": 12, "color": PRIMARY_INK}, name="Reference Point"))


def plot_attainment_surfaces(
    surfaces: dict[str, dict[int, np.ndarray]],
    *,
    ranges: tuple[tuple[float, float], tuple[float, float]],
    x_title: str,
    y_title: str,
    label: str,
    auto_scale: bool,
) -> go.Figure:
    fig = go.Figure()
    for color, (group, group_surfaces) in zip(CATEGORICAL_COLORS, surfaces.items(), strict=False):
        for percentile, (percentile_name, dash, width) in PERCENTILES.items():
            surface = group_surfaces[percentile]
            fig.add_trace(
                go.Scatter(
                    x=surface[:, 0],
                    y=surface[:, 1],
                    mode="lines",
                    line_shape="hv",
                    line={"color": color, "width": width, "dash": dash},
                    name=group,
                    legendgroup=group,
                    showlegend=percentile == 50,
                    hovertemplate=f"{group}, {percentile_name} surface<br>%{{x:.3f}} / %{{y:.3f}}<extra></extra>",
                )
            )
    add_reference_point(fig, ranges)
    apply_chart_style(fig, title="Attainment surfaces", label=label, x_title=x_title, y_title=y_title, ranges=ranges, auto_scale=auto_scale)
    fig.update_layout(margin={"b": 110})
    fig.add_annotation(
        text="Dotted: best (attained by at least one run), solid: median, dashed: worst (attained by all runs)",
        xref="paper",
        yref="paper",
        x=0,
        y=-0.16,
        showarrow=False,
        font={"size": 11, "color": MUTED_INK},
    )
    return fig


def plot_eaf_difference(
    rectangles: np.ndarray,
    *,
    group_a: str,
    group_b: str,
    intervals: int,
    ranges: tuple[tuple[float, float], tuple[float, float]],
    x_title: str,
    y_title: str,
    label: str,
    auto_scale: bool,
) -> go.Figure:
    color_a, color_b = CATEGORICAL_COLORS[0], CATEGORICAL_COLORS[1]
    colors = get_difference_colors(intervals, color_a, color_b)
    fig = go.Figure()

    for level in [*range(-intervals, 0), *range(1, intervals + 1)]:
        level_rectangles = rectangles[rectangles[:, 4] == level]
        if len(level_rectangles) == 0:
            continue
        x, y = get_rectangle_outlines(level_rectangles)
        winner = group_a if level > 0 else group_b
        low, high = (abs(level) - 1) / intervals, abs(level) / intervals
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line={"width": 0.6, "color": colors[level]},  # a hairline in the fill color closes anti-aliasing seams between neighbouring rectangles
                fill="toself",
                fillcolor=colors[level],
                hoveron="fills",
                hoverinfo="text",
                text=f"{winner} attains this region in {low:.0%} to {high:.0%} more of its runs",
                showlegend=False,
            )
        )

    # Legend entries for the two sides and a colorbar for the intensity
    for group, color in ((group_a, color_a), (group_b, color_b)):
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers", marker={"symbol": "square", "size": 12, "color": color}, name=f"{group} attains more", hoverinfo="skip"
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={
                "color": [0],
                "cmin": -intervals - 0.5,
                "cmax": intervals + 0.5,
                "colorscale": [
                    [(level + intervals + offset) / (2 * intervals + 1), colors[level]] for level in range(-intervals, intervals + 1) for offset in (0, 1)
                ],
                "showscale": True,
                "colorbar": {
                    "title": {"text": "Difference in share of<br>runs attaining (upper bound)"},
                    "tickvals": list(range(-intervals, intervals + 1)),
                    "ticktext": [f"{100 * level / intervals:+.0f}%" if level else "0%" for level in range(-intervals, intervals + 1)],
                },
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )
    add_reference_point(fig, ranges)
    fig.update_layout(legend={"orientation": "h", "yanchor": "top", "y": -0.14, "x": 0}, margin={"b": 110})
    apply_chart_style(
        fig, title=f"EAF difference: {group_a} vs. {group_b}", label=label, x_title=x_title, y_title=y_title, ranges=ranges, auto_scale=auto_scale
    )
    return fig


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare groups of optimization runs with empirical attainment functions")
    parser.add_argument("filename", type=str, nargs="+")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose logging")
    parser.add_argument("--drop-columns", type=str, nargs="*", default=None, help="Columns/objectives to be dropped, exactly two objectives must remain")
    parser.add_argument(
        "--compare",
        type=str,
        nargs=2,
        metavar=("A", "B"),
        default=None,
        help="The two groups (parameter combinations) to compare, required for more than two groups",
    )
    parser.add_argument("--intervals", type=int, default=5, help="Number of intervals the difference of the attainment is divided into")
    parser.add_argument("--auto-scale", action="store_true", default=False, help="Automatically scale axis ranges")
    return parser.parse_args()


def main(args: argparse.Namespace | None = None):
    if args is None:
        # Entry point of the plot_eaf script
        args = get_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    results = FunVarResults.from_files(args.filename, fun_only=True)
    label = results.label

    if args.drop_columns:
        results.drop_columns(args.drop_columns)
        label = f"{label} (drop [{', '.join([c.replace('/', '_') for c in args.drop_columns])}])".strip()

    if len(results.fun_columns) != 2:
        raise ValueError(f"EAF plots need exactly two objectives, got {len(results.fun_columns)}: {results.fun_columns}. Use --drop-columns to select two.")

    all_groups = get_groups(results.df)
    groups = select_groups(all_groups, args.compare)
    for name, group in groups.items():
        logging.info(f"Group '{name}': {group['file_path'].nunique()} runs, {len(group)} solutions")

    points_and_sets = {name: get_points_and_sets(group, results.fun_columns) for name, group in groups.items()}
    ranges = get_axis_ranges(np.vstack([points for points, _ in points_and_sets.values()]), args.auto_scale)
    x_title, y_title = (prettify_objective(column) for column in results.fun_columns)

    surfaces = {name: compute_attainment_surfaces(points, sets) for name, (points, sets) in points_and_sets.items()}
    fig = plot_attainment_surfaces(surfaces, ranges=ranges, x_title=x_title, y_title=y_title, label=label, auto_scale=args.auto_scale)
    export_fig("eaf", fig, results, label)

    if len(groups) != 2:
        logging.warning(f"The EAF difference needs exactly two groups, got {len(groups)}. Select two with --compare: {list(groups)}")
        return

    (group_a, (points_a, sets_a)), (group_b, (points_b, sets_b)) = points_and_sets.items()
    if sets_a.max() != sets_b.max():
        logging.warning(f"The EAF difference needs the same number of runs, using the first {min(sets_a.max(), sets_b.max())} runs (by path) of both groups")
    points_a, sets_a, points_b, sets_b = equalize_run_counts(points_a, sets_a, points_b, sets_b)
    rectangles = clip_rectangles(compute_eaf_difference(points_a, sets_a, points_b, sets_b, args.intervals), ranges)
    fig = plot_eaf_difference(
        rectangles,
        group_a=group_a,
        group_b=group_b,
        intervals=args.intervals,
        ranges=ranges,
        x_title=x_title,
        y_title=y_title,
        label=label,
        auto_scale=args.auto_scale,
    )
    export_fig("eaf-difference", fig, results, label)


if __name__ == "__main__":
    main()
