import argparse
import html
import logging
from datetime import datetime

import pandas as pd

from bmh_ml.tracking.runs import EXPERIMENT_NAMES, configure_mlflow
from bmh_ml.tracking.store import get_reports_directory

SETS = ("val", "valop", "T1", "T2", "T2s", "T3", "T5")
OBJECTIVES = ("F1", "F2")
PARAMETER_COLUMNS = ("params.model", "params.bundle", "params.seed")
TRANSFER_COLUMNS = {"metrics.transfer/hv_ratio": "transfer hv", "metrics.transfer/hv_ratio_run_mean": "transfer hv/run"}
COST_COLUMNS = {"metrics.train/seconds": "train s", "metrics.throughput/batch_10000": "pred/s (10k)"}


def get_metric_columns(runs: pd.DataFrame, metric: str) -> list[str]:
    """The columns `<set>/<objective>/<metric>` that exist, in the order of the sets."""
    names = [f"{name}/{objective}/{metric}" for name in SETS for objective in OBJECTIVES]
    return [name for name in names if f"metrics.{name}" in runs.columns]


def build_leaderboard(runs: pd.DataFrame, metric: str = "nrmse", sort_by: str | None = None, top: int | None = None) -> pd.DataFrame:
    """A table with a row for each run, sorted by the best value of `sort_by` (lower is better for the errors, higher for r2, correlations and
    the transfer test). Runs without any of the metrics, like the parent runs of sweeps, are left out.

    `runs` is a table of runs like MLflow returns it: columns `params.*`, `metrics.*` and `tags.mlflow.runName`.
    """
    columns = get_metric_columns(runs, metric)
    if not columns:
        raise ValueError(f"No run has the metric '{metric}' ({', '.join(f'{s}/F1/{metric}' for s in SETS[:2])} ...)")
    transfer_columns = [title for column, title in TRANSFER_COLUMNS.items() if column in runs.columns]
    default_sort = f"T2/F2/{metric}"
    sort_by = sort_by or (default_sort if default_sort in columns else columns[0])
    if sort_by not in columns + transfer_columns:
        raise ValueError(f"Cannot sort by '{sort_by}', choose one of {columns + transfer_columns}")
    runs = runs[runs[[f"metrics.{name}" for name in columns]].notna().any(axis=1)]

    table = pd.DataFrame(index=runs.index)
    table["run"] = runs.get("tags.mlflow.runName", pd.Series(index=runs.index, dtype=object)).fillna(runs["run_id"].str[:8])
    for column in PARAMETER_COLUMNS:
        table[column.removeprefix("params.")] = runs.get(column)
    for name in columns:
        table[name] = runs[f"metrics.{name}"]
    for column, title in {**TRANSFER_COLUMNS, **COST_COLUMNS}.items():
        if column in runs.columns:
            table[title] = runs[column]
    higher_is_better = metric in ("r2", "spearman", "pairwise_accuracy") or sort_by in transfer_columns
    table = table.sort_values(sort_by, ascending=not higher_is_better, na_position="last")
    return table.head(top) if top else table


def get_noise_ceilings(runs: pd.DataFrame) -> pd.DataFrame:
    """The best possible R2 of each set, it is the same for every run of the same bundle. Rows are the sets, columns the objectives."""
    ceilings = {}
    for name in SETS:
        for objective in OBJECTIVES:
            column = f"metrics.{name}/{objective}/r2_ceiling"
            if column in runs.columns and runs[column].notna().any():
                ceilings.setdefault(name, {})[objective] = float(runs[column].dropna().iloc[0])
    return pd.DataFrame(ceilings).T


def format_value(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def to_markdown(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(table.columns) + " |"
    separator = "|" + "|".join("---" for _ in table.columns) + "|"
    rows = ["| " + " | ".join(format_value(value) for value in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([header, separator, *rows])


HTML_STYLE = "body{font-family:sans-serif;margin:2em}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px;text-align:right}"


def to_html(table: pd.DataFrame, title: str, note: str) -> str:
    head = "".join(f"<th>{html.escape(str(column))}</th>" for column in table.columns)
    rows = ("".join(f"<td>{html.escape(format_value(value))}</td>" for value in row) for row in table.itertuples(index=False))
    body = "".join(f"<tr>{row}</tr>" for row in rows)
    return (
        f"<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title><style>{HTML_STYLE}</style>"
        f"<h1>{html.escape(title)}</h1><p>{html.escape(note)}</p><table><tr>{head}</tr>{body}</table>"
    )


def get_note(metric: str) -> str:
    if metric == "nrmse":
        return (
            "nrmse: error divided by the noise of a single simulation (lower is better). "
            "A perfect model reaches 1 / sqrt(repeats of the labels), 0.25 for 16 repeats."
        )
    return f"{metric}, sorted by the best value first."


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leaderboard of the runs of an experiment")
    parser.add_argument("--experiment", default=EXPERIMENT_NAMES["S1"], help=f"Experiment name, {' or '.join(EXPERIMENT_NAMES.values())}")
    parser.add_argument("--metric", default="nrmse", help="rmse, mae, bias, r2, nrmse, tail_rmse, tail_bias, spearman, pairwise_accuracy, negative_rate")
    parser.add_argument("--sort", help="Column to sort by, default T2/F2/<metric> if it exists, or 'transfer hv'")
    parser.add_argument("--top", type=int, help="Only the best runs")
    parser.add_argument("--bundle", help="Only runs trained on this bundle")
    parser.add_argument("--html", action="store_true", help="Also write an HTML file to the reports directory of the store")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = get_args(argv)
    logging.basicConfig(level=logging.INFO)
    mlflow = configure_mlflow()
    experiment = mlflow.get_experiment_by_name(args.experiment)
    if experiment is None:
        raise SystemExit(f"No experiment '{args.experiment}' yet, train a model first (python -m bmh_ml.train)")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string=f"params.bundle = '{args.bundle}'" if args.bundle else "")
    if runs.empty:
        raise SystemExit(f"No runs in experiment '{args.experiment}'" + (f" for bundle {args.bundle}" if args.bundle else ""))

    table = build_leaderboard(runs, args.metric, args.sort, args.top)
    ceilings = get_noise_ceilings(runs)
    print(f"# {args.experiment}, {len(runs)} runs\n")
    print(to_markdown(table))
    if not ceilings.empty:
        print("\nBest possible R2 (noise ceiling of the labels):\n")
        print(to_markdown(ceilings.reset_index(names="set")))
    print(f"\n{get_note(args.metric)}")

    if args.html:
        file = get_reports_directory() / f"leaderboard-{args.experiment}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.html"
        file.write_text(to_html(table, f"{args.experiment} leaderboard", get_note(args.metric)))
        print(f"Written {file}")


if __name__ == "__main__":
    main()
