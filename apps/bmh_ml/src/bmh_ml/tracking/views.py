"""Saved views of the MLflow UI for tracking the attempts: which columns, charts, filters and sort order to look at.

MLflow (3.16) stores a saved view as an experiment tag `mlflow.sharedViewState.<id>`: JSON with a name and the page state, compressed
with zlib and base64 encoded after `deflate;`. The views are defined here and written with fixed ids, so installing them again updates them
instead of adding copies. They appear in the "Views" menu of the runs page of the experiment.

    python -m bmh_ml.tracking.views        # install or update the views, prints their links
"""

import argparse
import base64
import json
import time
import zlib

from bmh_ml.tracking.runs import EXPERIMENT_NAMES, configure_mlflow

TAG_PREFIX = "mlflow.sharedViewState."
DEFAULT_URL = "http://127.0.0.1:5000"


def metric_column(name: str) -> str:
    return f"metrics.`{name}`"


def param_column(name: str) -> str:
    return f"params.`{name}`"


def line_chart(uuid: str, section: str, metric: str) -> dict:
    """A line chart of one metric over the steps, e.g. the rounds of the refinement loop."""
    return {
        "uuid": uuid,
        "type": "LINE",
        "runsCountToCompare": 20,
        "metricSectionId": section,
        "deleted": False,
        "isGenerated": False,
        "metricKey": metric,
        "selectedMetricKeys": [metric],
        "lineSmoothness": 0,
        "xAxisScaleType": "linear",
        "scaleType": "linear",
        "range": {},
        "xAxisKey": "step",
        "selectedXAxisMetricKey": "",
        "yAxisKey": "metric",
        "yAxisExpressions": [],
        "ignoreOutliers": False,
        "useGlobalXaxisKey": False,
        "useGlobalLineSmoothing": True,
        "selectedYAxisMetricKey": "metric",
    }


def bar_chart(uuid: str, section: str, metric: str) -> dict:
    """A bar per run of the latest value of one metric."""
    return {
        "uuid": uuid,
        "type": "BAR",
        "runsCountToCompare": 50,
        "metricSectionId": section,
        "deleted": False,
        "isGenerated": False,
        "displayName": metric,
        "metricKey": metric,
    }


def section(uuid: str, name: str) -> dict:
    return {"uuid": uuid, "name": name, "display": True, "isReordered": True, "deleted": False, "isGenerated": False}


def charts_state(sections: list[tuple[str, str, str, list[str]]]) -> dict:
    """`sections`: (section id, title, "LINE" or "BAR", metrics)."""
    charts, section_list = [], []
    for section_id, title, kind, metrics in sections:
        section_list.append(section(section_id, title))
        make = line_chart if kind == "LINE" else bar_chart
        charts += [make(f"{section_id}-{index}", section_id, metric) for index, metric in enumerate(metrics)]
    return {"compareRunCharts": charts, "compareRunSections": section_list, "isAccordionReordered": True}


def base_state(search_filter: str = "", order_by: str = "attributes.start_time", ascending: bool = False, columns: list[str] | None = None) -> dict:
    return {
        "searchFilter": search_filter,
        "orderByKey": order_by,
        "orderByAsc": ascending,
        "startTime": "ALL",
        "lifecycleFilter": "Active",
        "datasetsFilter": [],
        "modelVersionFilter": "All Runs",
        "selectedColumns": columns or [],
        "columnOrder": columns or [],  # the order of the list, otherwise the UI sorts the columns by name
        "columnWidths": {},
        "runsHiddenMode": "SHOW_ALL",
        "viewMaximized": False,
        "runListHidden": False,
        "isAccordionReordered": False,
        "useGroupedValuesInCharts": True,
        "hideEmptyCharts": True,
        "groupBy": None,
        "groupsExpanded": {},
        "globalLineChartConfig": {"xAxisKey": "step", "lineSmoothness": 0, "selectedXAxisMetricKey": ""},
    }


LEADERBOARD_COLUMNS = [
    "attributes.`Description`",
    *map(metric_column, ["transfer/hv_ratio", "transfer/hv_ratio_run_mean", "transfer/negative_rate", "transfer/F1/bias", "transfer/F2/bias"]),
    *map(metric_column, ["valop/F1/nrmse", "valop/F2/nrmse", "T2/F1/r2", "T2/F2/r2", "T1/F1/r2", "T1/F2/r2", "val/F1/nrmse", "val/F2/nrmse"]),
    metric_column("train/seconds"),
    *map(param_column, ["model", "bundle", "seed"]),
]
HAS_TRANSFER = "metrics.`transfer/hv_ratio` >= 0"


def get_views() -> dict[str, tuple[str, dict]]:
    """id: (name, state). The names start with a number so the menu lists them in the order they are meant to be used."""
    refine_sections = [
        (
            "transfer",
            "Transfer test: optimize the model, simulate what it finds (higher hv is better)",
            "LINE",
            ["transfer/hv_ratio", "transfer/hv_ratio_run_mean", "transfer/negative_rate", "transfer/F1/bias", "transfer/F2/bias"],
        ),
        (
            "front",
            "Error of the model on the solutions it found, before they were added (0 is right)",
            "LINE",
            ["refine/F1/front_bias", "refine/F2/front_bias", "refine/front_negative_rate"],
        ),
        ("accuracy", "Accuracy on the test sets (R2, 1 is perfect)", "LINE", ["T2/F1/r2", "T2/F2/r2", "T2s/F1/r2", "T2s/F2/r2", "T1/F1/r2", "T1/F2/r2"]),
        ("cost", "Cost of a round", "LINE", ["refine/rows_added", "refine/round_seconds"]),
    ]
    comparison_sections = [
        (
            "transfer",
            "Transfer test (higher hv is better, negative rate and bias should be 0)",
            "BAR",
            ["transfer/hv_ratio", "transfer/hv_ratio_run_mean", "transfer/negative_rate", "transfer/F2/bias", "transfer/F1/bias"],
        ),
        (
            "accuracy",
            "Accuracy in the operating region and on random inputs (nrmse, lower is better, 0.25 perfect)",
            "BAR",
            ["T2/F1/nrmse", "T2/F2/nrmse", "T1/F1/nrmse", "T1/F2/nrmse"],
        ),
    ]
    sweep_columns = [
        "attributes.`Description`",
        *map(metric_column, ["valop/F1/nrmse", "valop/F2/nrmse", "val/F1/nrmse", "val/F2/nrmse", "train/seconds"]),
        *map(param_column, ["model", "bundle", "learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree"]),
        *map(param_column, ["width", "depth", "batch_size", "activation", "dropout", "alpha"]),
    ]
    return {
        "bmhml1leaderboard": (
            "1 Leaderboard: transfer test and accuracy",
            base_state(HAS_TRANSFER, metric_column("transfer/hv_ratio"), False, LEADERBOARD_COLUMNS),
        ),
        "bmhml2comparison": (
            "2 Comparison charts: transfer test and accuracy (chart tab)",
            {**base_state(HAS_TRANSFER, metric_column("transfer/hv_ratio"), False, LEADERBOARD_COLUMNS), **charts_state(comparison_sections)},
        ),
        "bmhml3refinement": (
            "3 Refinement loops, round by round (chart tab)",
            {**base_state("attributes.run_name LIKE 'refine-%'", columns=["attributes.`Description`"]), **charts_state(refine_sections)},
        ),
        "bmhml4sweeps": ("4 Sweep trials: parameters and validation", base_state("tags.trial LIKE '%'", metric_column("valop/F2/nrmse"), True, sweep_columns)),
        "bmhml5all": ("5 All runs, newest first", base_state(columns=LEADERBOARD_COLUMNS)),
    }


def encode_state(state: dict) -> str:
    return "deflate;" + base64.b64encode(zlib.compress(json.dumps(state).encode())).decode()


def decode_state(value: str) -> dict:
    encoded = value.split(";", 1)[1]
    return json.loads(zlib.decompress(base64.b64decode(encoded + "=" * (-len(encoded) % 4))))


def install_views(url: str = DEFAULT_URL) -> list[str]:
    """Writes the views to every bmh_ml experiment and returns their links."""
    client = configure_mlflow().MlflowClient()
    links = []
    now = int(time.time() * 1000)
    for experiment_name in EXPERIMENT_NAMES.values():
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            continue
        for view_id, (name, state) in get_views().items():
            existing = experiment.tags.get(TAG_PREFIX + view_id)
            created = json.loads(existing)["createdAt"] if existing else now
            value = {"name": name, "createdAt": created, "updatedAt": now, "state": encode_state(state)}
            client.set_experiment_tag(experiment.experiment_id, TAG_PREFIX + view_id, json.dumps(value))
            mode = "&compareRunsMode=CHART" if state.get("compareRunCharts") else ""  # opens the chart tab, the view does not store it
            links.append(f"{experiment_name} | {name}: {url}/#/experiments/{experiment.experiment_id}/runs?viewStateShareKey={view_id}{mode}")
    return links


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Installs the saved views of the MLflow UI for the bmh_ml experiments")
    parser.add_argument("--url", default=DEFAULT_URL, help="Address of the UI, for the printed links")
    args = parser.parse_args(argv)
    for link in install_views(args.url):
        print(link)


if __name__ == "__main__":
    main()
