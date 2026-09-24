"""Human-readable descriptions of the runs and experiments, so an attempt can be reviewed and revisited without any other notes.

The description of a run (the tag `mlflow.note.content`, the "Description" in the MLflow UI) is generated from what the run logged: its
purpose, the data, the parameters, the results, links to the related runs and how to reproduce it. Generating it from the logged data means
new and old runs get the same description, and it can be refreshed whenever a run gets new results (a later transfer test or evaluation).
The generated text sits between two markers; text written outside of them in the UI is kept.

    python -m bmh_ml.tracking.annotate --all      # (re)generate the descriptions of all runs and experiments
"""

import argparse
import json
import logging
import math

from bmh_ml.tracking.runs import EXPERIMENT_NAMES, configure_mlflow
from bmh_ml.tracking.store import get_bundles_directory, get_datasets_directory

NOTE = "mlflow.note.content"
BEGIN = "<!-- bmh_ml: generated description, text outside of these markers is kept -->"
END = "<!-- bmh_ml: end of the generated description -->"
SETS = ("val", "valop", "T1", "T2", "T2s", "T3", "T5")
SET_NAMES = {
    "val": "validation, random inputs, stops the training",
    "valop": "validation, operating region, for choosing models",
    "T1": "test, random inputs",
    "T2": "test, fronts of the optimization on the simulator",
    "T2s": "test, fronts of the optimization on the old LSTM surrogate",
    "T3": "test, unseen materials",
    "T5": "test, extreme depositions",
}
INFRASTRUCTURE_PARAMS = ("model", "bundle", "scope", "seed", "train_dataset", "train_extra_datasets", "train_repeats")
EXPERIMENT_DESCRIPTIONS = {
    "S1": (
        "Scope **S1, fixed material**: models predict the objectives F1 (volume-weighted standard deviation of the reclaimed quality) and F2 "
        "(standard deviation of the reclaimed volume per slice from the ideal stockpile) from the 20 deposition positions, for one material. "
        "This is the setting of the optimization experiments. Plan, decisions and conclusions: `apps/bmh_ml/PLAN.md`; "
        "leaderboard: `python -m bmh_ml.report`."
    ),
    "S2": (
        "Scope **S2, general material**: models predict F1 and F2 from the 50 material values and the 20 deposition positions. "
        "Plan, decisions and conclusions: `apps/bmh_ml/PLAN.md`."
    ),
}


def run_link(experiment_id: str, run_id: str, text: str | None = None) -> str:
    """A link that the MLflow UI opens as the run page."""
    return f"[{text or run_id[:8]}](#/experiments/{experiment_id}/runs/{run_id})"


def merge_description(existing: str | None, generated: str) -> str:
    """The generated text, replacing an earlier generated text, and the rest of the existing description kept.

    The first line of the generated text is a summary, it comes before the begin marker so that the description column of the runs table
    shows it; the rest sits between the markers.
    """
    summary, _, body = generated.strip().partition("\n")
    block = f"{summary}\n{BEGIN}\n{body.strip()}\n{END}"
    if not existing:
        return block
    if BEGIN in existing and END in existing:
        begin = existing.index(BEGIN)
        start = existing.rfind("\n", 0, max(begin - 1, 0)) + 1  # the summary line before the marker belongs to the generated text
        end = existing.index(END, begin) + len(END)
        return f"{existing[:start]}{block}{existing[end:]}"
    return f"{existing.rstrip()}\n\n{block}"


def get_summary(metrics: dict) -> str:
    """The results that tell attempts apart, for the first line of a description."""
    parts = []
    for name, label in (("transfer/hv_ratio", "transfer hv"), ("valop/F2/nrmse", "valop F2 nrmse"), ("T2/F2/r2", "T2 F2 R2"), ("val/F1/nrmse", "val F1 nrmse")):
        if name in metrics:
            parts.append(f"{label} {format_number(metrics[name])}")
    return "; ".join(parts)


def format_number(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{digits}g}" if abs(value) < 1000 else f"{value:,.0f}"


def read_manifest(dataset_id: str) -> dict | None:
    path = get_datasets_directory() / dataset_id / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else None


def read_bundle(name: str) -> dict | None:
    path = get_bundles_directory() / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def describe_dataset(dataset_id: str) -> str:
    manifest = read_manifest(dataset_id)
    if manifest is None:
        return f"`{dataset_id}` (not in this store)"
    repeats = f", labels the mean of {manifest['repeats']} simulations" if manifest["repeats"] > 1 else ""
    return f"`{dataset_id}`: {manifest['size']:,} rows, generator `{manifest['generator']}`{repeats}"


def get_kind(params: dict, tags: dict) -> str:
    if "sweep.model" in params:
        return "sweep"
    if "refine.base" in params:
        return "refine"
    if "sweep" in tags and "trial" in tags:
        return "sweep-trial"
    if "refine" in tags and "round" in tags:
        return "refine-control" if tags["round"] == "-1" else "refine-round"
    if "model" in params:
        return "training"
    return "other"


def describe_purpose(kind: str, run, parent_link: str) -> list[str]:
    params, tags = run.data.params, run.data.tags
    experiment_id = run.info.experiment_id
    model, bundle, seed = params.get("model"), params.get("bundle"), params.get("seed")
    lines = [f"Training of the **{model}** model on bundle **{bundle}** (scope {params.get('scope')}), seed {seed}."]
    if kind == "refine-round":
        round_number = int(tags["round"])
        if round_number == 0:
            lines.append(f"Round 0 of the refinement loop {parent_link}: the starting model, trained on the base data only.")
        else:
            source = tags.get("source_model_run") or get_source_model_run(params)
            found_by = f"the model of round {round_number - 1} ({run_link(experiment_id, source)})" if source else f"the model of round {round_number - 1}"
            lines.append(
                f"Round {round_number} of the refinement loop {parent_link}: trained on the base data plus the solutions found by optimizing "
                f"{found_by} and perturbations of them, together with the data added in the earlier rounds."
            )
    elif kind == "refine-control":
        lines.append(
            f"Control of the refinement loop {parent_link}: the base data plus as many *random* rows as the loop added, to separate the "
            "effect of where the added data lies from the effect of its amount."
        )
    elif kind == "sweep-trial":
        lines.append(f"Trial {tags['trial']} of the hyperparameter sweep {parent_link}, parameters chosen by Optuna.")
    if tags.get("purpose"):
        lines.append(tags["purpose"])
    return lines


def get_source_model_run(params: dict) -> str | None:
    """The run whose model found the data added last, as the manifest of the added dataset records it."""
    added = list(filter(None, params.get("train_extra_datasets", "").split(",")))
    manifest = read_manifest(added[-1]) if added else None
    return manifest["source"].get("model_run") if manifest else None


def describe_data(params: dict) -> list[str]:
    bundle = read_bundle(params.get("bundle", ""))
    lines = ["", "**Data**", ""]
    lines.append(f"- training: {describe_dataset(params['train_dataset'])}" if "train_dataset" in params else "- training: unknown")
    lines.extend(f"- training, added: {describe_dataset(dataset_id)}" for dataset_id in filter(None, params.get("train_extra_datasets", "").split(",")))
    lines.extend(f"- {name}, {SET_NAMES[name]}: {describe_dataset(params[f'{name}_dataset'])}" for name in SETS if f"{name}_dataset" in params)
    if bundle and bundle.get("notes"):
        notes = {key: value for key, value in bundle["notes"].items() if key != "arguments"}
        if notes:
            lines.append(f"- bundle notes: {json.dumps(notes, sort_keys=True)}")
    return lines


def describe_results(metrics: dict, prefix: str = "") -> list[str]:
    rows = []
    for name in SETS:
        if f"{prefix}{name}/F1/nrmse" not in metrics and f"{prefix}{name}/F1/r2" not in metrics:
            continue
        cells = [name]
        for objective in ("F1", "F2"):
            key = f"{prefix}{name}/{objective}"
            cells += [format_number(metrics.get(f"{key}/nrmse")), format_number(metrics.get(f"{key}/r2")), format_number(metrics.get(f"{key}/r2_ceiling"))]
        cells.append(format_number(metrics.get(f"{prefix}{name}/F2/negative_rate")))
        rows.append("| " + " | ".join(cells) + " |")
    if not rows:
        return []
    header = "| set | F1 nrmse | F1 R2 | F1 R2 ceiling | F2 nrmse | F2 R2 | F2 R2 ceiling | F2 negative |"
    return [header, "|---|---|---|---|---|---|---|---|", *rows]


def describe_transfer(metrics: dict, params: dict) -> list[str]:
    if "transfer/hv_ratio" not in metrics:
        return []
    settings = ", ".join(f"{key.removeprefix('transfer.')} {value}" for key, value in sorted(params.items()) if key.startswith("transfer."))
    get = metrics.get
    return [
        "",
        "**Transfer test** (optimize the model, simulate what it finds, compare with the best front of the simulator)",
        "",
        f"- hypervolume ratio {format_number(get('transfer/hv_ratio'))} over all runs, {format_number(get('transfer/hv_ratio_run_mean'))} per run "
        f"(min {format_number(get('transfer/hv_ratio_run_min'))}); IGD+ {format_number(get('transfer/igd_plus'))}",
        f"- the model promised a hypervolume ratio of {format_number(get('transfer/predicted_hv_ratio'))}; bias of its predictions on the solutions it "
        f"found: F1 {format_number(get('transfer/F1/bias'))}, F2 {format_number(get('transfer/F2/bias'))}; solutions with a negative prediction "
        f"{format_number(100 * get('transfer/negative_rate', float('nan')))} %",
        f"- settings: {settings}",
    ]


def get_training_title(run) -> str:
    params, tags = run.data.params, run.data.tags
    kind = get_kind(params, tags)
    subject = f"{params.get('model')} on {params.get('bundle')}, seed {params.get('seed')}"
    if kind == "refine-round":
        return f"Refinement {tags['refine']} round {tags['round']}: {subject}"
    if kind == "refine-control":
        return f"Control of refinement {tags['refine']}: {subject}"
    if kind == "sweep-trial":
        return f"Sweep {tags['sweep']} trial {tags['trial']}: {subject}"
    return subject


def describe_training(run, parent_link: str) -> str:
    params, tags, metrics = run.data.params, run.data.tags, run.data.metrics
    kind = get_kind(params, tags)
    summary = get_summary(metrics)
    lines = [get_training_title(run) + (f" | {summary}" if summary else "")]
    lines += describe_purpose(kind, run, parent_link)
    lines += describe_data(params)
    model_params = {key: value for key, value in sorted(params.items()) if key not in INFRASTRUCTURE_PARAMS and not key.endswith("_dataset") and "." not in key}
    lines += ["", "**Parameters:** " + ", ".join(f"{key}={value}" for key, value in model_params.items())]
    training = [f"{format_number(metrics.get('train/rows'))} rows", f"{format_number(metrics.get('train/seconds'))} s"]
    for objective in ("F1", "F2"):
        if f"train/{objective}/epochs" in metrics:
            training.append(f"{objective} {format_number(metrics[f'train/{objective}/epochs'])} epochs")
        if f"train/{objective}/trees" in metrics:
            training.append(f"{objective} {format_number(metrics[f'train/{objective}/trees'])} trees")
    lines += [f"**Training:** {', '.join(training)}; prediction {format_number(metrics.get('throughput/batch_10000'))} rows/s in batches of 10,000"]
    results = describe_results(metrics)
    if results:
        lines += ["", "**Results** (nrmse: error in units of the noise of one simulation, 0.25 is perfect with 16-repeat labels)", "", *results]
    lines += describe_transfer(metrics, params)
    later = sorted({name.split("/")[0] for name in metrics if name.count("/") == 3})
    for bundle in later:
        lines += ["", f"**Evaluated later on bundle {bundle}**", "", *describe_results(metrics, f"{bundle}/")]
    reproduce = f"python -m bmh_ml.train --bundle {params.get('bundle')} --model {params.get('model')} --seed {params.get('seed')}"
    if model_params:
        reproduce += " --param " + " ".join(f"{key}={value}" for key, value in model_params.items())
    lines += ["", f"**Reproduce:** `{reproduce}`"]
    lines += [f"**Code:** {tags.get('code_version', 'unknown')}, uv.lock {tags.get('lock_hash', 'unknown')}, {tags.get('hardware.cpu_count', '?')} CPUs"]
    return "\n".join(lines)


def get_children(client, run) -> list:
    return client.search_runs(
        [run.info.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{run.info.run_id}'", order_by=["attributes.start_time ASC"], max_results=1000
    )


def describe_refine(client, run) -> str:
    params, tags = run.data.params, run.data.tags
    experiment_id = run.info.experiment_id
    setup = {key.removeprefix("refine."): value for key, value in params.items() if key.startswith("refine.")}
    model_params = {key: value for key, value in params.items() if not key.startswith("refine.")}
    final = client.get_run(tags["final_run"]).data.metrics if tags.get("final_run") else {}
    summary = get_summary(final)
    lines = [
        f"Refinement loop {setup.get('name')}: {setup.get('model')}, {setup.get('rounds')} rounds from {setup.get('base')}"
        + (f" | final round: {summary}" if summary else ""),
        f"Training data in the region the optimizer works in. Each round optimizes the current model "
        f"({setup.get('optimizations')} NSGA-III runs, population {setup.get('population_size')}, {setup.get('evaluations')} evaluations), "
        f"simulates the solutions found and {setup.get('perturbations')} perturbations of each (sd {setup.get('perturbation_sd')}), adds them "
        f"to the training data and trains the model again. Base bundle **{setup.get('base')}**, model **{setup.get('model')}**, "
        f"{setup.get('rounds')} rounds; final bundle **{setup.get('name')}** with the operating-region validation set `valop`.",
        "",
        "**Model parameters:** " + (", ".join(f"{key}={value}" for key, value in sorted(model_params.items())) or "defaults"),
    ]
    if setup.get("start_run"):
        lines.append(f"**Round 0** is the existing run {run_link(experiment_id, setup['start_run'])}.")
    history = {metric.step: metric.value for metric in client.get_metric_history(run.info.run_id, "refine/rows_added")}
    rows = []
    children = get_children(client, run)
    if setup.get("start_run"):
        children = [client.get_run(setup["start_run"]), *children]
    for child in children:
        metrics = child.data.metrics
        round_tag = child.data.tags.get("round", "0")
        label = "control" if round_tag == "-1" else round_tag
        valop = metrics.get("valop/F2/nrmse", metrics.get(f"{setup.get('name')}/valop/F2/nrmse"))
        cells = [
            label,
            run_link(experiment_id, child.info.run_id, child.info.run_name),
            child.data.params.get("bundle", ""),
            format_number(metrics.get("train/rows")),
            format_number(history.get(int(round_tag))) if round_tag not in ("-1", "0") else "",
            format_number(metrics.get("transfer/hv_ratio")),
            format_number(metrics.get("transfer/hv_ratio_run_mean")),
            format_number(metrics.get("transfer/negative_rate")),
            format_number(metrics.get("transfer/F2/bias")),
            format_number(metrics.get("T2/F2/r2")),
            format_number(valop),
        ]
        rows.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "**Rounds**",
        "",
        "| round | run | bundle | training rows | rows added | transfer hv | hv per run | negative | F2 bias | T2 F2 R2 | valop F2 nrmse |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        *rows,
    ]
    if tags.get("final_run"):
        lines += ["", f"**Final model:** {run_link(experiment_id, tags['final_run'])} on bundle {tags.get('final_bundle')}."]
    lines += [
        "",
        f"**Reproduce:** `python -m bmh_ml.refine --base {setup.get('base')} --name {setup.get('name')} --model {setup.get('model')} ...` "
        "(with the settings above)",
    ]
    lines += [f"**Code:** {tags.get('code_version', 'unknown')}"]
    return "\n".join(lines)


def describe_sweep(client, run) -> str:
    params, tags = run.data.params, run.data.tags
    experiment_id = run.info.experiment_id
    fixed = {key.removeprefix("sweep.fixed."): value for key, value in params.items() if key.startswith("sweep.fixed.")}
    best = run.data.metrics.get("sweep/best_objective")
    lines = [
        f"Sweep {tags.get('sweep')}: {params.get('sweep.model')} on {params.get('sweep.bundle')}"
        + (f" | best objective {format_number(best, 4)}" if best is not None else ""),
        f"Hyperparameter sweep {tags.get('sweep')} of the **{params.get('sweep.model')}** model on bundle **{params.get('sweep.bundle')}** with "
        f"Optuna (TPE), {params.get('sweep.trials')} trials in this call, {params.get('sweep.workers', '1')} at a time. Objective (minimized): the mean of "
        f"`{params.get('sweep.objective')}`, validation sets only.",
        "",
        "**Fixed parameters:** " + (", ".join(f"{key}={value}" for key, value in sorted(fixed.items())) or "none"),
    ]
    trials = []
    for child in get_children(client, run):
        objective = [child.data.metrics.get(metric) for metric in params.get("sweep.objective", "").split(",")]
        if objective and all(value is not None for value in objective):
            trials.append((sum(objective) / len(objective), child))
    trials.sort(key=lambda item: item[0])
    rows = []
    for value, child in trials[:10]:
        child_params = child.data.params
        shown = ", ".join(f"{key}={child_params[key]}" for key in sorted(child_params) if key not in INFRASTRUCTURE_PARAMS and not key.endswith("_dataset"))
        rows.append(
            f"| {child.data.tags.get('trial', '')} | {run_link(experiment_id, child.info.run_id, child.info.run_name)} | {format_number(value, 4)} | {shown} |"
        )
    lines += ["", f"**Best trials** ({len(trials)} finished)", "", "| trial | run | objective | parameters |", "|---|---|---|---|", *rows]
    if tags.get("best_run"):
        lines += ["", f"**Best run:** {run_link(experiment_id, tags['best_run'])}, trial {tags.get('best_trial')}."]
    lines += [
        f"**Continue:** `python -m bmh_ml.sweep --name {tags.get('sweep')} --bundle {params.get('sweep.bundle')} "
        f"--model {params.get('sweep.model')} --trials N`"
    ]
    return "\n".join(lines)


def describe_run(client, run) -> str | None:
    kind = get_kind(run.data.params, run.data.tags)
    if kind == "refine":
        return describe_refine(client, run)
    if kind == "sweep":
        return describe_sweep(client, run)
    if kind == "other":
        return None
    parent_id = run.data.tags.get("mlflow.parentRunId")
    parent_link = ""
    if parent_id:
        parent_link = run_link(run.info.experiment_id, parent_id, client.get_run(parent_id).info.run_name)
    return describe_training(run, parent_link)


def annotate_run(run_id: str) -> None:
    """Writes the generated description of a run. Never fails the caller: a run is worth more than its description."""
    try:
        client = configure_mlflow().MlflowClient()
        run = client.get_run(run_id)
        annotate_experiments()
        generated = describe_run(client, run)
        if generated:
            client.set_tag(run_id, NOTE, merge_description(run.data.tags.get(NOTE), generated))
    except Exception:
        logging.warning(f"Could not write the description of run {run_id}", exc_info=True)


def annotate_experiments() -> None:
    client = configure_mlflow().MlflowClient()
    for scope, name in EXPERIMENT_NAMES.items():
        experiment = client.get_experiment_by_name(name)
        if experiment is not None:
            client.set_experiment_tag(experiment.experiment_id, NOTE, merge_description(experiment.tags.get(NOTE), EXPERIMENT_DESCRIPTIONS[scope]))


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="(Re)generates the descriptions of runs and experiments from what they logged")
    parser.add_argument("--run-id", nargs="*", default=[])
    parser.add_argument("--all", action="store_true", help="All runs of the bmh_ml experiments, and the experiments")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    mlflow = configure_mlflow()
    run_ids = list(args.run_id)
    if args.all:
        annotate_experiments()
        experiments = [experiment for name in EXPERIMENT_NAMES.values() if (experiment := mlflow.get_experiment_by_name(name))]
        if experiments:
            runs = mlflow.search_runs([experiment.experiment_id for experiment in experiments], order_by=["attributes.start_time ASC"])
            run_ids += list(runs["run_id"])
    for run_id in run_ids:
        annotate_run(run_id)
    print(f"Descriptions written for {len(run_ids)} runs")


if __name__ == "__main__":
    main()
