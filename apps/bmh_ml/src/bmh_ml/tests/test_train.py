import argparse
import json
from dataclasses import replace

import numpy as np
import pytest

from bmh_ml import build_bundle
from bmh_ml.datasets.store import load_bundle, save_bundle
from bmh_ml.evaluation.transfer import TransferConfig
from bmh_ml.tracking.runs import get_experiment_name, get_finite_metrics
from bmh_ml.train import get_args, parse_parameters, run_training
from bmh_ml.transfer import recompute_transfer, transfer_run

SMALL = ["--train-size", "40", "--val-size", "10", "--test-size", "10", "--val-repeats", "2", "--test-repeats", "2", "--n-jobs", "1", "--stress-random", "1"]


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_ML_STORE", str(tmp_path / "store"))
    return tmp_path / "store"


def test_parameters_are_read_as_yaml_values():
    assert parse_parameters(["width=512", "alpha=0.5", "flag=true", "name=abc", "sizes=[1, 2]"]) == {
        "width": 512,
        "alpha": 0.5,
        "flag": True,
        "name": "abc",
        "sizes": [1, 2],
    }


def test_a_parameter_needs_a_value():
    with pytest.raises(argparse.ArgumentTypeError, match="key=value"):
        parse_parameters(["width"])


def test_the_command_line_is_parsed():
    args = get_args(["--bundle", "B", "--model", "ridge", "--param", "alpha=2", "--seed", "3", "4", "--workers", "2"])

    assert (args.bundle, args.model, args.params, args.seed, args.workers) == ("B", "ridge", {"alpha": 2}, [3, 4], 2)
    with pytest.raises(SystemExit):
        get_args(["--bundle", "B", "--model", "forest"])


def test_metrics_that_are_not_finite_are_not_logged():
    assert get_finite_metrics({"a": 1.0, "b": float("nan"), "c": float("inf")}) == {"a": 1.0}


def test_the_experiment_is_named_after_the_scope():
    assert get_experiment_name("S1") == "S1-fixed-material"
    assert get_experiment_name("S2") == "S2-general-material"


def make_bundle(name: str, *extra: str):
    build_bundle.build_bundle(build_bundle.get_args(["--name", name, "--scope", "S2", "--t3-materials", "1", "--t3-depositions", "3", *SMALL, *extra]))


def test_a_training_run_is_logged_with_everything_needed_to_compare_and_reproduce_it():
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("sklearn")
    make_bundle("B1")

    result = run_training("B1", "ridge", {"alpha": 2.0}, seed=5, run_name="test-run", with_plots=False)

    run = mlflow.get_run(result.run_id)
    bundle = load_bundle("B1")
    assert run.info.run_name == "test-run"
    assert run.data.params["model"] == "ridge"
    assert run.data.params["alpha"] == "2.0"
    assert run.data.params["seed"] == "5"
    assert run.data.params["bundle"] == "B1"
    assert run.data.params["train_dataset"] == bundle.train
    assert run.data.params["T1_dataset"] == bundle.tests["T1"]
    assert "code_version" in run.data.tags
    for name in ("val/F1/rmse", "T1/F2/nrmse", "T3/F1/r2_ceiling", "T5/F2/tail_rmse", "train/seconds", "throughput/batch_100"):
        assert name in run.data.metrics, name
    artifacts = {artifact.path for artifact in mlflow.MlflowClient().list_artifacts(result.run_id)}
    assert {"model", "predictions"} <= artifacts
    description = run.data.tags["mlflow.note.content"]
    assert "**ridge** model on bundle **B1**" in description
    assert bundle.tests["T1"] in description
    assert "python -m bmh_ml.train --bundle B1 --model ridge --seed 5" in description
    contexts = {dataset_input.dataset.name: dataset_input.tags[0].value for dataset_input in run.inputs.dataset_inputs}
    assert contexts[bundle.train] == "training"
    assert contexts[bundle.val] == "validation"
    assert contexts[bundle.tests["T1"]] == "testing"
    assert mlflow.get_experiment(run.info.experiment_id).name == "S2-general-material"


def test_the_model_of_a_run_can_be_evaluated_on_another_bundle():
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("sklearn")
    from bmh_ml.evaluate_run import evaluate_run

    make_bundle("B1")
    make_bundle("B2", "--seed", "2")
    result = run_training("B1", "ridge", {}, with_plots=False)

    metrics = evaluate_run(result.run_id, "B2")

    assert "B2/T1/F1/r2" in metrics
    assert "B2/T1/F1/r2" in mlflow.get_run(result.run_id).data.metrics
    with pytest.raises(ValueError, match="choose another bundle"):
        evaluate_run(result.run_id, "B1")


def test_a_model_is_only_evaluated_on_a_bundle_with_the_same_scope(tmp_path):
    pytest.importorskip("mlflow")
    pytest.importorskip("sklearn")
    from bmh_ml.evaluate_run import evaluate_run

    make_bundle("B1")
    material = tmp_path / "run.json"
    material.write_text(json.dumps({"variables": [[1.0]], "material_variables": np.linspace(5, 6, 50).tolist()}))
    build_bundle.build_bundle(build_bundle.get_args(["--name", "S1-B", "--scope", "S1", "--material-from", f"results:{tmp_path}", *SMALL]))
    result = run_training("B1", "ridge", {}, with_plots=False)

    with pytest.raises(ValueError, match="scope"):
        evaluate_run(result.run_id, "S1-B")


def test_plots_are_logged_if_wanted():
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("matplotlib")
    make_bundle("B1")

    result = run_training("B1", "mean", {}, with_plots=True)

    plots = {artifact.path for artifact in mlflow.MlflowClient().list_artifacts(result.run_id, "plots")}
    assert {"plots/val.png", "plots/T1.png", "plots/T3.png", "plots/T5.png"} <= plots


def test_the_extra_datasets_of_a_bundle_are_trained_on_and_evaluated():
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("sklearn")
    make_bundle("B1")
    base = load_bundle("B1")
    extended = replace(base, name="B2", train_extra=[base.val], val_extra={"valop": base.tests["T1"]})
    save_bundle(extended)

    result = run_training("B2", "ridge", {}, with_plots=False, tags={"purpose": "test"})

    run = mlflow.get_run(result.run_id)
    assert run.data.metrics["train/rows"] == 50
    assert run.data.params["train_extra_datasets"] == base.val
    assert run.data.params["valop_dataset"] == base.tests["T1"]
    assert run.data.metrics["valop/F1/rmse"] == run.data.metrics["T1/F1/rmse"]
    assert run.data.tags["purpose"] == "test"


def test_the_transfer_test_is_logged_with_the_training_or_later():
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("sklearn")
    make_bundle("B1")
    config = TransferConfig(seeds=(1,), population_size=6, evaluations=12, repeats=2, reference_set="T1", n_jobs=1)

    result = run_training("B1", "ridge", {}, with_plots=False, transfer=config)

    run = mlflow.get_run(result.run_id)
    assert run.data.metrics["transfer/hv_ratio"] == result.metrics["transfer/hv_ratio"]
    assert run.data.params["transfer.seeds"] == "(1,)"
    assert "transfer/solutions.npz" in {artifact.path for artifact in mlflow.MlflowClient().list_artifacts(result.run_id, "transfer")}

    later = run_training("B1", "ridge", {}, with_plots=False)
    end_time = mlflow.get_run(later.run_id).info.end_time
    transfer_run(later.run_id, config, with_plots=False)
    assert "transfer/hv_ratio" in mlflow.get_run(later.run_id).data.metrics
    assert mlflow.get_run(later.run_id).info.end_time == end_time  # the finished run is not reopened

    recomputed = recompute_transfer(later.run_id, with_plots=False)
    metrics = mlflow.get_run(later.run_id).data.metrics
    assert metrics["transfer/chevron_hv"] == pytest.approx(recomputed["chevron_hv"])
    assert metrics["transfer/hv_ratio"] == pytest.approx(recomputed["hv_ratio"])  # the same solutions give the same metrics
    assert "transfer/optimize_seconds" in metrics
    with pytest.raises(ValueError, match="T2"):
        transfer_run(later.run_id, TransferConfig(), with_plots=False)
