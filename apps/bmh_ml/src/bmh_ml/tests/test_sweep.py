import pytest

from bmh_ml import build_bundle
from bmh_ml.models.registry import get_model_class
from bmh_ml.sweep import SEARCH_SPACES, SweepConfig, get_args, get_objective_metrics, run_sweep, split_trials, suggest

SMALL = ["--train-size", "40", "--val-size", "10", "--test-size", "10", "--val-repeats", "2", "--test-repeats", "2", "--n-jobs", "1", "--stress-random", "1"]


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_ML_STORE", str(tmp_path / "store"))
    return tmp_path / "store"


class FixedTrial:
    """Answers every suggestion with the lowest value, like an Optuna trial would with a fixed choice."""

    def suggest_float(self, name, low, high, log=False):  # noqa: ARG002 - the interface of an Optuna trial
        return low

    def suggest_int(self, name, low, high, log=False):  # noqa: ARG002 - the interface of an Optuna trial
        return low

    def suggest_categorical(self, name, choices):  # noqa: ARG002 - the interface of an Optuna trial
        return choices[0]


def test_every_search_space_belongs_to_a_model_and_only_uses_its_parameters():
    for model, space in SEARCH_SPACES.items():
        assert set(space) <= set(get_model_class(model).defaults), model  # the model modules import their frameworks only when used


def test_fixed_parameters_are_not_searched():
    params = suggest(FixedTrial(), SEARCH_SPACES["lightgbm"], {"num_leaves": 7, "n_jobs": 1})

    assert params["num_leaves"] == 7
    assert params["n_jobs"] == 1
    assert params["learning_rate"] == 0.01


def test_the_objective_is_the_validation_sets_unless_given_and_never_a_test_set():
    config = SweepConfig("s", "B", "ridge")
    assert get_objective_metrics(config, ["val", "valop"]) == ["val/F1/nrmse", "val/F2/nrmse", "valop/F1/nrmse", "valop/F2/nrmse"]

    config.objective = ["valop/F2/rmse"]
    assert get_objective_metrics(config, ["val", "valop"]) == ["valop/F2/rmse"]

    config.objective = ["T2/F2/nrmse"]
    with pytest.raises(ValueError, match="test sets"):
        get_objective_metrics(config, ["val", "valop"])


def test_the_command_line_fills_the_configuration():
    args = get_args(["--name", "s", "--bundle", "B", "--model", "lightgbm", "--trials", "3", "--param", "n_jobs=2", "--objective", "val/F1/nrmse"])

    assert (args.name, args.bundle, args.model, args.trials, args.fixed, args.objective) == ("s", "B", "lightgbm", 3, {"n_jobs": 2}, ["val/F1/nrmse"])


def test_every_trial_is_a_nested_run_and_the_sweep_can_be_continued():
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("optuna")
    pytest.importorskip("sklearn")
    build_bundle.build_bundle(build_bundle.get_args(["--name", "B", "--scope", "S2", "--t3-materials", "1", "--t3-depositions", "2", *SMALL]))

    study = run_sweep(SweepConfig("s", "B", "ridge", trials=3))
    study = run_sweep(SweepConfig("s", "B", "ridge", trials=2))

    assert len(study.trials) == 5
    runs = mlflow.search_runs(experiment_names=["S2-general-material"], filter_string="tags.sweep = 's'")
    parents = runs[runs["tags.mlflow.parentRunId"].isna()]
    trials = runs[runs["tags.mlflow.parentRunId"].notna()]
    assert len(parents) == 2
    assert len(trials) == 5
    assert set(trials["tags.mlflow.parentRunId"]) == set(parents["run_id"])
    best_run = mlflow.get_run(study.best_trial.user_attrs["run_id"])
    assert best_run.data.params["alpha"] == str(study.best_trial.params["alpha"])
    expected = (best_run.data.metrics["val/F1/nrmse"] + best_run.data.metrics["val/F2/nrmse"]) / 2
    assert study.best_value == pytest.approx(expected)
    assert set(parents["tags.best_run"]) <= set(trials["run_id"])
    description = mlflow.get_run(parents["run_id"].iloc[-1]).data.tags["mlflow.note.content"]
    assert "Hyperparameter sweep s" in description
    assert study.best_trial.user_attrs["run_id"] in description


def test_trials_can_run_in_parallel_processes():
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("optuna")
    pytest.importorskip("sklearn")
    build_bundle.build_bundle(build_bundle.get_args(["--name", "B", "--scope", "S2", "--t3-materials", "1", "--t3-depositions", "2", *SMALL]))

    study = run_sweep(SweepConfig("p", "B", "ridge", trials=3, workers=2))

    assert len(study.trials) == 3
    runs = mlflow.search_runs(experiment_names=["S2-general-material"], filter_string="tags.sweep = 'p'")
    parent = runs[runs["tags.mlflow.parentRunId"].isna()]
    assert len(parent) == 1
    assert set(runs[runs["tags.mlflow.parentRunId"].notna()]["tags.mlflow.parentRunId"]) == {parent["run_id"].iloc[0]}


def test_the_trials_are_split_between_the_workers():
    assert split_trials(5, 2) == [3, 2]
    assert split_trials(1, 3) == [1, 0, 0]
