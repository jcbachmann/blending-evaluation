import numpy as np
import pytest

from bmh_ml import build_bundle
from bmh_ml.datasets.store import load_bundle, load_dataset, load_training_dataset
from bmh_ml.evaluation.transfer import TransferConfig
from bmh_ml.refine import RefineConfig, get_args, get_round_seeds, perturb, refine
from bmh_ml.settings import X_MAX, X_MIN

SMALL = ["--train-size", "40", "--val-size", "10", "--test-size", "10", "--val-repeats", "2", "--test-repeats", "2", "--n-jobs", "1", "--stress-random", "1"]


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_ML_STORE", str(tmp_path / "store"))
    return tmp_path / "store"


def test_perturbations_stay_within_the_bounds():
    deposition = np.array([[X_MIN] * 20, [X_MAX] * 20, [30.0] * 20])

    perturbed = perturb(deposition, 3, 5.0, np.random.default_rng(0))

    assert perturbed.shape == (9, 20)
    assert perturbed.min() >= X_MIN
    assert perturbed.max() <= X_MAX
    assert not np.array_equal(perturbed[6:], np.repeat(deposition[2:], 3, axis=0))


def test_the_seeds_of_the_rounds_do_not_overlap_with_each_other_or_the_transfer_test():
    seeds = [set(get_round_seeds(k, 10)) | set(get_round_seeds(k, 10, 500)) for k in (1, 2, 3)]

    assert not seeds[0] & seeds[1]
    assert not seeds[1] & seeds[2]
    assert not seeds[0] & set(TransferConfig().seeds)


def test_the_command_line_fills_the_configuration():
    args = get_args(["--base", "B", "--name", "R", "--model", "lightgbm", "--param", "num_leaves=15", "--rounds", "2", "--control"])

    assert (args.base, args.name, args.model, args.params, args.rounds, args.control) == ("B", "R", "lightgbm", {"num_leaves": 15}, 2, True)


def test_the_loop_adds_the_solutions_found_to_the_training_data_round_by_round():
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("sklearn")
    build_bundle.build_bundle(build_bundle.get_args(["--name", "base", "--scope", "S2", "--t3-materials", "1", "--t3-depositions", "2", *SMALL]))
    config = RefineConfig(
        base="base",
        name="R",
        model="ridge",
        params={},
        rounds=2,
        optimizations=1,
        population_size=6,
        evaluations=12,
        perturbations=2,
        valop_optimizations=1,
        valop_repeats=2,
        control=True,
        transfer=TransferConfig(seeds=(1,), population_size=6, evaluations=12, repeats=2, reference_set="T1", n_jobs=1),
        reference_set="T1",
        n_jobs=1,
    )

    run_ids = refine(config)

    assert len(run_ids) == 4  # rounds 0, 1, 2 and the control
    first, final, control = load_bundle("R-r1"), load_bundle("R"), load_bundle("R-control")
    base = load_bundle("base")
    assert first.train == final.train == base.train
    assert len(first.train_extra) == 1
    assert final.train_extra[:1] == first.train_extra
    assert len(final.train_extra) == 2
    assert "valop" not in first.val_extra
    assert final.val_extra["valop"] == control.val_extra["valop"]
    added = [load_dataset(dataset_id) for dataset_id in final.train_extra]
    assert all(dataset.generator == "refinement" for dataset in added)
    assert added[0].source["front_solutions"] * 3 == len(added[0])  # the solutions and two perturbations of each
    assert len(load_training_dataset(control)) == len(load_training_dataset(final))
    assert load_dataset(final.val_extra["valop"]).repeats == 2

    runs = {run_id: mlflow.get_run(run_id) for run_id in run_ids}
    assert all(run.data.tags["refine"] == "R" for run in runs.values())
    parent_id = runs[run_ids[1]].data.tags["mlflow.parentRunId"]
    assert all(run.data.tags["mlflow.parentRunId"] == parent_id for run in runs.values())
    parent = mlflow.get_run(parent_id)
    assert parent.data.tags["final_run"] == run_ids[2]
    history = mlflow.MlflowClient().get_metric_history(parent_id, "T1/F1/nrmse")
    assert sorted(metric.step for metric in history) == [0, 1, 2]
    assert "refine/F2/front_bias" in parent.data.metrics
    assert "transfer/hv_ratio" in runs[run_ids[2]].data.metrics
    assert "valop/F1/rmse" in runs[run_ids[2]].data.metrics
    assert "R/valop/F1/rmse" in runs[run_ids[0]].data.metrics  # the earlier rounds on the final validation sets
    description = mlflow.get_run(parent_id).data.tags["mlflow.note.content"]
    assert "Refinement loop R" in description
    assert all(run_id in description for run_id in run_ids)
    assert runs[run_ids[2]].data.tags["source_model_run"] == run_ids[1]
    assert "Round 2 of the refinement loop" in runs[run_ids[2]].data.tags["mlflow.note.content"]


def test_the_loop_can_draw_new_materials_every_round():
    pytest.importorskip("mlflow")
    pytest.importorskip("sklearn")
    build_bundle.build_bundle(build_bundle.get_args(["--name", "base", "--scope", "S2", "--t3-materials", "1", "--t3-depositions", "2", *SMALL]))
    config = RefineConfig(
        base="base",
        name="M",
        model="ridge",
        params={},
        rounds=2,
        optimizations=1,
        population_size=6,
        evaluations=12,
        perturbations=1,
        valop_optimizations=1,
        valop_repeats=2,
        control=True,
        materials_per_round=3,
        n_jobs=1,
    )

    refine(config)

    final = load_bundle("M")
    first, second = (load_dataset(dataset_id) for dataset_id in final.train_extra)
    assert len(np.unique(first.full_material(), axis=0)) == 3
    assert not np.array_equal(np.unique(first.full_material(), axis=0), np.unique(second.full_material(), axis=0))  # new materials each round
    assert first.source["materials"] == 3
    valop = load_dataset(final.val_extra["valop"])
    assert len(np.unique(valop.full_material(), axis=0)) == 6
    control = load_dataset(load_bundle("M-control").train_extra[0])
    assert len(control) == len(first) + len(second)
    assert len(np.unique(control.full_material(), axis=0)) == len(control)
