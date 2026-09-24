import json

import numpy as np
import pytest

from bmh_ml import build_bundle
from bmh_ml.datasets.generators import (
    find_result_files,
    load_fixed_material,
    load_front_depositions,
    random_depositions,
    random_materials,
    stress_depositions,
)
from bmh_ml.datasets.manifest import Dataset, concatenate_datasets, make_features
from bmh_ml.datasets.simulate import build_dataset, simulate
from bmh_ml.datasets.store import (
    Bundle,
    get_bundle_file,
    get_evaluation_sets,
    list_bundles,
    list_datasets,
    load_bundle,
    load_dataset,
    load_training_dataset,
    save_bundle,
    save_dataset,
)
from bmh_ml.evaluation.noise import get_noise_summary
from bmh_ml.settings import DEPOSITION_LENGTH, MATERIAL_LENGTH, MATERIAL_MAX, MATERIAL_MIN, X_MAX, X_MIN
from bmh_ml.variables import generate_deposition_variables, generate_material_variables


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_ML_STORE", str(tmp_path / "store"))
    return tmp_path / "store"


def make_dataset(n: int = 5, seed: int = 0, repeats: int = 1, name: str = "test") -> Dataset:
    rng = np.random.default_rng(seed)
    return Dataset(
        name=name,
        generator="random",
        seed=seed,
        repeats=repeats,
        material=random_materials(n, rng),
        deposition=random_depositions(n, rng),
        y=rng.random((n, 2)),
        y_noise_sd=rng.random((n, 2)) if repeats > 1 else None,
    )


def test_the_variable_generators_are_reproducible_with_a_generator():
    first = generate_material_variables(MATERIAL_LENGTH, MATERIAL_MIN, MATERIAL_MAX, rng=np.random.default_rng(3))
    second = generate_material_variables(MATERIAL_LENGTH, MATERIAL_MIN, MATERIAL_MAX, rng=np.random.default_rng(3))
    assert first == second
    assert np.array_equal(
        generate_deposition_variables(DEPOSITION_LENGTH, X_MIN, X_MAX, rng=np.random.default_rng(3)),
        generate_deposition_variables(DEPOSITION_LENGTH, X_MIN, X_MAX, rng=np.random.default_rng(3)),
    )


def test_the_variable_generators_are_random_without_a_generator():
    assert generate_material_variables(MATERIAL_LENGTH, MATERIAL_MIN, MATERIAL_MAX) != generate_material_variables(MATERIAL_LENGTH, MATERIAL_MIN, MATERIAL_MAX)


def test_random_inputs_have_the_right_shape_and_range():
    rng = np.random.default_rng(0)
    deposition = random_depositions(30, rng)
    material = random_materials(4, rng)

    assert deposition.shape == (30, DEPOSITION_LENGTH)
    assert deposition.min() >= X_MIN
    assert deposition.max() <= X_MAX
    assert material.shape == (4, MATERIAL_LENGTH)


def test_stress_depositions_stay_in_the_range_and_contain_the_extremes():
    deposition = stress_depositions(np.random.default_rng(0), n_random_binary=10)

    assert deposition.shape[1] == DEPOSITION_LENGTH
    assert deposition.min() == X_MIN
    assert deposition.max() == X_MAX
    assert any(np.all(row == X_MIN) for row in deposition)
    assert len(np.unique(deposition, axis=0)) == len(deposition)


def test_the_content_hash_depends_on_the_content_only():
    first, second = make_dataset(seed=1), make_dataset(seed=1)
    other = make_dataset(seed=2)

    assert first.content_hash() == second.content_hash()
    assert first.dataset_id() == second.dataset_id()
    assert first.content_hash() != other.content_hash()


def test_a_dataset_is_checked_on_creation():
    dataset = make_dataset()
    with pytest.raises(ValueError, match="deposition"):
        Dataset("bad", "random", 0, 1, dataset.material, dataset.deposition[:, :10], dataset.y)
    with pytest.raises(ValueError, match="y must have shape"):
        Dataset("bad", "random", 0, 1, dataset.material, dataset.deposition, dataset.y[:2])


def test_features_of_the_scopes():
    dataset = make_dataset(n=3)

    assert np.array_equal(dataset.features("S1"), dataset.deposition)
    assert dataset.features("S2").shape == (3, MATERIAL_LENGTH + DEPOSITION_LENGTH)
    assert np.array_equal(dataset.features("S2")[:, :MATERIAL_LENGTH], dataset.material)
    with pytest.raises(ValueError, match="Unknown scope"):
        dataset.features("S3")


def test_a_single_material_is_used_for_all_samples():
    dataset = make_dataset(n=4)
    dataset.material = dataset.material[:1]

    assert dataset.features("S2").shape == (4, MATERIAL_LENGTH + DEPOSITION_LENGTH)
    assert np.all(dataset.features("S2")[:, :MATERIAL_LENGTH] == dataset.material[0])


def test_a_dataset_survives_a_round_trip_through_the_store():
    dataset = make_dataset(repeats=4)

    dataset_id = save_dataset(dataset)
    loaded = load_dataset(dataset_id)

    assert loaded.dataset_id() == dataset_id
    assert np.array_equal(loaded.y, dataset.y)
    assert np.array_equal(loaded.y_noise_sd, dataset.y_noise_sd)
    assert loaded.repeats == 4
    assert list_datasets()[0]["id"] == dataset_id


def test_identical_content_is_stored_once(store):
    save_dataset(make_dataset())
    save_dataset(make_dataset())

    assert len(list(store.glob("datasets/*/manifest.json"))) == 1


def test_a_damaged_dataset_is_detected(store):
    dataset_id = save_dataset(make_dataset())
    manifest_file = store / "datasets" / dataset_id / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["content_hash"] = "0" * 64
    manifest_file.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="damaged"):
        load_dataset(dataset_id)


def test_an_unknown_dataset_is_reported():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_dataset("missing-0123456789")


def test_bundles_are_frozen():
    bundle = Bundle("B1", "S1", "train-id", "val-id", {"T1": "t1-id"})

    save_bundle(bundle)

    assert load_bundle("B1") == bundle
    assert list_bundles() == ["B1"]
    with pytest.raises(FileExistsError, match="frozen"):
        save_bundle(bundle)


def test_bundles_stored_before_the_extra_datasets_existed_still_load():
    get_bundle_file("old").write_text(json.dumps({"name": "old", "scope": "S1", "train": "t", "val": "v", "tests": {"T1": "a"}, "notes": {}, "created": "x"}))

    bundle = load_bundle("old")

    assert (bundle.train_extra, bundle.val_extra) == ([], {})
    assert get_evaluation_sets(bundle) == {"val": "v", "T1": "a"}


def fixed_material_dataset(n: int, seed: int, material: np.ndarray, name: str = "part") -> Dataset:
    rng = np.random.default_rng(seed)
    return Dataset(name=name, generator=f"g{seed}", seed=seed, repeats=1, material=material, deposition=random_depositions(n, rng), y=rng.random((n, 2)))


def test_the_training_data_is_the_base_followed_by_the_extra_datasets():
    material = random_materials(1, np.random.default_rng(0))
    base, first, second = (fixed_material_dataset(n, seed, material) for n, seed in ((5, 1), (2, 2), (3, 3)))
    bundle = Bundle("B", "S1", save_dataset(base), "v", {}, train_extra=[save_dataset(first), save_dataset(second)], val_extra={"valop": "o"})
    save_bundle(bundle)

    train = load_training_dataset(load_bundle("B"))

    assert len(train) == 10
    assert train.material.shape == (1, MATERIAL_LENGTH)
    assert np.array_equal(train.deposition, np.vstack([base.deposition, first.deposition, second.deposition]))
    assert np.array_equal(train.y[5:7], first.y)
    assert train.generator == "g1+g2+g3"
    assert list(get_evaluation_sets(bundle)) == ["val", "valop"]


def test_datasets_of_different_materials_are_concatenated_row_by_row():
    rng = np.random.default_rng(0)
    one = fixed_material_dataset(2, 1, random_materials(1, rng))
    other = fixed_material_dataset(3, 2, random_materials(1, rng))

    both = concatenate_datasets("both", [one, other])

    assert both.material.shape == (5, MATERIAL_LENGTH)
    assert np.array_equal(both.full_material()[2], other.material[0])
    assert concatenate_datasets("single", [one]) is one


def test_the_features_of_a_scope():
    rng = np.random.default_rng(0)
    material, deposition = random_materials(1, rng), random_depositions(3, rng)

    assert np.array_equal(make_features("S1", material, deposition), deposition)
    assert make_features("S2", material, deposition).shape == (3, MATERIAL_LENGTH + DEPOSITION_LENGTH)
    with pytest.raises(ValueError, match="scope"):
        make_features("S3", material, deposition)


def write_result(path, variables, **extra):
    path.write_text(json.dumps({"model": "test", "objectives": [[0.0, 0.0]] * len(variables), "variables": variables, **extra}))


def test_front_depositions_come_from_both_result_formats(tmp_path):
    rng = np.random.default_rng(0)
    deposition = random_depositions(3, rng)
    write_result(tmp_path / "run_1.json", deposition.tolist())
    surrogate = np.hstack([random_materials(3, rng), deposition])
    write_result(tmp_path / "run_2.json", surrogate.tolist())
    write_result(tmp_path / "run_2_recomputed_with_lstm.json", surrogate.tolist())

    files = find_result_files(tmp_path)
    loaded, source = load_front_depositions(files)

    assert [file.name for file in files] == ["run_1.json", "run_2.json"]
    assert loaded.shape == (3, DEPOSITION_LENGTH)  # the same solutions in both files count once
    assert set(source["files"]) == {"run_1.json", "run_2.json"}


def test_front_depositions_of_the_wrong_size_are_rejected(tmp_path):
    write_result(tmp_path / "run.json", [[1.0, 2.0]])

    with pytest.raises(ValueError, match="expected 20 or 70"):
        load_front_depositions(find_result_files(tmp_path))


def test_the_material_can_be_taken_from_results(tmp_path):
    rng = np.random.default_rng(0)
    material = random_materials(1, rng)[0]
    write_result(tmp_path / "new.json", random_depositions(2, rng).tolist(), material_variables=material.tolist())

    loaded, source = load_fixed_material(f"results:{tmp_path}", rng)

    assert np.array_equal(loaded, material)
    assert source["file"] == "new.json"


def test_the_material_of_old_surrogate_results_is_part_of_the_variables(tmp_path):
    rng = np.random.default_rng(0)
    material = random_materials(1, rng)
    write_result(tmp_path / "old.json", np.hstack([material, random_depositions(1, rng)]).tolist())

    loaded, _ = load_fixed_material(f"results:{tmp_path}", rng)

    assert np.array_equal(loaded, material[0])


def test_a_random_material_and_unknown_specifications():
    material, _ = load_fixed_material("random", np.random.default_rng(0))

    assert material.shape == (MATERIAL_LENGTH,)
    with pytest.raises(ValueError, match="Unknown material"):
        load_fixed_material("nothing", np.random.default_rng(0))


def test_the_simulation_labels_repeated_inputs_with_mean_and_noise():
    rng = np.random.default_rng(0)
    material = random_materials(1, rng)
    deposition = random_depositions(3, rng)

    y, noise_sd = simulate(material, deposition, repeats=4, n_jobs=1)

    assert y.shape == (3, 2)
    assert noise_sd.shape == (3, 2)
    assert np.all(y > 0)
    assert np.all(noise_sd >= 0)


def test_a_single_repeat_has_no_noise_estimate():
    rng = np.random.default_rng(0)

    _, noise_sd = simulate(random_materials(1, rng), random_depositions(2, rng), repeats=1, n_jobs=1)

    assert noise_sd is None


def test_the_noise_summary_needs_repeats():
    with pytest.raises(ValueError, match="no repeated"):
        get_noise_summary(make_dataset())


def test_the_noise_ceiling_of_labels_without_noise_is_one():
    dataset = make_dataset(n=50, repeats=4)
    dataset.y_noise_sd = np.zeros_like(dataset.y)

    summary = get_noise_summary(dataset)

    assert summary["F1"]["r2_ceiling_labels"] == pytest.approx(1)
    assert summary["F2"]["r2_ceiling_single"] == pytest.approx(1)


def test_the_noise_ceiling_falls_with_the_noise():
    dataset = make_dataset(n=50, repeats=4)
    dataset.y_noise_sd = np.full_like(dataset.y, 0.2)

    summary = get_noise_summary(dataset)

    assert summary["F1"]["noise_sd"] == pytest.approx(0.2)
    assert summary["F1"]["r2_ceiling_single"] < summary["F1"]["r2_ceiling_labels"] < 1


def test_a_dataset_is_built_by_simulation():
    rng = np.random.default_rng(0)

    dataset = build_dataset("built", "random", 0, random_materials(1, rng), random_depositions(2, rng), repeats=2, n_jobs=1)

    assert len(dataset) == 2
    assert dataset.y_noise_sd is not None
    assert dataset.repeats == 2


SMALL = [
    "--train-size",
    "4",
    "--val-size",
    "3",
    "--test-size",
    "3",
    "--train-repeats",
    "1",
    "--val-repeats",
    "2",
    "--test-repeats",
    "2",
    "--n-jobs",
    "1",
    "--stress-random",
    "1",
]


def test_a_bundle_of_the_fixed_material_scope_is_built_and_frozen(tmp_path):
    rng = np.random.default_rng(0)
    write_result(tmp_path / "run.json", random_depositions(2, rng).tolist(), material_variables=random_materials(1, rng)[0].tolist())
    fronts = tmp_path / "fronts"
    fronts.mkdir()
    write_result(fronts / "run.json", random_depositions(3, rng).tolist())

    args = build_bundle.get_args(["--name", "S1-test", "--scope", "S1", "--material-from", f"results:{tmp_path}", "--fronts", str(fronts), *SMALL])
    bundle = build_bundle.build_bundle(args)

    assert set(bundle.tests) == {"T1", "T2", "T5"}
    train = load_dataset(bundle.train)
    assert len(train) == 4
    assert train.material.shape == (1, MATERIAL_LENGTH)
    assert np.array_equal(train.material, load_dataset(bundle.tests["T2"]).material)
    assert load_dataset(bundle.val).repeats == 2
    with pytest.raises(FileExistsError, match="frozen"):
        build_bundle.build_bundle(args)


def test_a_bundle_of_the_general_material_scope_has_unseen_materials():
    args = build_bundle.get_args(["--name", "S2-test", "--scope", "S2", "--t3-materials", "2", "--t3-depositions", "2", *SMALL])

    bundle = build_bundle.build_bundle(args)

    assert set(bundle.tests) == {"T1", "T3", "T5"}
    assert load_dataset(bundle.train).material.shape == (4, MATERIAL_LENGTH)
    t3 = load_dataset(bundle.tests["T3"])
    assert len(t3) == 4
    assert len(np.unique(t3.material, axis=0)) == 2


def test_a_bundle_can_reuse_the_test_sets_of_another_bundle():
    build_bundle.build_bundle(build_bundle.get_args(["--name", "first", "--scope", "S2", "--t3-materials", "1", "--t3-depositions", "2", *SMALL]))

    second = build_bundle.build_bundle(build_bundle.get_args(["--name", "second", "--scope", "S2", "--tests-from", "first", "--seed", "2", *SMALL]))

    first = load_bundle("first")
    assert second.tests == first.tests
    assert second.val_extra == first.val_extra
    assert second.val == first.val
    assert second.train != first.train


def test_a_bundle_cannot_reuse_the_tests_of_another_scope(tmp_path):
    rng = np.random.default_rng(0)
    write_result(tmp_path / "run.json", random_depositions(1, rng).tolist(), material_variables=random_materials(1, rng)[0].tolist())
    build_bundle.build_bundle(build_bundle.get_args(["--name", "one", "--scope", "S1", "--material-from", f"results:{tmp_path}", *SMALL]))

    with pytest.raises(ValueError, match="scope"):
        build_bundle.build_bundle(build_bundle.get_args(["--name", "two", "--scope", "S2", "--tests-from", "one", *SMALL]))
