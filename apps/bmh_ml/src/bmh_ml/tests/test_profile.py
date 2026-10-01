import numpy as np
import pytest

from bmh_ml import build_bundle
from bmh_ml.datasets.generators import random_depositions, random_materials
from bmh_ml.datasets.manifest import Dataset, concatenate_datasets
from bmh_ml.datasets.simulate import build_dataset, simulate_with_profiles
from bmh_ml.datasets.store import load_bundle, load_dataset, save_dataset
from bmh_ml.evaluation.profile import get_ideal_volumes, get_profile_objectives
from bmh_ml.models.registry import create_model, load_model
from bmh_ml.settings import PROFILE_LENGTH, TOTAL_VOLUME

SMALL = ["--train-size", "40", "--val-size", "10", "--test-size", "10", "--val-repeats", "2", "--test-repeats", "2", "--n-jobs", "1", "--stress-random", "1"]


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_ML_STORE", str(tmp_path / "store"))
    return tmp_path / "store"


@pytest.fixture(scope="module")
def simulated():
    rng = np.random.default_rng(0)
    material, deposition = random_materials(1, rng), random_depositions(30, rng)
    return material, deposition, *simulate_with_profiles(material, deposition, repeats=1, n_jobs=1)


def test_the_objectives_computed_from_a_profile_are_those_of_the_simulator(simulated):
    _, _, y, _, profiles = simulated

    assert profiles.shape == (30, 2, PROFILE_LENGTH)
    assert np.allclose(get_profile_objectives(profiles), y, rtol=1e-4, atol=1e-4)


def test_the_profile_of_repeated_simulations_is_their_mean(simulated):
    material, deposition, *_ = simulated

    y, noise_sd, profiles = simulate_with_profiles(material, deposition[:3], repeats=4, n_jobs=1)

    assert noise_sd is not None
    assert np.allclose(profiles[:, 0].sum(axis=1), TOTAL_VOLUME, rtol=0.02)  # the reclaimed volume is the stacked volume
    # F2 is a norm of (almost) a linear function of the volumes, so the F2 of the mean profile is at most the mean F2 (Jensen); F1 weights
    # the quality with varying volumes and has no such bound
    assert np.all(get_profile_objectives(profiles)[:, 1] <= y[:, 1] * (1 + 1e-3))


def test_the_interpolated_ideal_volumes_match_the_exact_ones():
    from bmh.helpers.stockpile_math import get_ideal_stockpile_volumes

    from bmh_ml.settings import X_MAX, X_MIN

    totals = np.array([2381.7, 2537.3, 2690.1])
    exact = np.stack([get_ideal_stockpile_volumes(np.arange(PROFILE_LENGTH, dtype=float), total, X_MIN, X_MAX) for total in totals])

    assert np.allclose(get_ideal_volumes(totals), exact, atol=1e-3)
    assert np.isclose(exact[1].sum(), 2537.3, rtol=0.01)


def test_profiles_are_stored_with_the_dataset_and_in_its_hash(simulated):
    material, deposition, y, _, profiles = simulated
    with_profiles = Dataset("p", "random", 0, 1, material, deposition, y, profiles=profiles)
    without = Dataset("p", "random", 0, 1, material, deposition, y)

    loaded = load_dataset(save_dataset(with_profiles))

    assert loaded.profiles.dtype == np.float16
    assert np.array_equal(loaded.profiles, with_profiles.profiles)
    assert with_profiles.content_hash() != without.content_hash()
    assert concatenate_datasets("both", [with_profiles, with_profiles]).profiles.shape == (60, 2, PROFILE_LENGTH)
    assert concatenate_datasets("mixed", [with_profiles, without]).profiles is None
    with pytest.raises(ValueError, match="profiles"):
        Dataset("p", "random", 0, 1, material, deposition, y, profiles=profiles[:, :, :10])


def test_a_bundle_with_profiles_and_a_new_validation_set_keeps_the_test_sets():
    build_bundle.build_bundle(build_bundle.get_args(["--name", "first", "--scope", "S2", "--t3-materials", "1", "--t3-depositions", "2", *SMALL]))

    args = ["--name", "second", "--scope", "S2", "--tests-from", "first", "--new-val", "--profiles", *SMALL]
    second = build_bundle.build_bundle(build_bundle.get_args(args))

    first = load_bundle("first")
    assert second.tests == first.tests
    assert second.val != first.val
    assert load_dataset(second.train).profiles is not None
    assert load_dataset(second.val).profiles is not None
    assert load_dataset(first.train).profiles is None


def test_the_profile_model_learns_the_profiles_and_survives_saving(tmp_path):
    pytest.importorskip("keras")
    pytest.importorskip("tensorflow")
    rng = np.random.default_rng(1)
    material = random_materials(1, rng)
    train = build_dataset("t", "random", 0, material, random_depositions(300, rng), n_jobs=1, with_profiles=True)
    val = build_dataset("v", "random", 0, material, random_depositions(60, rng), repeats=4, n_jobs=1, with_profiles=True)  # repeats: noise correction
    model = create_model("profile_mlp", width=64, depth=2, epochs=100, patience=10, batch_size=64, objective_weight=0.0)  # the profile path alone

    with pytest.raises(ValueError, match="profiles"):
        model.fit(train.deposition, train.y, val.deposition, val.y, seed=0)
    info = model.fit(train.deposition, train.y, val.deposition, val.y, seed=0, profiles_train=train.profiles, profiles_val=val.profiles, val_repeats=4)

    prediction = model.predict(val.deposition)
    assert prediction.shape == (60, 2)
    assert np.all(prediction >= 0)
    assert info["F2/noise_correction"] > 0
    assert model.predict_profiles(val.deposition[:5]).shape == (5, 2, PROFILE_LENGTH)
    r2 = 1 - np.sum((val.y[:, 1] - prediction[:, 1]) ** 2) / np.sum((val.y[:, 1] - val.y[:, 1].mean()) ** 2)
    assert r2 > 0.15  # F2 is learned from 300 profiles (predicting the mean scores 0; training is not bit-reproducible, about 0.25 to 0.35)
    model.save(tmp_path / "model")
    assert np.allclose(load_model(tmp_path / "model").predict(val.deposition), prediction, atol=1e-4)


def test_training_a_profile_model_needs_a_bundle_with_profiles():
    pytest.importorskip("keras")
    pytest.importorskip("mlflow")
    from bmh_ml.train import run_training

    build_bundle.build_bundle(build_bundle.get_args(["--name", "plain", "--scope", "S2", "--t3-materials", "1", "--t3-depositions", "2", *SMALL]))
    build_bundle.build_bundle(build_bundle.get_args(["--name", "rich", "--scope", "S2", "--tests-from", "plain", "--new-val", "--profiles", *SMALL]))
    params = {"width": 16, "depth": 1, "epochs": 2}

    with pytest.raises(ValueError, match="--profiles"):
        run_training("plain", "profile_mlp", params, with_plots=False)
    result = run_training("rich", "profile_mlp", params, with_plots=False)

    assert "T1/F2/nrmse" in result.metrics
    assert "train/F2/noise_correction" in result.metrics


def test_the_noise_correction_comes_from_repeated_data_alone(simulated):
    pytest.importorskip("keras")
    from bmh_ml.models.keras_models import get_noise_correction

    material, deposition, *_ = simulated
    y, _, profiles = simulate_with_profiles(material, deposition[:20], repeats=8, n_jobs=1)

    correction = get_noise_correction(profiles, y, repeats=8)

    assert correction.shape == (2,)
    assert 0.1 < correction[1] < 5  # the slice noise adds about 1 to F2 squared (measured on S1-v2: 1.12)
    assert np.array_equal(get_noise_correction(profiles, y, repeats=1), np.zeros(2))


def test_each_objective_can_come_from_the_profile_or_the_direct_output():
    pytest.importorskip("keras")
    with pytest.raises(ValueError, match="f1_source"):
        create_model("profile_mlp", f1_source="guess")
    rng = np.random.default_rng(2)
    material = random_materials(1, rng)
    train = build_dataset("t", "random", 0, material, random_depositions(200, rng), n_jobs=1, with_profiles=True)
    val = build_dataset("v", "random", 0, material, random_depositions(40, rng), repeats=2, n_jobs=1, with_profiles=True)
    arguments = {"profiles_train": train.profiles, "profiles_val": val.profiles, "val_repeats": 2}
    predictions = {}
    for sources in (("head", "profile"), ("profile", "head")):
        model = create_model("profile_mlp", width=32, depth=1, epochs=5, f1_source=sources[0], f2_source=sources[1])
        model.fit(train.deposition, train.y, val.deposition, val.y, seed=0, **arguments)
        predictions[sources] = model.predict(val.deposition)

    first, second = predictions.values()
    assert not np.allclose(first[:, 0], second[:, 0])
    assert np.all(first >= 0)
    assert np.all(second >= 0)
