import numpy as np
import pytest

from bmh_ml import build_bundle
from bmh_ml.datasets.store import get_bundle_simulator, load_bundle, load_dataset, load_manifest
from bmh_ml.settings import SIMULATOR_ENVIRONMENT_VARIABLE, SimulatorSettings

SMALL = ["--scope", "S2", "--seed", "5", "--val-size", "6", "--test-size", "6", "--val-repeats", "2", "--test-repeats", "2", "--n-jobs", "1"]
TESTS = ["--t3-materials", "1", "--t3-depositions", "2", "--stress-random", "1"]
FRONTS = ["--material-fronts", "2", "--front-evaluations", "40", "--front-seeds", "1", "--front-population", "8"]


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_ML_STORE", str(tmp_path / "store"))
    monkeypatch.delenv(SIMULATOR_ENVIRONMENT_VARIABLE, raising=False)
    return tmp_path / "store"


def test_fronts_of_a_cheap_level_become_training_data_at_another_level():
    pytest.importorskip("pymoo")
    build_bundle.build_bundle(build_bundle.get_args(["--name", "cheap", "--train-size", "4", *SMALL, *TESTS, *FRONTS]))
    build_bundle.build_bundle(build_bundle.get_args(["--name", "detailed", "--ppm3", "4", "--train-size", "12", "--profiles", *SMALL, *TESTS]))

    args = ["--name", "detailed-opt", "--ppm3", "4", *SMALL, "--tests-from", "detailed", "--train-from", "detailed", "--extra-from", "cheap:T6", "--profiles"]
    bundle = build_bundle.build_bundle(build_bundle.get_args(args))

    detailed, fronts = load_bundle("detailed"), load_dataset(load_bundle("cheap").tests["T6"])
    assert bundle.train == detailed.train  # the training data is reused, not simulated again
    assert bundle.tests == detailed.tests
    assert len(bundle.train_extra) == 1
    extra = load_dataset(bundle.train_extra[0])
    assert np.array_equal(extra.deposition, fronts.deposition)  # the same solutions, labeled at this level
    assert extra.profiles is not None
    assert extra.repeats == 1
    assert load_manifest(bundle.train_extra[0])["source"]["extra_from"] == "cheap"
    assert get_bundle_simulator(bundle) == SimulatorSettings(ppm3=4.0)
    with pytest.raises(ValueError, match="other simulator settings"):
        build_bundle.build_bundle(build_bundle.get_args(["--name", "mixed", "--ppm3", "1", *SMALL, "--tests-from", "cheap", "--train-from", "detailed"]))
