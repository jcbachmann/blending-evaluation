import json

import numpy as np
import pytest

from bmh_ml.datasets.generators import random_depositions, random_materials
from bmh_ml.datasets.import_pool import import_results, main
from bmh_ml.datasets.store import load_dataset
from bmh_ml.settings import PROFILE_LENGTH, SIMULATOR_ENVIRONMENT_VARIABLE, SimulatorSettings


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_ML_STORE", str(tmp_path / "store"))
    monkeypatch.delenv(SIMULATOR_ENVIRONMENT_VARIABLE, raising=False)
    return tmp_path / "store"


def write_simulate_job(directory, i, rows, rng, settings: dict, seed=None, image="sim-test"):
    """The files of one `bmh-job simulate` job of the pool: its input and its result."""
    (directory / "inputs").mkdir(exist_ok=True)
    (directory / "results").mkdir(exist_ok=True)
    material, deposition = random_materials(rows, rng), random_depositions(rows, rng)
    np.savez_compressed(directory / "inputs" / f"inputs-{i}.npz", material=material, deposition=deposition, ppm3=np.array(settings["ppm3"]))
    np.savez_compressed(
        directory / "results" / f"results-{i}.npz",
        y=rng.random((rows, 2)),
        y_sd=rng.random((rows, 2)),
        profiles=rng.random((rows, 2, PROFILE_LENGTH)).astype(np.float32),
        ppm3=np.array(settings["ppm3"]),
        seconds=np.array(1.0),
        versions=np.array(json.dumps({"image": image, "simulator": settings["version"], "host": "worker-1"})),
        settings=np.array(json.dumps({**settings, "seed": seed})),
    )
    return material, deposition


def test_simulate_results_become_one_dataset_with_their_inputs_and_settings(tmp_path):
    rng = np.random.default_rng(1)
    settings = SimulatorSettings(ppm3=16.0).as_dict()
    inputs = [write_simulate_job(tmp_path, i, 3, rng, settings, seed=7) for i in (1, 0, 10, 2)]
    results = list((tmp_path / "results").glob("*.npz"))

    dataset = import_results("pool-test", results, tmp_path / "inputs", repeats=4)
    order = [1, 0, 3, 2]  # by file number 0, 1, 2, 10
    assert np.array_equal(dataset.full_material(), np.vstack([inputs[k][0] for k in order]))
    assert np.array_equal(dataset.deposition, np.vstack([inputs[k][1] for k in order]))
    assert dataset.y_noise_sd.shape == (12, 2)
    assert dataset.profiles.shape == (12, 2, PROFILE_LENGTH)
    assert dataset.repeats == 4
    assert SimulatorSettings.from_dict(dataset.settings["simulator"]) == SimulatorSettings(ppm3=16.0)
    assert dataset.source["seed"] == 7
    assert dataset.source["image"] == "sim-test"
    assert dataset.source["files"][:2] == ["inputs-0.npz", "results-0.npz"]

    main(["--name", "pool-test", "--inputs", str(tmp_path / "inputs"), "--repeats", "4", *map(str, results)])
    stored = load_dataset(dataset.dataset_id())
    assert np.array_equal(stored.y, dataset.y)


def test_optimize_results_become_a_dataset_of_the_front(tmp_path):
    rng = np.random.default_rng(2)
    settings = SimulatorSettings(ppm3=1.0, simulator="cone").as_dict()
    material = random_materials(1, rng)
    front = random_depositions(5, rng)
    np.savez_compressed(
        tmp_path / "front-0.npz",
        deposition=front,
        objectives=rng.random((5, 2)),
        material=material,
        seconds=np.array(1.0),
        versions=np.array(json.dumps({"image": "sim-test", "simulator": settings["version"], "host": "worker-1"})),
        settings=np.array(json.dumps({**settings, "seed": None})),
    )
    dataset = import_results("front", [tmp_path / "front-0.npz"])
    assert dataset.generator == "pool-optimize"
    assert np.array_equal(dataset.material, material)
    assert np.array_equal(dataset.deposition, front)
    assert dataset.settings["simulator"]["simulator"] == "cone"


def test_results_of_different_settings_or_of_the_dem_are_refused(tmp_path):
    rng = np.random.default_rng(3)
    write_simulate_job(tmp_path, 0, 2, rng, SimulatorSettings(ppm3=16.0).as_dict())
    write_simulate_job(tmp_path, 1, 2, rng, SimulatorSettings(ppm3=64.0).as_dict())
    with pytest.raises(ValueError, match="different simulator settings"):
        import_results("mixed", list((tmp_path / "results").glob("*.npz")), tmp_path / "inputs")

    dem = tmp_path / "dem"
    dem.mkdir()
    write_simulate_job(dem, 0, 2, rng, {**SimulatorSettings(ppm3=64.0).as_dict(), "simulator": "detailed"})
    with pytest.raises(ValueError, match="detailed simulation"):
        import_results("dem", [dem / "results" / "results-0.npz"], dem / "inputs")
