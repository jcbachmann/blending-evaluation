import numpy as np
import pytest

from bmh_ml import build_bundle
from bmh_ml.datasets.generators import random_depositions, random_materials
from bmh_ml.datasets.simulate import build_dataset
from bmh_ml.datasets.store import Bundle, get_bundle_simulator, load_bundle, load_dataset, save_dataset
from bmh_ml.settings import SIMULATOR_ENVIRONMENT_VARIABLE, SimulatorSettings, get_simulator_settings, use_simulator_settings

SMALL = ["--train-size", "20", "--val-size", "6", "--test-size", "6", "--val-repeats", "2", "--test-repeats", "2", "--n-jobs", "1", "--stress-random", "1"]
S2_SMALL = ["--scope", "S2", "--t3-materials", "1", "--t3-depositions", "2", *SMALL]


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("BMH_ML_STORE", str(tmp_path / "store"))
    monkeypatch.delenv(SIMULATOR_ENVIRONMENT_VARIABLE, raising=False)
    return tmp_path / "store"


def test_the_default_level_is_the_simulator_of_all_earlier_data():
    assert get_simulator_settings() == SimulatorSettings(ppm3=1.0, reclaim_increment=1.0)  # the simulator's own step 1/sqrt(1) is 1 as well
    assert SimulatorSettings.from_dict(None) == SimulatorSettings()


def test_the_active_level_reaches_the_simulator(monkeypatch):
    from bmh_ml import simulation

    created = []
    original = simulation.BslBlendingSimulator

    def recording(*args, **kwargs):
        created.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(simulation, "BslBlendingSimulator", recording)
    rng = np.random.default_rng(0)
    material, deposition = random_materials(1, rng)[0], random_depositions(1, rng)[0]
    use_simulator_settings(SimulatorSettings(ppm3=4.0))

    simulation.evaluate_sim(material, deposition, 59, 20, 2500)

    assert created[-1]["ppm3"] == 4.0
    assert created[-1]["reclaimincrement"] == 1.0  # fixed, so every level reclaims the same 60 slices


def get_level_in_worker(_: int) -> float:
    return get_simulator_settings().ppm3


def test_worker_processes_simulate_at_the_level_of_the_process_that_starts_them():
    from bmh_ml.parallel import run_parallel

    use_simulator_settings(SimulatorSettings(ppm3=16.0))

    assert run_parallel(get_level_in_worker, [(1,), (2,)], workers=2) == [16.0, 16.0]


def test_a_dataset_records_the_level_of_its_labels():
    rng = np.random.default_rng(1)
    use_simulator_settings(SimulatorSettings(ppm3=4.0))

    dataset = build_dataset("d", "random", 0, random_materials(3, rng), random_depositions(3, rng), n_jobs=1, settings={"scope": "S2"})

    assert dataset.settings == {"scope": "S2", "simulator": {"ppm3": 4.0, "reclaim_increment": 1.0}}
    assert load_dataset(save_dataset(dataset)).settings["simulator"]["ppm3"] == 4.0


def test_a_bundle_has_one_level():
    rng = np.random.default_rng(2)
    ids = []
    for ppm3 in (1.0, 4.0):
        use_simulator_settings(SimulatorSettings(ppm3=ppm3))
        ids.append(save_dataset(build_dataset(f"d{ppm3}", "random", 0, random_materials(2, rng), random_depositions(2, rng), n_jobs=1)))

    assert get_bundle_simulator(Bundle("same", "S2", ids[1], ids[1], {})) == SimulatorSettings(ppm3=4.0)
    with pytest.raises(ValueError, match="mixes simulator settings"):
        get_bundle_simulator(Bundle("mixed", "S2", ids[0], ids[1], {}))


def test_a_bundle_is_built_at_a_level_and_can_label_the_test_sets_of_another_level():
    build_bundle.build_bundle(build_bundle.get_args(["--name", "low", *S2_SMALL]))
    high = build_bundle.build_bundle(build_bundle.get_args(["--name", "high", "--ppm3", "4", *S2_SMALL, "--cross-from", "low", "--cross-sets", "T1", "T3"]))

    low = load_bundle("low")
    assert get_bundle_simulator(low) == SimulatorSettings()
    assert get_bundle_simulator(high) == SimulatorSettings(ppm3=4.0)
    same_inputs = load_dataset(low.tests["T1"]), load_dataset(high.tests["T1x"])
    assert np.array_equal(same_inputs[0].deposition, same_inputs[1].deposition)  # the same solutions, labeled at the other level
    assert not np.allclose(same_inputs[0].y, same_inputs[1].y)
    assert same_inputs[1].source["cross_from"] == "low"
    assert np.array_equal(load_dataset(low.tests["T1"]).deposition, load_dataset(high.tests["T1"]).deposition)  # same seed, same inputs
    with pytest.raises(ValueError, match="other simulator settings"):
        build_bundle.build_bundle(build_bundle.get_args(["--name", "mixed", "--ppm3", "4", "--tests-from", "low", *S2_SMALL]))


def test_the_chevron_reference_is_cached_per_level():
    from bmh_ml.evaluation.chevron import get_reference_key

    material = random_materials(1, np.random.default_rng(3))
    default_key = get_reference_key(material, 64)
    use_simulator_settings(SimulatorSettings(ppm3=64.0))

    assert get_reference_key(material, 64) != default_key
    use_simulator_settings(SimulatorSettings())
    assert get_reference_key(material, 64) == default_key  # references cached before the levels existed are still found
