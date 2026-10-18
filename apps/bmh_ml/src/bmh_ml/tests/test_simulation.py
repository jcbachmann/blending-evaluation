import math

import numpy as np
import pandas as pd
import pytest
from bmh.benchmark.material_deposition import Material, MaterialDeposition
from bmh.helpers.math import stdev
from bmh.helpers.reclaimed_material_evaluator import ReclaimedMaterialEvaluator
from bmh.helpers.simple import generate_deposition, generate_material
from bmh.helpers.stockpile_math import get_stockpile_height, get_stockpile_slice_volume

from bmh_ml import simulation
from bmh_ml.settings import SimulatorSettings, simulator_settings
from bmh_ml.simulation import evaluate_sim, evaluate_sim_with_profile, evaluate_sims_with_profiles

BED_SIZE_X = 59.0
BED_SIZE_Z = 20.0
X_MIN = 0.5 * BED_SIZE_Z
X_MAX = BED_SIZE_X - X_MIN


def make_reclaimed_material() -> Material:
    rng = np.random.default_rng(3)
    slices = 60
    return Material.from_data(
        pd.DataFrame(
            {
                "x": np.linspace(0.5, BED_SIZE_X, slices),
                "volume": rng.uniform(20.0, 50.0, slices),
                "quality": rng.uniform(5.0, 10.0, slices),
                "timestamp": np.linspace(0.0, 86400.0, slices),
            }
        )
    )


def reference_f1(reclaimed: Material) -> float:
    quality, volume = reclaimed.data["quality"], reclaimed.data["volume"]
    average = np.average(quality, weights=volume)
    return math.sqrt(np.average((quality - average) ** 2, weights=volume))


def reference_f2(reclaimed: Material) -> float:
    ideal_df = reclaimed.data.copy()
    height = get_stockpile_height(volume=ideal_df["volume"].sum(), core_length=X_MAX - X_MIN)
    ideal_df["x_diff"] = (ideal_df["x"] - ideal_df["x"].shift(1)).fillna(0.0)
    ideal_df["volume"] = ideal_df.apply(
        lambda row: get_stockpile_slice_volume(x=row["x"], core_length=X_MAX - X_MIN, height=height, x_min=X_MIN, x_diff=row["x_diff"]), axis=1
    )
    return stdev((ideal_df["volume"] - reclaimed.data["volume"]).to_numpy())


def test_evaluate_sim_returns_homogenization_and_volume_deviation(monkeypatch):
    # The simulation is stochastic, so it is replaced by a fixed reclaimed material to test the evaluation
    reclaimed = make_reclaimed_material()
    batch = {"x": reclaimed.data["x"].to_numpy(), **{name: reclaimed.data[name].to_numpy()[np.newaxis] for name in ["volume", "quality"]}}
    monkeypatch.setattr(simulation, "simulate_batch", lambda *_args, **_kwargs: batch)

    def evaluate():
        return evaluate_sim(
            material_variables=np.linspace(5, 10, 50),
            deposition_variables=np.linspace(X_MIN, X_MAX, 20),
            bed_size_x=BED_SIZE_X,
            bed_size_z=BED_SIZE_Z,
            total_volume=2500.0,
        )

    with simulator_settings(SimulatorSettings(ideal_stockpile="point")):  # the row-wise reference uses the point rule
        f1, f2 = evaluate()
    assert f1 == pytest.approx(reference_f1(reclaimed), rel=1e-12)
    assert f2 == pytest.approx(reference_f2(reclaimed), rel=1e-12)
    # new settings integrate the ideal stockpile, as bmh's evaluator does by default
    f1, f2 = evaluate()
    expected = ReclaimedMaterialEvaluator(reclaimed=reclaimed, x_min=X_MIN, x_max=X_MAX).get_volume_stdev()
    assert f2 == pytest.approx(expected, rel=1e-12)


def test_evaluate_sim_with_point_drops_stacks_the_rows_of_the_material_deposition(monkeypatch):
    rng = np.random.default_rng(5)
    material_variables = list(rng.uniform(5, 10, 50))
    deposition_variables = list(rng.uniform(X_MIN, X_MAX, 20))
    stacked = []

    def record(data, columns, *_args, **_kwargs):
        stacked.append((data, columns))
        reclaimed = make_reclaimed_material().data
        return {"x": reclaimed["x"].to_numpy(), **{name: reclaimed[name].to_numpy()[np.newaxis] for name in ["volume", "quality"]}}

    monkeypatch.setattr(simulation, "simulate_batch", record)
    with simulator_settings(SimulatorSettings(drops="point")):
        evaluate_sim(material_variables, deposition_variables, bed_size_x=BED_SIZE_X, bed_size_z=BED_SIZE_Z, total_volume=2500.0)

    data, columns = stacked[0][0][0], stacked[0][1]
    expected = MaterialDeposition(
        material=generate_material(material_variables, total_volume=2500.0),
        deposition=generate_deposition(deposition_variables, bed_size_x=BED_SIZE_X, bed_size_z=BED_SIZE_Z),
    ).point_drops()[columns]
    np.testing.assert_array_equal(data, expected.to_numpy())


def test_evaluate_sim_of_the_real_simulation_is_plausible():
    f1, f2 = evaluate_sim(
        material_variables=np.linspace(5, 10, 50),
        deposition_variables=np.linspace(X_MIN, X_MAX, 20),
        bed_size_x=BED_SIZE_X,
        bed_size_z=BED_SIZE_Z,
        total_volume=2500.0,
    )

    # The simulation is stochastic, only the order of magnitude is stable (about 1.4 and 16 in repeated runs)
    assert 0.5 < f1 < 3.0
    assert 5.0 < f2 < 40.0


def test_a_batch_evaluates_like_single_inputs():
    from blending_simulator_lib import simulate_batch

    if "cone" not in (simulate_batch.__doc__ or ""):
        pytest.skip("blending_simulator_lib built without the cone simulator")
    rng = np.random.default_rng(6)
    materials = rng.uniform(5, 10, (3, 50))
    depositions = rng.uniform(X_MIN, X_MAX, (3, 20))
    # The cone simulation is deterministic, so the batch must give exactly the results of single inputs (single runs of the stochastic
    # fast simulation differ by up to about 0.08 in F1 with continuous drops)
    with simulator_settings(SimulatorSettings(simulator="cone")):
        f, profiles = evaluate_sims_with_profiles(materials, depositions, BED_SIZE_X, BED_SIZE_Z, 2500.0, threads=2)
        single = [evaluate_sim_with_profile(materials[i], depositions[i], BED_SIZE_X, BED_SIZE_Z, 2500.0) for i in range(3)]

    assert f.shape == (3, 2)
    assert profiles.shape == (3, 2, 60)
    for i, (f1, f2, profile) in enumerate(single):
        assert f[i, 0] == pytest.approx(f1, rel=1e-12)
        assert f[i, 1] == pytest.approx(f2, rel=1e-12)
        np.testing.assert_allclose(profiles[i], profile, rtol=1e-12)


def test_generated_material_and_deposition_have_the_requested_length():
    material = generate_material(list(np.linspace(5, 10, 50)), total_volume=2500.0)
    deposition = generate_deposition(list(np.linspace(X_MIN, X_MAX, 20)), bed_size_x=BED_SIZE_X, bed_size_z=BED_SIZE_Z)

    assert len(material.data) == 50
    assert material.data["volume"].sum() == pytest.approx(2500.0)
    assert len(deposition.data) == 20


@pytest.mark.parametrize("simulator", ["fast", "lattice", "cone"])
def test_continuous_drops_conserve_the_volume_of_every_simulator(simulator):
    from blending_simulator_lib import simulate_batch

    if simulator == "cone" and "cone" not in (simulate_batch.__doc__ or ""):
        pytest.skip("blending_simulator_lib built without the cone simulator")
    rng = np.random.default_rng(3)
    materials = rng.uniform(5, 10, (2, 50))
    depositions = rng.uniform(10, 49, (2, 20))
    with simulator_settings(SimulatorSettings(ppm3=4.0, simulator=simulator, drops="cells")):
        f, profiles = evaluate_sims_with_profiles(materials, depositions, 59.0, 20.0, 2500.0)
    assert f.shape == (2, 2)
    assert np.all(np.isfinite(f))
    # at most the volume of one particle stays in the simulator's buffer at the end
    np.testing.assert_allclose(profiles[:, 0].sum(axis=1), 2500.0, atol=0.25 + 1e-6)


def test_bmh_stack_reclaim_drops_like_bmh_ml():
    from blending_simulator_lib import simulate_batch

    if "cone" not in (simulate_batch.__doc__ or ""):
        pytest.skip("blending_simulator_lib built without the cone simulator")
    from bmh.simulation.bsl_blending_simulator import BslBlendingSimulator

    rng = np.random.default_rng(8)
    material_variables = list(rng.uniform(5, 10, 50))
    deposition_variables = list(rng.uniform(X_MIN, X_MAX, 20))
    with simulator_settings(SimulatorSettings(simulator="cone")):
        _, _, profile = evaluate_sim_with_profile(material_variables, deposition_variables, BED_SIZE_X, BED_SIZE_Z, 2500.0)
    material_deposition = MaterialDeposition(
        material=generate_material(material_variables, total_volume=2500.0),
        deposition=generate_deposition(deposition_variables, bed_size_x=BED_SIZE_X, bed_size_z=BED_SIZE_Z),
    )
    reclaimed = BslBlendingSimulator(bed_size_x=BED_SIZE_X, bed_size_z=BED_SIZE_Z, cone=True, reclaimincrement=1.0).stack_reclaim(material_deposition)
    volume = reclaimed.data["volume"].to_numpy()
    # the same drops give the same reclaimed volumes; bmh_ml evaluates the slices between x = 0 and the bed length
    np.testing.assert_allclose(volume[volume > 0].sum(), profile[0].sum())
    np.testing.assert_allclose(np.sort(volume[volume > 1e-9]), np.sort(profile[0][profile[0] > 1e-9]), rtol=1e-9)
