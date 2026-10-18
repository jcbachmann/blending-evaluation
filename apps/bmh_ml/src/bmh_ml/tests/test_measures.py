import numpy as np
import pytest
from bmh.helpers.chevron import stops_path
from bmh.helpers.stockpile_math import get_ideal_stockpile_volumes

from bmh_ml.evaluation import measures
from bmh_ml.settings import X_MAX, X_MIN, SimulatorSettings, simulator_settings
from bmh_ml.simulation import evaluate_sims_with_profiles, get_objectives, reclaim_paths
from bmh_ml.tests.test_simulation import skip_unless_built_with

DAY = 86400.0


def random_weights(rng, slices=12, portions=6):
    weights = rng.uniform(0, 1, (slices, portions))
    return weights / weights.sum(axis=1, keepdims=True), rng.uniform(1, 3, slices)


def test_the_quality_form_gives_f1_of_the_mixed_quality():
    rng = np.random.default_rng(1)
    weights, volume = random_weights(rng)
    material = rng.uniform(5, 10, 6)
    quality = weights @ material
    mean = np.average(quality, weights=volume)
    expected = np.sqrt(np.average((quality - mean) ** 2, weights=volume))
    form = measures.quality_form(weights, volume)
    assert measures.f1(form, material) == pytest.approx(expected)
    # a constant material is reclaimed without variation
    assert measures.f1(form, np.full(6, 7.0)) == pytest.approx(0.0, abs=1e-6)


def test_without_mixing_the_output_varies_like_the_input():
    # every slice one portion: the homogenization ratio is 1
    material = np.array([5.0, 9.0, 6.0, 8.0])
    form = measures.quality_form(np.eye(4), np.ones(4))
    shares = measures.portion_shares(np.eye(4), np.ones(4))
    np.testing.assert_allclose(shares, 0.25)
    np.testing.assert_allclose(form, measures.input_form(shares))
    assert measures.homogenization_ratio(form, shares, material) == pytest.approx(1.0)
    assert measures.f1(form, material) == pytest.approx(np.std(material))


def test_the_expected_f1_matches_sampled_material_errors():
    rng = np.random.default_rng(2)
    weights, volume = random_weights(rng)
    form = measures.quality_form(weights, volume)
    material = rng.uniform(5, 10, 6)
    covariance = measures.error_covariance(6, 0.5, 0.8)
    errors = rng.multivariate_normal(np.zeros(6), covariance, 200_000)
    sampled = np.mean(measures.quadratic(form, material + errors))
    assert measures.expected_f1(form, material, covariance) ** 2 == pytest.approx(sampled, rel=0.01)
    assert measures.prior_f1(form, covariance) ** 2 == pytest.approx(np.mean(measures.quadratic(form, errors)), rel=0.01)


def test_error_covariance_is_ar1():
    covariance = measures.error_covariance(3, 2.0, 0.5)
    np.testing.assert_allclose(covariance, 4.0 * np.array([[1, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 1]]))
    np.testing.assert_allclose(measures.error_covariance(3, 2.0), 4.0 * np.eye(3))


def test_the_ideal_stockpile_has_no_volume_deviation():
    x = np.arange(1.0, 60.0)
    ideal = get_ideal_stockpile_volumes(x, 2500.0, X_MIN, X_MAX)
    result = measures.volume_measures(x, ideal[None], X_MIN, X_MAX)
    for value in result.values():
        assert value[0] == pytest.approx(0.0, abs=1e-9)
    shifted = ideal.copy()
    core = ideal.max()
    shifted[30] += 0.2 * core
    shifted[31] -= 0.2 * core
    result = measures.volume_measures(x, shifted[None], X_MIN, X_MAX, band=0.1)
    assert result["max_deviation"][0] == pytest.approx(0.2)
    assert result["band_violation"][0] == pytest.approx(2 / len(x))
    assert result["cumulative_deviation"][0] == pytest.approx(0.2 * core / 2500.0)


@pytest.mark.parametrize("simulator", ["cone", "formula"])
def test_the_simulated_mixing_weights_reproduce_the_reclaimed_quality(simulator):
    skip_unless_built_with(simulator)
    rng = np.random.default_rng(3)
    materials = rng.uniform(5, 10, (2, 50))
    times, x, _ = stops_path(2500.0, DAY, X_MIN, X_MAX, 9, shrinking=True)
    with simulator_settings(SimulatorSettings(simulator=simulator)):
        reclaimed = reclaim_paths(materials, [(times, x)] * 2, 59.0, 20.0, 2500.0, mixing=True)
        f, _ = evaluate_sims_with_profiles(materials, x[None].repeat(2, axis=0), 59.0, 20.0, 2500.0, deposition_times=times)
    weights = reclaimed["weights"]
    assert weights.shape == (2, 60, 50)
    np.testing.assert_allclose(np.einsum("isp,ip->is", weights, materials), reclaimed["quality"], atol=1e-9)
    form = measures.quality_form(weights, reclaimed["volume"])
    np.testing.assert_allclose(measures.f1(form, materials), f[:, 0], rtol=1e-9)
    np.testing.assert_allclose(measures.portion_shares(weights, reclaimed["volume"]), 1 / 50, atol=1e-9)


@pytest.mark.parametrize("simulator", ["cone", "formula"])
def test_a_partial_pile_holds_the_stacked_part_of_the_material(simulator):
    skip_unless_built_with(simulator)
    rng = np.random.default_rng(4)
    material = rng.uniform(5, 10, 50)
    times = np.linspace(0, DAY, 20)
    deposition = rng.uniform(X_MIN, X_MAX, 20)
    with simulator_settings(SimulatorSettings(simulator=simulator)):
        full = reclaim_paths(material, [(times, deposition)], 59.0, 20.0, 2500.0)
        half = reclaim_paths(material, [(times, deposition)], 59.0, 20.0, 2500.0, fraction=0.5, mixing=True)
        # the same pile as a path that ends halfway
        cut = np.append(times[times < DAY / 2], DAY / 2)
        ended = reclaim_paths(material, [(cut, np.interp(cut, times, deposition))], 59.0, 20.0, 2500.0)
    assert full["volume"].sum() == pytest.approx(2500.0)
    assert half["volume"].sum() == pytest.approx(1250.0)
    np.testing.assert_allclose(ended["volume"], half["volume"], atol=1e-9)
    shares = measures.portion_shares(half["weights"], half["volume"])[0]
    np.testing.assert_allclose(shares[:25], 1 / 25, atol=1e-9)
    np.testing.assert_allclose(shares[25:], 0.0, atol=1e-9)
    objectives = get_objectives(half["x"], half["volume"], half["quality"], X_MIN, X_MAX)
    assert np.all(np.isfinite(objectives))
