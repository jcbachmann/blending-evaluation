"""Measures of a stockpile beyond F1 and F2: F1 against the material itself, under imperfect knowledge of the material, and how close
the reclaimed volume curve stays to that of the ideal stockpile.

**Quality.** Where the simulation mixes linearly (the cone, the formula and the detailed simulation per seed), the reclaimed quality is
q = W m for the mixing weights W of the deposition (`bmh_ml.simulation.reclaim_paths` with `mixing`), so F1² is a quadratic form of the
material, F1² = m' B m (`quality_form`). The input variation is the same kind of form of the stacked material, m' C m with
C = diag(a) - a a' for the shares a of the portions in the pile (`input_form`): a pile without any mixing would reclaim it unchanged.
Their ratio F1 / sd(input) is the homogenization ratio (lower is better; the classic variance reduction ratio is its inverse squared),
measured against the material itself instead of against a reference stacking. If the material is known only up to an error of zero mean
and covariance S, the expected F1² of the true material is m' B m + trace(B S) (`expected_f1`); levels of knowledge are levels of S
(`error_covariance`). Without any knowledge of the material, a material process with covariance R gives the deposition-only measures
trace(B R) and the expected homogenization ratio sqrt(trace(B R) / trace(C R)) (`prior_f1`, `prior_homogenization_ratio`).

**Volume.** `volume_measures` compares the reclaimed slice volumes with those of the ideal stockpile of the same volume (the rule of
`bmh.helpers.stockpile_math.get_ideal_stockpile_volumes`): F2 as bmh_ml computes it, the largest deviation of a slice and the share of
slices outside a band, both relative to the ideal core slice, and the largest deviation of the cumulative volume curve.
"""

import numpy as np
from bmh.helpers.stockpile_math import get_ideal_stockpile_volumes


def quality_form(weights: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """B with F1² = m' B m for the mixing weights W (shape (..., slices, portions)) and the reclaimed volumes (shape (..., slices)):
    W' diag(v) W / V - a a' with the shares a = W' v / V of the portions in the reclaimed material."""
    weights, volume = np.asarray(weights, dtype=float), np.asarray(volume, dtype=float)
    total = volume.sum(axis=-1)[..., None, None]
    second = np.einsum("...sp,...s,...sq->...pq", weights, volume, weights) / total
    shares = portion_shares(weights, volume)
    return second - shares[..., :, None] * shares[..., None, :]


def portion_shares(weights: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """The share of each material portion in the reclaimed material, shape (..., portions); 1 / n each for a complete pile of equal
    portions, the stacked part for a partial pile."""
    weights, volume = np.asarray(weights, dtype=float), np.asarray(volume, dtype=float)
    return np.einsum("...sp,...s->...p", weights, volume) / volume.sum(axis=-1)[..., None]


def input_form(shares: np.ndarray) -> np.ndarray:
    """C with sd(input)² = m' C m: the volume-weighted variance of the portions with the given shares, diag(a) - a a'."""
    shares = np.asarray(shares, dtype=float)
    return np.einsum("...p,pq->...pq", shares, np.eye(shares.shape[-1])) - shares[..., :, None] * shares[..., None, :]


def quadratic(form: np.ndarray, material: np.ndarray) -> np.ndarray:
    """m' A m for forms of shape (..., n, n) and materials of shape (..., n) (broadcast), clipped at 0 against rounding."""
    material = np.asarray(material, dtype=float)
    return np.maximum(np.einsum("...p,...pq,...q->...", material, form, material), 0.0)


def f1(form: np.ndarray, material: np.ndarray) -> np.ndarray:
    """F1 of materials from the quality form B of their depositions."""
    return np.sqrt(quadratic(form, material))


def homogenization_ratio(form: np.ndarray, shares: np.ndarray, material: np.ndarray) -> np.ndarray:
    """F1 / sd(input): the reclaimed quality's standard deviation over that of the stacked material (lower is better)."""
    return np.sqrt(quadratic(form, material) / quadratic(input_form(shares), material))


def error_covariance(portions: int, sd: float, correlation: float = 0.0) -> np.ndarray:
    """Covariance of a material error with standard deviation `sd` per portion and the correlation `correlation` ** |i - j| between
    portions i and j (an AR(1) process: 0 for independent errors, close to 1 for errors that drift slowly, as forecasts do)."""
    lags = np.abs(np.subtract.outer(np.arange(portions), np.arange(portions)))
    return sd**2 * float(correlation) ** lags if correlation else sd**2 * np.eye(portions)


def expected_f1(form: np.ndarray, material: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """The root of the expected F1² of the true material when `material` is its prediction with an error of zero mean and the given
    covariance: sqrt(m' B m + trace(B S))."""
    return np.sqrt(quadratic(form, material) + np.maximum(np.einsum("...pq,qp->...", form, covariance), 0.0))


def prior_f1(form: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """The root of the expected F1² for a material process of zero mean and the given covariance, a measure of the deposition alone:
    sqrt(trace(B R)) (with R the identity, F1 of white noise of unit variance)."""
    return np.sqrt(np.maximum(np.einsum("...pq,qp->...", form, covariance), 0.0))


def prior_homogenization_ratio(form: np.ndarray, shares: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """sqrt(E F1² / E sd(input)²) for a material process with the given covariance: sqrt(trace(B R) / trace(C R))."""
    stacked = np.einsum("...pq,qp->...", input_form(shares), covariance)
    return np.sqrt(np.maximum(np.einsum("...pq,qp->...", form, covariance), 0.0) / stacked)


def volume_measures(x: np.ndarray, volume: np.ndarray, x_min: float, x_max: float, band: float = 0.1, rule: str = "integrated") -> dict[str, np.ndarray]:
    """How far reclaimed slice volumes (shape (inputs, slices), at the reclaimer positions x) are from those of the ideal stockpile of
    the same volume with the core from x_min to x_max, per input:

    * `F2`: the standard deviation of the deviations, as bmh_ml computes F2 (`bmh_ml.simulation.get_objectives`);
    * `max_deviation`: the largest deviation of a slice, relative to the ideal core slice (the largest ideal slice);
    * `band_violation`: the share of slices whose deviation exceeds `band` times the ideal core slice;
    * `cumulative_deviation`: the largest deviation of the cumulative reclaimed volume from the ideal one, relative to the volume (how far
      the reclaimer runs ahead of or behind the ideal flow)."""
    volume = np.atleast_2d(np.asarray(volume, dtype=float))
    total = volume.sum(axis=1, keepdims=True)
    ideal = get_ideal_stockpile_volumes(np.asarray(x, dtype=float), total, x_min, x_max, rule=rule)
    deviation = volume - ideal
    core = ideal.max(axis=1, keepdims=True)
    return {
        "F2": np.sqrt(((deviation - deviation.mean(axis=1, keepdims=True)) ** 2).mean(axis=1)),
        "max_deviation": (np.abs(deviation) / core).max(axis=1),
        "band_violation": (np.abs(deviation) > band * core).mean(axis=1),
        "cumulative_deviation": np.abs(np.cumsum(deviation, axis=1)).max(axis=1) / total[:, 0],
    }
