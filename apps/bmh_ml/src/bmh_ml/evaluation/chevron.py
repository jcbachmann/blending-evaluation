"""Chevron stacking as the reference of the relative objectives: F1 and F2 of a solution divided by those of Chevron for the same material.

Traditionally, F1 is the homogenization efficiency ratio against full-speed Chevron stacking, and 1 means "as good as Chevron". F2 is
divided by Chevron's F2 as well (since 2026-09-24), so the point (1, 1) is Chevron in both objectives and a solution with both relative
objectives below 1 is better than Chevron in both.

The reference Chevron (`CHEVRON_VARIANT`, since simulator release 2026.3) stacks `CHEVRON_LAYERS` layers with the stops at the turning
points that keep the ridge at the height of an ideal stockpile after every layer (`bmh.helpers.chevron.stops_path`), with turning points
that move inwards as the pile grows so that its toe stays in place ("stops-shrinking"); "stops-fixed" turns at the ends of the final core.
Results with point drops (before 2026.3) used "constant-speed": the 20 deposition variables alternating between the ends of the bed, 19
passes without stops, which the optimizer can express but which leaves the ridge uneven (the ends of the pile too low).

The reference is simulated `CHEVRON_REPEATS` times and cached in the store per material, variant and setting, so every run uses the same
values.
"""

import hashlib
import json
from datetime import UTC, datetime

import numpy as np
from bmh.helpers.chevron import stops_path
from bmh.helpers.fronts import get_hypervolume_2d

from bmh_ml.datasets.simulate import simulate
from bmh_ml.parallel import get_cpu_count
from bmh_ml.settings import BED_SIZE_X, BED_SIZE_Z, DEPOSITION_LENGTH, TOTAL_VOLUME, X_MAX, X_MIN, SimulatorSettings, get_simulator_settings
from bmh_ml.simulation import MAX_TIMESTAMP, evaluate_sims_with_profiles
from bmh_ml.tracking.environment import get_code_version
from bmh_ml.tracking.store import get_subdirectory

CHEVRON_REPEATS = 64
CHEVRON_LAYERS = 19
CHEVRON_VARIANT = "stops-shrinking"
CHEVRON_VARIANTS = ("stops-shrinking", "stops-fixed", "constant-speed")


def chevron_deposition() -> np.ndarray:
    """The constant-speed 19-pass Chevron in the 20 deposition variables, shape (1, 20): alternating between the two ends of the bed. The
    reference of results with point drops; new results take `chevron_path` of their settings' variant (`get_default_variant`). The front
    study uses it directly, as the reproduction of results made with point drops (`front_study.common`)."""
    return np.where(np.arange(DEPOSITION_LENGTH) % 2 == 0, X_MIN, X_MAX).astype(float)[None, :]


def chevron_path(variant: str = CHEVRON_VARIANT) -> tuple[np.ndarray, np.ndarray]:
    """Times (0 to `MAX_TIMESTAMP`) and stacker positions of a Chevron variant: its knots, linear in between."""
    if variant == "constant-speed":
        return np.linspace(0, MAX_TIMESTAMP, DEPOSITION_LENGTH), chevron_deposition()[0]
    if variant not in CHEVRON_VARIANTS:
        raise ValueError(f"unknown Chevron variant {variant!r}, one of {CHEVRON_VARIANTS}")
    times, x, _ = stops_path(TOTAL_VOLUME, MAX_TIMESTAMP, X_MIN, X_MAX, CHEVRON_LAYERS, shrinking=variant == "stops-shrinking")
    return times, x


def get_default_variant() -> str:
    """The Chevron of the active simulator settings: settings with point drops (results before 2026.3) keep the constant-speed Chevron."""
    return "constant-speed" if get_simulator_settings().drops == "point" else CHEVRON_VARIANT


def get_reference_key(material: np.ndarray, repeats: int, variant: str | None = None) -> str:
    variant = variant or get_default_variant()
    digest = hashlib.sha256(np.ascontiguousarray(material, dtype=np.float64).tobytes())
    digest.update(json.dumps([BED_SIZE_X, BED_SIZE_Z, TOTAL_VOLUME, DEPOSITION_LENGTH, X_MIN, X_MAX, repeats]).encode())
    simulator = get_simulator_settings()
    unrecorded = SimulatorSettings.from_dict(None)
    if simulator != unrecorded:  # references cached before detail levels existed keep their keys
        # The fields recorded since 2026-10-15 count only where they differ, so that references cached before keep their keys as well
        values = {key: value for key, value in simulator.as_dict().items() if key in ("ppm3", "reclaim_increment") or value != getattr(unrecorded, key)}
        digest.update(json.dumps(values, sort_keys=True).encode())
    if variant != "constant-speed":
        digest.update(json.dumps({"chevron": variant, "layers": CHEVRON_LAYERS}).encode())
    return digest.hexdigest()[:16]


def simulate_chevron(material: np.ndarray, variant: str, repeats: int, n_jobs: int | None = None) -> np.ndarray:
    """F1 and F2 of `repeats` simulations of a Chevron variant for one material, shape (repeats, 2)."""
    times, x = chevron_path(variant)
    if variant == "constant-speed":
        y, _ = simulate(material, np.repeat(x[None, :], repeats, axis=0), 1, n_jobs)  # each row is one simulation
        return y
    y, _ = evaluate_sims_with_profiles(
        np.repeat(material, repeats, axis=0),
        np.repeat(x[None, :], repeats, axis=0),
        BED_SIZE_X,
        BED_SIZE_Z,
        TOTAL_VOLUME,
        threads=n_jobs or get_cpu_count(),
        deposition_times=times,
    )
    return y


def get_chevron_objectives(material: np.ndarray, repeats: int = CHEVRON_REPEATS, n_jobs: int | None = None, variant: str | None = None) -> np.ndarray:
    """F1 and F2 of the reference Chevron for a material, the mean of `repeats` simulations, simulated once and then read from the store.
    `variant` defaults to the Chevron of the active simulator settings (`get_default_variant`)."""
    variant = variant or get_default_variant()
    material = np.atleast_2d(material)[:1]
    path = get_subdirectory("references") / f"chevron-{get_reference_key(material, repeats, variant)}.json"
    if path.exists():
        return np.array(json.loads(path.read_text())["objectives"])
    times, x = chevron_path(variant)
    y = simulate_chevron(material, variant, repeats, n_jobs)
    objectives = y.mean(axis=0)
    record = {
        "objectives": objectives.tolist(),
        "noise_sd": y.std(axis=0, ddof=1).tolist(),
        "repeats": repeats,
        "variant": variant,
        "layers": CHEVRON_LAYERS if variant != "constant-speed" else DEPOSITION_LENGTH - 1,
        "deposition": x.tolist(),
        "deposition_times": times.tolist(),
        "material": material[0].tolist(),
        "settings": {"bed_size_x": BED_SIZE_X, "bed_size_z": BED_SIZE_Z, "total_volume": TOTAL_VOLUME, "simulator": get_simulator_settings().as_dict()},
        "code_version": get_code_version(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(record, indent=2))
    return objectives


def get_chevron_hypervolume(relative: np.ndarray) -> float:
    """Hypervolume of solutions in objectives relative to Chevron, with the reference point (1, 1): only what beats Chevron in both counts."""
    return get_hypervolume_2d(relative, np.ones(2))


def get_chevron_metrics(objectives: np.ndarray, chevron: np.ndarray, prefix: str = "chevron_") -> dict[str, float]:
    """The solutions relative to Chevron: hypervolume at (1, 1), the share that beats Chevron in both objectives (`beaten_rate`) and the
    best relative F1 and F2."""
    relative = objectives / chevron
    return {
        f"{prefix}hv": get_chevron_hypervolume(relative),
        f"{prefix}beaten_rate": float(np.mean(np.all(relative < 1, axis=1))),
        f"{prefix}best_F1": float(relative[:, 0].min()),
        f"{prefix}best_F2": float(relative[:, 1].min()),
    }
