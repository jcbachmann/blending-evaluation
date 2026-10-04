"""
Deposition patterns, resolutions and simulation methods of the simulator comparison.

All runs use the same bed: 60 m long and 20 m deep, so that an ideal stockpile with a 45° angle of repose has a radius and height of 10 m.
Chevron patterns fill the bed with exactly that ideal stockpile (core from x = 10 m to 50 m), the cone pattern builds the cone of radius 10 m
in the middle of the bed. The material quality rises linearly from 0 to 1 over the deposition time, so colors show the deposition order.

Chevron stacking follows the schedule of chevron.py: the stacker stops at the turning points so that the stockpile is an ideal stockpile
after every layer (for one layer just a start cone), at a constant material flow. The turning points either stay at the ends of the core
or move inwards as the stockpile grows.
"""

import numpy as np
import pandas as pd
from bmh.benchmark.material_deposition import Deposition, Material, MaterialDeposition
from bmh.helpers.stockpile_math import get_stockpile_height, get_stockpile_volume

from .chevron import stops_path

BED_SIZE_X = 60.0
BED_SIZE_Z = 20.0
RADIUS = 0.5 * BED_SIZE_Z
X_MIN = RADIUS
X_MAX = BED_SIZE_X - RADIUS
# Material increments: at least STEPS, and enough for the stacker to cover a pass in about 80 increments
STEPS = 4000
STEPS_PER_LAYER = 80
DURATION = 24 * 60 * 60.0

CHEVRON_LAYERS = (1, 5, 25, 100, 500)

BASE_PATTERNS = {"cone": "1 cone", **{f"chevron-{n}": f"{n} layer chevron" for n in CHEVRON_LAYERS}}

# Turning points of chevrons with several layers, "" stays at the ends of the core
END_VARIANTS = {
    "": "Fixed turning points",
    "limits": "Shrinking turning points",
}

PATTERNS = {"cone": "1 cone", "chevron-1": "1 layer chevron"}
for _layers in CHEVRON_LAYERS[1:]:
    for _variant, _label in END_VARIANTS.items():
        PATTERNS[f"chevron-{_layers}" + (f"-{_variant}" if _variant else "")] = f"{_layers} layer chevron, {_label.lower()}"


def parse_pattern(pattern: str) -> tuple[str, int, str]:
    """Kind ("cone" or "chevron"), number of layers and end variant of a pattern identifier."""
    if pattern == "cone":
        return "cone", 0, ""
    parts = pattern.split("-")
    if parts[0] != "chevron" or len(parts) not in (2, 3):
        raise ValueError(f"unknown pattern {pattern}")
    variant = parts[2] if len(parts) == 3 else ""
    if variant not in END_VARIANTS:
        raise ValueError(f"unknown end variant {variant}")
    return "chevron", int(parts[1]), variant


# Particles per cubic meter: particle sizes of 2 m, 1 m, 0.5 m and 0.25 m, which are 1/5 to 1/40 of the pile height
RESOLUTIONS = {
    "low": 0.125,
    "mid": 1.0,
    "high": 8.0,
    "very-high": 64.0,
}

METHODS = {
    "mathematical": "Mathematical (closed form binning, bmh)",
    "smooth": "Smooth (Gaussian kernel, bmh)",
    "fast": "Fast (C++ height grid, blending_simulator_lib)",
    "hcp": "HCP lattice (C++ port of hexsim compressed to 45°, blending_simulator_lib)",
    "detailed": "Detailed (C++ Bullet physics, blending_simulator_lib)",
}


def get_pattern_volume(pattern: str) -> float:
    if pattern == "cone":
        return get_stockpile_volume(RADIUS, 0.0)
    return get_stockpile_volume(RADIUS, X_MAX - X_MIN)


def get_start_cone_volume(volume: float, layers: int) -> float:
    """Volume of the cone at the start of chevron stacking, which reaches the height of the first layer on the core."""
    first_layer_height = float(get_stockpile_height(volume / layers, X_MAX - X_MIN))
    return get_stockpile_volume(first_layer_height, 0.0)


def get_steps(pattern: str) -> int:
    return max(STEPS, STEPS_PER_LAYER * parse_pattern(pattern)[1])


def make_deposition_path(pattern: str, volume: float) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Stacker path as timestamps and x positions, and the times at which each layer is complete."""
    kind, layers, variant = parse_pattern(pattern)
    if kind == "cone":
        return np.array([0.0, DURATION]), np.full(2, 0.5 * BED_SIZE_X), []
    return stops_path(volume, DURATION, X_MIN, X_MAX, layers, shrinking=variant == "limits")


def make_material_deposition(pattern: str) -> MaterialDeposition:
    steps = get_steps(pattern)
    timestamps = np.linspace(0.0, DURATION, steps)
    volume = get_pattern_volume(pattern)
    material = Material.from_data(
        pd.DataFrame({"timestamp": timestamps, "volume": np.full(steps, volume / steps), "quality": timestamps / DURATION}),
        identifier=pattern,
    )
    deposition_timestamps, x, _ = make_deposition_path(pattern, volume)
    deposition = Deposition.from_data(
        pd.DataFrame({"timestamp": deposition_timestamps, "x": x, "z": np.full(len(x), RADIUS)}),
        identifier=pattern,
        bed_size_x=BED_SIZE_X,
        bed_size_z=BED_SIZE_Z,
        reclaim_x_per_s=BED_SIZE_X / DURATION,
    )
    return MaterialDeposition(material, deposition)


def get_layer_end_times(pattern: str) -> list[float]:
    """Times at which each layer of a chevron pattern is complete."""
    return make_deposition_path(pattern, get_pattern_volume(pattern))[2]
