"""
Stacker schedules for chevron stacking that keep the stockpile an ideal stockpile (constant ridge height) after every layer.

Geometry (see the Entropy notes "Ideal Stockpile" and "Chevron Stacking Schedule"): an ideal stockpile with core length L, radius r and
angle of repose theta has the height h = r tan(theta) and the volume tan(theta) (r^2 L + pi/3 r^3). Layer k of Z turns the ideal stockpile
after layer k - 1 (radius r_{k-1}) into the one after layer k (radius r_k) holding k V / Z.

While the stacker travels along the ridge, the material ahead of it forms a half-cone that moves along (the front). On the core it adds
the cross section tan(theta) (r_k^2 - r_{k-1}^2) per meter. At the start of a layer the stacker stops until the cone around it reaches the
new height: the end shell behind it, D_k, and the front above the old ridge, tan(theta) W(r_k, r_{k-1}). Arriving at the other end, the
front only holds W but the end needs D_k, so the stacker stops for the difference before the next layer starts there.

The turning points either stay at the ends of the final core (fixed limits) or move inwards by delta_k = r_k - r_{k-1} per layer, so that
the toe of the stockpile stays in place (shrinking limits). In both cases the end shell is the new end half-cone minus the old stockpile
inside it, D_k = tan(theta) (pi/6 (r_k^3 - r_{k-1}^3) - r_{k-1}^2 delta_k).
"""

import math

import numpy as np
from scipy.optimize import brentq


def front_volume(r: float, s: float) -> float:
    """
    Volume above a ridge prism of half width s inside a half-cone of radius r standing on the ridge, both with a 45° slope.

    The half-cone is the front of a stacker at height r moving along a ridge of height s. For other angles of repose multiply by
    tan(theta), with r and s as horizontal radii. W(r, 0) = pi/6 r^3 (plain half-cone) and W(r, r) = 0.
    """
    if s <= 0.0:
        return math.pi / 6.0 * r**3
    if s >= r:
        return 0.0
    root = math.sqrt(r * r - s * s)
    return math.pi / 6.0 * r**3 - r**3 / 3.0 * math.asin(s / r) + root * (2.0 / 9.0 * r * r - r * s / 3.0 - 2.0 / 9.0 * s * s) - 2.0 / 9.0 * (r - s) ** 3


def ideal_radius(volume: float, core_length: float, tan_theta: float = 1.0) -> float:
    """Radius of the ideal stockpile with the given volume and core length."""
    if volume <= 0.0:
        return 0.0
    return brentq(lambda r: tan_theta * (r * r * core_length + math.pi / 3.0 * r**3) - volume, 0.0, 1e4)


def schedule(volume: float, x_min: float, x_max: float, layers: int, shrinking: bool = False, tan_theta: float = 1.0) -> list[dict]:
    """
    Turning points and volumes of every layer: stacked while standing at the start (start) and at the end (end) and per meter of travel.

    x_min and x_max are the ends of the core of the final stockpile. Layer 0 starts at the low end, consecutive layers alternate the
    direction; the stop between layers k and k + 1 is end of k plus start of k + 1, at the turning point of k + 1.
    """
    final_radius = ideal_radius(volume, x_max - x_min, tan_theta)
    toe_min, toe_max = x_min - final_radius, x_max + final_radius
    span = toe_max - toe_min
    layer_list = []
    previous = 0.0
    for k in range(1, layers + 1):
        target = volume * k / layers
        if shrinking:
            # The core shrinks with the radius so that the toe stays in place: core length = toe span - 2 r
            radius = brentq(lambda r, v=target: tan_theta * (r * r * (span - 2.0 * r) + math.pi / 3.0 * r**3) - v, 0.0, final_radius + 1e-9)
            low, high = toe_min + radius, toe_max - radius
            shift = radius - previous
        else:
            radius = ideal_radius(target, x_max - x_min, tan_theta)
            low, high = x_min, x_max
            shift = 0.0
        end_shell = tan_theta * (math.pi / 6.0 * (radius**3 - previous**3) - previous**2 * shift)
        front = tan_theta * front_volume(radius, previous)
        layer_list.append(
            {
                "radius": radius,
                "low": low,
                "high": high,
                "start": end_shell + front,
                "end": end_shell - front,
                "per_meter": tan_theta * (radius**2 - previous**2),
            }
        )
        previous = radius
    return layer_list


def stops_path(volume: float, duration: float, x_min: float, x_max: float, layers: int, shrinking: bool = False, tan_theta: float = 1.0):
    """
    Stacker path (timestamps, positions) and layer end times at a constant material flow, with the stops of schedule(). With shrinking
    limits the stacker moves to the next turning point within one second between two layers.
    """
    seconds_per_volume = duration / volume
    t = 0.0
    timestamps, positions, layer_ends = [], [], []
    for k, layer in enumerate(schedule(volume, x_min, x_max, layers, shrinking, tan_theta)):
        here, there = (layer["low"], layer["high"]) if k % 2 == 0 else (layer["high"], layer["low"])
        if positions and positions[-1] != here:
            t += 1.0
        timestamps.append(t)
        positions.append(here)
        t += layer["start"] * seconds_per_volume
        timestamps.append(t)
        positions.append(here)
        t += layer["per_meter"] * abs(there - here) * seconds_per_volume
        timestamps.append(t)
        positions.append(there)
        t += layer["end"] * seconds_per_volume
        timestamps.append(t)
        positions.append(there)
        layer_ends.append(t)
    # The moves between turning points take a few seconds in total, scale them away so that the path ends with the material flow
    timestamps = np.array(timestamps) * duration / t
    return timestamps, np.array(positions), [e * duration / t for e in layer_ends]
