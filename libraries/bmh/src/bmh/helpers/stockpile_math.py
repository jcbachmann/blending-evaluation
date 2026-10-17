from math import pi, sqrt

import numpy as np


def get_stockpile_volume(height, core_length):
    """
    Compute the volume of a stockpile given the height and core length
    :param height: stockpile height
    :param core_length: stockpile core length
    :return: stockpile volume
    """
    return pow(height, 2.0) * core_length + pi / 3.0 * pow(height, 3.0)


def get_stockpile_height(volume, core_length):
    """
    Compute the height of a stockpile given the volume and core length

    https://www.wolframalpha.com/input/?i=solve+v%3Dpi%2F3*h%5E3%2Bl*h%5E2+for+h

    h = (sqrt(3) π sqrt(3 π^2 v^2 - 4 l^3 v) - 2 l^3 + 3 π^2 v)^(1/3)/(2^(1/3) π)
        + (2^(1/3) l^2)/(π (sqrt(3) π sqrt(3 π^2 v^2 - 4 l^3 v) - 2 l^3 + 3 π^2 v)^(1/3)) - l/π

    :param volume: stockpile volume
    :param core_length: stockpile core length
    :return: stockpile height
    """

    pi_sq_3_vol = 3.0 * pow(pi, 2.0) * volume
    two_3r = pow(2.0, 1.0 / 3.0)
    l_cu = pow(core_length, 3.0)
    inner = (pi_sq_3_vol - 4.0 * l_cu) * volume
    part = pow(sqrt(3.0) * pi * np.sqrt(np.array(inner, dtype=complex)) - 2.0 * l_cu + pi_sq_3_vol, 1.0 / 3.0)
    height = part / (two_3r * pi) + (two_3r * pow(core_length, 2.0)) / (pi * part) - core_length / pi

    return height.real


def get_stockpile_slice_core_area(x: float | np.ndarray, core_length: float, height: float):
    def clamp0h(v):
        return np.clip(v, 0.0, height)

    return sqrt(2.0) * (clamp0h(x) ** 2.0 - clamp0h(x - core_length) ** 2.0)


def get_stockpile_slice_half_cones_area(x: float | np.ndarray, core_length: float, height: float):
    def clamp0h(v):
        return np.clip(v, 0.0, height)

    def clamp02h(v):
        return np.clip(v, 0.0, 2.0 * height)

    def a(s: float | np.ndarray):
        return sqrt(2.0) * 2.0 / 3.0 * np.sqrt(clamp02h(s) * clamp02h(2.0 * height - s) ** 3.0)

    def f(s: float | np.ndarray):
        div = np.where(s < 2.0 * height, (height - clamp0h(s)) / (2.0 * height - s), 0.0)
        return (1.0 - 2.0 * div) ** 1.5

    return a(x) * (1.0 - f(x)) + a(x - core_length) * f(x - core_length)


def get_stockpile_slice_area(x: float | np.ndarray, core_length: float, height: float):
    core_area = get_stockpile_slice_core_area(x, core_length, height)
    half_cones_area = get_stockpile_slice_half_cones_area(x, core_length, height)
    return core_area + half_cones_area


def get_stockpile_slice_volume_norm(x: float | np.ndarray, core_length: float, height: float, x_diff: float | np.ndarray):
    return get_stockpile_slice_area(x, core_length, height) / sqrt(2.0) * x_diff


def get_stockpile_slice_volume(x: float | np.ndarray, core_length: float, height: float, x_min: float, x_diff: float | np.ndarray):
    return get_stockpile_slice_volume_norm(x - x_min + height, core_length, height, x_diff)


def get_ideal_reclaimed_volume(p: float | np.ndarray, core_length: float, radius: float) -> np.ndarray:
    """
    Volume reclaimed from an ideal stockpile with a 45 degree slope (base radius and height `radius`, core from 0 to `core_length`) when
    the cut of the reclaimer has reached the ground position p: the integral of the cut area over the reclaimer's travel, in closed form
    for any position and any core length (core and the two half-cones are integrated separately and added). 0 for p <= -radius, the
    whole volume for p >= core_length + radius.
    """
    p = np.asarray(p, dtype=float)
    r = radius

    def clamp(v, a, b):
        return np.clip(v, a, b)

    def k(u):
        return clamp(u, 0.0, r) ** 3 / 3.0 + r * r * np.maximum(u - r, 0.0)

    def w(q):  # a complete cone up to q
        q = clamp(q, -r, r)
        root = np.sqrt(np.maximum(r * r - q * q, 0.0))
        return r / 3.0 * q * root + r**3 / 3.0 * np.arcsin(q / r) + pi / 6.0 * r**3 + 2.0 / 9.0 * root**3

    def right_part(q):  # the part of a cone beyond its split plane
        return np.where(q <= 0.0, 2.0 / 9.0 * clamp(r + q, 0.0, r) ** 3, w(q) - pi / 6.0 * r**3)

    left_cone = w(p) - right_part(p)
    core = k(r + p) - k(r + p - core_length)
    return left_cone + core + right_part(p - core_length)


IDEAL_STOCKPILE_RULES = ("integrated", "point")


def get_ideal_stockpile_volumes(x: np.ndarray, volume: float, x_min: float, x_max: float, rule: str = "integrated") -> np.ndarray:
    """
    Volume of an ideal stockpile (core from x_min to x_max, 45 degree slopes, the given volume) in each reclaimed slice, the slice at x
    reaching from the previous reclaimer position to x (the first one from the start of the stockpile).

    rule "integrated" (default) integrates the cut area over each slice (`get_ideal_reclaimed_volume`), so the slices sum to the volume;
    "point" takes the cut area at x times the slice width, the rule of all results before 2026-10-17.
    """
    x = np.asarray(x, dtype=float)
    core_length = x_max - x_min
    height = get_stockpile_height(volume=volume, core_length=core_length)
    if rule == "point":
        x_diff = np.diff(x, prepend=x[0])
        return get_stockpile_slice_volume(x=x, core_length=core_length, height=height, x_min=x_min, x_diff=x_diff)
    if rule != "integrated":
        raise ValueError(f"unknown ideal stockpile rule {rule!r}, one of {IDEAL_STOCKPILE_RULES}")
    reclaimed = get_ideal_reclaimed_volume(x - x_min, core_length, height)
    return np.diff(reclaimed, prepend=0.0)
