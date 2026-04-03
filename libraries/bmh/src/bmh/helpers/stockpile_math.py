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


def get_ideal_stockpile_volumes(x: np.ndarray, volume: float, x_min: float, x_max: float) -> np.ndarray:
    core_length = x_max - x_min
    height = get_stockpile_height(volume=volume, core_length=core_length)
    x_diff = np.diff(x, prepend=x[0])
    return get_stockpile_slice_volume(x=x, core_length=core_length, height=height, x_min=x_min, x_diff=x_diff)
