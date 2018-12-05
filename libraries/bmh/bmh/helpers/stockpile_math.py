from cmath import sqrt, pi

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
