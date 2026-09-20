import numpy as np
from scipy.ndimage import gaussian_filter


def generate_material_variables(material_length: int, material_min: float, material_max: float) -> list[float]:
    rng = np.random.default_rng()
    start = float(rng.uniform(material_min, material_max))
    step_size = 1
    walk = [start]
    for _ in range(material_length - 1):
        step = float(rng.uniform(-step_size, step_size))
        next_value = min(max(material_min, walk[-1] + step), material_max)
        walk.append(next_value)
    return gaussian_filter(np.array(walk), sigma=0.75).tolist()


def generate_deposition_variables(deposition_length, x_min, x_max):
    rng = np.random.default_rng()
    return rng.uniform(x_min, x_max, deposition_length)
