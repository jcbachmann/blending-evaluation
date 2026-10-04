"""Labels inputs with the simulator: F1 and F2, the mean of several simulations of the same input."""

import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bmh_ml.datasets.manifest import Dataset
from bmh_ml.parallel import get_cpu_count
from bmh_ml.settings import BED_SIZE_X, BED_SIZE_Z, PROFILE_LENGTH, TOTAL_VOLUME, get_simulator_settings
from bmh_ml.simulation import evaluate_sim_with_profile


def simulate_chunk(material: np.ndarray, deposition: np.ndarray, repeats: int, with_profiles: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
    """Objectives of shape (rows, repeats, 2) and, if asked for, the mean profile of the repeats, shape (rows, 2, slices).

    `material` has one row per deposition."""
    result = np.empty((len(deposition), repeats, 2))
    profiles = np.zeros((len(deposition), 2, PROFILE_LENGTH)) if with_profiles else None
    for i, (material_variables, deposition_variables) in enumerate(zip(material, deposition, strict=True)):
        for r in range(repeats):
            f1, f2, profile = evaluate_sim_with_profile(list(material_variables), list(deposition_variables), BED_SIZE_X, BED_SIZE_Z, TOTAL_VOLUME)
            result[i, r] = f1, f2
            if profiles is not None:
                profiles[i] += profile / repeats
    return result, profiles


def simulate(
    material: np.ndarray, deposition: np.ndarray, repeats: int = 1, n_jobs: int | None = None, chunk_size: int = 500
) -> tuple[np.ndarray, np.ndarray | None]:
    """The mean of the simulations of each input and, if repeated, their standard deviation.

    The simulation is random itself (particles are sampled), so the labels differ between calls. `material` is one row for all depositions or one row each.
    """
    y, y_noise_sd, _ = simulate_with_profiles(material, deposition, repeats, n_jobs, chunk_size, with_profiles=False)
    return y, y_noise_sd


def simulate_with_profiles(
    material: np.ndarray, deposition: np.ndarray, repeats: int = 1, n_jobs: int | None = None, chunk_size: int = 500, with_profiles: bool = True
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Like `simulate`, and the mean reclaimed profile of each input (volume and quality per slice, shape (rows, 2, slices))."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    material = np.broadcast_to(material, (len(deposition), material.shape[1]))
    n_jobs = n_jobs or get_cpu_count()
    starts = range(0, len(deposition), chunk_size)
    chunks = [(material[s : s + chunk_size], deposition[s : s + chunk_size]) for s in starts]

    if n_jobs == 1 or len(chunks) == 1:
        results = [simulate_chunk(m, d, repeats, with_profiles) for m, d in chunks]
    else:
        logging.info(f"Simulating {len(deposition)} inputs x {repeats} repeats with {n_jobs} processes")
        # spawn, not fork: forking a process that already runs TensorFlow threads can deadlock (seen in the refinement loop)
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=multiprocessing.get_context("spawn")) as executor:
            results = list(executor.map(simulate_chunk, *zip(*chunks, strict=True), [repeats] * len(chunks), [with_profiles] * len(chunks)))

    samples = np.concatenate([result[0] for result in results])
    y = samples.mean(axis=1)
    y_noise_sd = samples.std(axis=1, ddof=1) if repeats > 1 else None
    profiles = np.concatenate([result[1] for result in results]) if with_profiles else None
    return y, y_noise_sd, profiles


def build_dataset(
    name: str,
    generator: str,
    seed: int,
    material: np.ndarray,
    deposition: np.ndarray,
    repeats: int = 1,
    n_jobs: int | None = None,
    settings: dict | None = None,
    source: dict | None = None,
    with_profiles: bool = False,
) -> Dataset:
    """Simulates the inputs and returns the dataset; with profiles, the mean reclaimed profile of each input is stored as well."""
    material = np.atleast_2d(np.asarray(material, dtype=float))
    deposition = np.asarray(deposition, dtype=float)
    y, y_noise_sd, profiles = simulate_with_profiles(material, deposition, repeats, n_jobs, with_profiles=with_profiles)
    settings = {**(settings or {}), "simulator": get_simulator_settings().as_dict()}  # the detail level the labels come from
    return Dataset(
        name=name,
        generator=generator,
        seed=seed,
        repeats=repeats,
        material=material,
        deposition=deposition,
        y=y,
        y_noise_sd=y_noise_sd,
        settings=settings or {},
        source=source or {},
        profiles=profiles,
    )
