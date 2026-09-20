"""Labels inputs with the simulator: F1 and F2, the mean of several simulations of the same input."""

import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bmh_ml.datasets.manifest import Dataset
from bmh_ml.settings import BED_SIZE_X, BED_SIZE_Z, TOTAL_VOLUME
from bmh_ml.simulation import evaluate_sim


def simulate_chunk(material: np.ndarray, deposition: np.ndarray, repeats: int) -> np.ndarray:
    """Shape (rows, repeats, 2). `material` has one row per deposition."""
    result = np.empty((len(deposition), repeats, 2))
    for i, (material_variables, deposition_variables) in enumerate(zip(material, deposition, strict=True)):
        for r in range(repeats):
            result[i, r] = evaluate_sim(list(material_variables), list(deposition_variables), BED_SIZE_X, BED_SIZE_Z, TOTAL_VOLUME)
    return result


def simulate(
    material: np.ndarray, deposition: np.ndarray, repeats: int = 1, n_jobs: int | None = None, chunk_size: int = 500
) -> tuple[np.ndarray, np.ndarray | None]:
    """The mean of the simulations of each input and, if repeated, their standard deviation.

    The simulation is random itself (particles are sampled), so the labels differ between calls. `material` is one row for all depositions or one row each.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    material = np.broadcast_to(material, (len(deposition), material.shape[1]))
    n_jobs = n_jobs or os.cpu_count() or 1
    starts = range(0, len(deposition), chunk_size)
    chunks = [(material[s : s + chunk_size], deposition[s : s + chunk_size]) for s in starts]

    if n_jobs == 1 or len(chunks) == 1:
        results = [simulate_chunk(m, d, repeats) for m, d in chunks]
    else:
        logging.info(f"Simulating {len(deposition)} inputs x {repeats} repeats with {n_jobs} processes")
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(simulate_chunk, *zip(*chunks, strict=True), [repeats] * len(chunks)))

    samples = np.concatenate(results)
    y = samples.mean(axis=1)
    y_noise_sd = samples.std(axis=1, ddof=1) if repeats > 1 else None
    return y, y_noise_sd


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
) -> Dataset:
    material = np.atleast_2d(np.asarray(material, dtype=float))
    deposition = np.asarray(deposition, dtype=float)
    y, y_noise_sd = simulate(material, deposition, repeats, n_jobs)
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
    )
