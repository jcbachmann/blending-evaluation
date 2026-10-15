import warnings

import blending_simulator_lib
import numpy as np
from blending_simulator_lib import BlendingSimulatorLib
from pandas import DataFrame

from .blending_simulator import BlendingSimulator, Material, MaterialDeposition


def optional_arguments(seed: int | None, lattice: bool, record_particles: bool) -> tuple[tuple, dict]:
    """
    Arguments of newer simulator versions, only when used: bindings built against simulator v2026.1 have no seed, lattice and particle recording
    """
    seed_argument = () if seed is None else (seed,)
    options: dict = {}
    if lattice:
        options["lattice"] = True
    if record_particles:
        options["record_particles"] = True
    return seed_argument, options


DEFAULT_RECLAIM_ANGLE = 45.0
DEFAULT_EIGHT_LIKELIHOOD = 0.87


def default_reclaim_increment(ppm3: float) -> float:
    """One particle size, the edge of a cube of the particle volume, like the C++ command line program"""
    return ppm3 ** (-1.0 / 3.0)


def simulate_batch(
    data: np.ndarray,
    columns: list[str],
    bed_size_x: float,
    bed_size_z: float,
    ppm3: float = 1.0,
    reclaimincrement: float | None = None,
    seed: int | None = None,
    lattice: bool = False,
    threads: int = 1,
) -> dict[str, np.ndarray]:
    """
    Stack and reclaim many inputs at once with the defaults of BslBlendingSimulator, in one call into the simulator: no Python per simulation,
    and with several threads the simulations run in parallel in this process.
    :param data: shape (inputs, rows, columns), for every input the rows of stack_reclaim_array
    :param columns: names of the columns of data
    :param seed: input i is seeded with seed + i; without a seed every simulation draws its own
    :param threads: simulations at the same time; 0 for one per core
    :return: reclaimer positions x, shape (slices,), and the reclaimed volume and the volume weighted average of each parameter, shape
        (inputs, slices)
    """
    if reclaimincrement is None:
        reclaimincrement = default_reclaim_increment(ppm3)
    batch = getattr(blending_simulator_lib, "simulate_batch", None)
    if batch is not None:
        result = batch(
            data, columns, bed_size_x, bed_size_z, DEFAULT_RECLAIM_ANGLE, ppm3, False, DEFAULT_EIGHT_LIKELIHOOD, reclaimincrement, seed, lattice, threads
        )
        return {name: np.asarray(values) for name, values in result.items()}

    # Bindings built before the batch function: one simulator per input, in this thread
    if threads != 1:
        warnings.warn(
            "blending_simulator_lib has no simulate_batch, simulating serially; rebuild it (uv sync --reinstall-package blending_simulator_lib)",
            stacklevel=2,
        )
    results = [
        BslBlendingSimulator(
            bed_size_x, bed_size_z, ppm3=ppm3, reclaimincrement=reclaimincrement, seed=None if seed is None else seed + i, lattice=lattice
        ).stack_reclaim_array(input_data, columns)
        for i, input_data in enumerate(data)
    ]
    return {"x": results[0]["x"], **{name: np.stack([result[name] for result in results]) for name in results[0] if name != "x"}}


class BslBlendingSimulator(BlendingSimulator):
    def __init__(
        self,
        bed_size_x: float,
        bed_size_z: float,
        reclaimangle: float | None = None,
        ppm3: float | None = None,
        circular: bool | None = None,
        eight: float | None = None,
        bulkdensity: float | None = None,
        dropheight: float | None = None,
        detailed: bool | None = None,
        reclaimincrement: float | None = None,
        seed: int | None = None,
        lattice: bool = False,
        record_particles: bool = False,
    ):
        super().__init__(bed_size_x, bed_size_z)
        if reclaimangle is None:
            reclaimangle = DEFAULT_RECLAIM_ANGLE
        if ppm3 is None:
            ppm3 = 1.0
        if circular is None:
            circular = False
        if eight is None:
            eight = DEFAULT_EIGHT_LIKELIHOOD
        if bulkdensity is None:
            bulkdensity = 1.0
        if dropheight is None:
            dropheight = 0.5 * bed_size_z
        if detailed is None:
            detailed = False
        if reclaimincrement is None:
            reclaimincrement = default_reclaim_increment(ppm3)

        seed_argument, options = optional_arguments(seed, lattice, record_particles)
        self.bsl = BlendingSimulatorLib(
            bed_size_x,
            bed_size_z,
            reclaimangle,
            ppm3,
            circular,
            eight,
            bulkdensity,
            dropheight,
            detailed,
            reclaimincrement,
            *seed_argument,
            **options,
        )

    def stack(self, timestamp: float, x: float, z: float, volume: float, parameter: list[float]) -> None:
        self.bsl.stack(timestamp, x, z, volume, parameter)  # noqa: PD013

    def reclaim(self) -> list[list[float | list[float]]]:
        data_dict = self.bsl.reclaim()
        x = data_dict.pop("x")
        volume = data_dict.pop("volume")
        parameters = list(zip(*data_dict.values(), strict=True)) if data_dict else [()] * len(x)
        return [[p, v, list(q)] for p, v, q in zip(x, volume, parameters, strict=True)]

    def stack_reclaim_array(self, data: np.ndarray, columns: list[str]) -> dict[str, np.ndarray]:
        """
        Stack material and reclaim it without pandas, for callers that simulate many small inputs: pandas costs several times the simulation
        at a coarse detail level.
        :param data: one row per stacked portion, with the columns x, z and volume, a parameter column for each further name and optionally
            a timestamp column, which is ignored
        :param columns: names of the columns of data
        :return: reclaimer positions (x), reclaimed volumes and the volume weighted average of each parameter per reclaimed slice
        """
        self.bsl.stack_list(data, columns)
        return {name: np.asarray(values, dtype=float) for name, values in self.bsl.reclaim().items()}

    def stack_reclaim(self, material_deposition: MaterialDeposition) -> Material:
        """
        Stack material according to material deposition and reclaim into new blended material.
        :param material_deposition: material and deposition data
        :return: reclaimed material
        """

        data_dict = self.stack_reclaim_array(material_deposition.data.to_numpy(), material_deposition.data.columns.to_list())

        # Extract reclaimer speed from deposition meta
        reclaim_x_per_s = material_deposition.deposition.meta.reclaim_x_per_s

        # calculate timestamp column from x positions
        data_dict["timestamp"] = data_dict["x"] / reclaim_x_per_s

        # reorganize reclaimed material into pandas DataFrame
        data = DataFrame(data_dict)

        return Material.from_data(data, category="reclaimed")

    def get_heights(self):
        return self.bsl.get_heights()

    def get_particles(self) -> dict:
        """Particles stacked so far as numpy arrays (position and size n x 3, parameters n x columns), needs record_particles."""
        return self.bsl.get_particles()
