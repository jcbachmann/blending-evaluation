import warnings

import blending_simulator_lib
import numpy as np
from blending_simulator_lib import BlendingSimulatorLib
from pandas import DataFrame

from ..helpers.continuous_drops import drop_grid
from .blending_simulator import BlendingSimulator, Material, MaterialDeposition


def optional_arguments(
    seed: int | None,
    lattice: bool,
    record_particles: bool,
    cone: bool = False,
    detailed_parameters: dict[str, float] | None = None,
    formula: bool = False,
    eight_height_scale: float = 0.0,
) -> tuple[tuple, dict]:
    """
    Arguments of newer simulator versions, only when used: bindings built against simulator v2026.1 have no seed, lattice and particle
    recording, those before 2026.3 no cone simulator and no parameters of the detailed simulation (the discrete element method), those
    of 2026.3 no formula simulator and no eight likelihood scaled with the pile height
    """
    seed_argument = () if seed is None else (seed,)
    options: dict = {}
    if lattice:
        options["lattice"] = True
    if cone:
        options["cone"] = True
    if formula:
        options["formula"] = True
    if eight_height_scale:
        options["eight_height_scale"] = eight_height_scale
    if record_particles:
        options["record_particles"] = True
    if detailed_parameters:
        options["detailed_parameters"] = detailed_parameters
    return seed_argument, options


DEFAULT_RECLAIM_ANGLE = 45.0
# 1 m slices at every particle size, so that F2 does not depend on the simulator's resolution through the slice width
DEFAULT_RECLAIM_INCREMENT = 1.0
# The eight likelihood of every fast simulation up to simulator release 2026.3; newer releases fit it to the particle size by default
DEFAULT_EIGHT_LIKELIHOOD = 0.87


def fitted_eight_likelihood(ppm3: float) -> float:
    """The fast simulation's eight likelihood fitted to the detailed simulation for the particle size, the default from simulator release
    2026.4 on (0.60, 0.60, 0.70, 0.75, 0.80, 0.85 at 1, 4, 16, 64, 256, 1,024 particles per cubic meter); bindings built against older
    releases have no fitted value and keep 0.87."""
    fitted = getattr(blending_simulator_lib, "fitted_eight_likelihood", None)
    return DEFAULT_EIGHT_LIKELIHOOD if fitted is None else float(fitted(ppm3))


def get_simulator_version() -> str | None:
    """The simulator release blending_simulator_lib is built from, e.g. "2026.3", or for a local checkout of the simulator "+local" with
    its commit where known, e.g. "2026.2+local.c79867b"; None for bindings built before they reported it."""
    return getattr(blending_simulator_lib, "simulator_version", None)


def particle_reclaim_increment(ppm3: float) -> float:
    """One particle size, the edge of a cube of the particle volume: the reclaim step of the C++ command line program, and of bmh before
    it took 1 m (DEFAULT_RECLAIM_INCREMENT) at every particle size"""
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
    reclaimangle: float = DEFAULT_RECLAIM_ANGLE,
    eight: float | None = None,
    cone: bool = False,
    detailed: bool = False,
    detailed_parameters: dict[str, float] | None = None,
    formula: bool = False,
    eight_height_scale: float = 0.0,
) -> dict[str, np.ndarray]:
    """
    Stack and reclaim many inputs at once like BslBlendingSimulator, in one call into the simulator: no Python per simulation,
    and with several threads the simulations run in parallel in this process.
    :param data: shape (inputs, rows, columns), for every input the rows of stack_reclaim_array
    :param columns: names of the columns of data
    :param seed: input i is seeded with seed + i; without a seed every simulation draws its own
    :param threads: simulations at the same time; 0 for one per core
    :param cone: the cone simulator (every row one portion, no particles) instead of the fast one
    :param formula: the formula simulator (the cone simulation reduced to the stacker line: every row one portion, all rows of an input at
        one z) instead of the fast one
    :param detailed: the detailed simulation (discrete element method) instead of the fast one
    :param detailed_parameters: parameters of the detailed simulation by their snake_case names (e.g. rolling_friction, flow_rate,
        threads for the threads of each simulation), see DetailedParameters in the simulator
    :param eight_height_scale: if positive, the fast simulation's eight likelihood is max(0, 1 - eight_height_scale / h) for a particle
        dropped on a pile h particles high instead of `eight`, which keeps round 45° piles at every ppm3 (6.5; simulator after 2026.3)
    :param eight: the fast simulation's eight likelihood; None for the value fitted to the particle size (`fitted_eight_likelihood`)
    :param reclaimincrement: the reclaimer's step, None for 1 m (`particle_reclaim_increment` gives the particle size)
    :return: reclaimer positions x, shape (slices,), and the reclaimed volume and the volume weighted average of each parameter, shape
        (inputs, slices)
    """
    if reclaimincrement is None:
        reclaimincrement = DEFAULT_RECLAIM_INCREMENT
    if eight is None:
        eight = fitted_eight_likelihood(ppm3)
    batch = getattr(blending_simulator_lib, "simulate_batch", None)
    if batch is not None:
        extra: dict = {"cone": True} if cone else {}
        if formula:
            extra["formula"] = True
        if eight_height_scale:
            extra["eight_height_scale"] = eight_height_scale
        if detailed:
            extra |= {"detailed": True, "detailed_parameters": detailed_parameters}
        result = batch(data, columns, bed_size_x, bed_size_z, reclaimangle, ppm3, False, eight, reclaimincrement, seed, lattice, threads, **extra)
        return {name: np.asarray(values) for name, values in result.items()}

    # Bindings built before the batch function: one simulator per input, in this thread
    if threads != 1:
        warnings.warn(
            "blending_simulator_lib has no simulate_batch, simulating serially; rebuild it (uv sync --reinstall-package blending_simulator_lib)",
            stacklevel=2,
        )
    results = [
        BslBlendingSimulator(
            bed_size_x,
            bed_size_z,
            reclaimangle=reclaimangle,
            ppm3=ppm3,
            eight=eight,
            reclaimincrement=reclaimincrement,
            seed=None if seed is None else seed + i,
            lattice=lattice,
            cone=cone,
            detailed=detailed,
            detailed_parameters=detailed_parameters,
            formula=formula,
            eight_height_scale=eight_height_scale,
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
        cone: bool = False,
        detailed_parameters: dict[str, float] | None = None,
        formula: bool = False,
        eight_height_scale: float = 0.0,
    ):
        """bulkdensity and dropheight are not used by simulators from 2026.3 on; detailed_parameters sets the parameters of their detailed
        simulation (discrete element method) by snake_case names, e.g. {"rolling_friction": 0.2, "threads": 4}; formula chooses the formula
        simulation (the cone simulation reduced to the stacker line; the stacker must not move across the bed, no circular beds);
        eight_height_scale replaces the fast simulation's constant eight likelihood by one scaled with the pile height (see simulate_batch);
        eight None is the eight likelihood fitted to the particle size (`fitted_eight_likelihood`)"""
        super().__init__(bed_size_x, bed_size_z)
        if reclaimangle is None:
            reclaimangle = DEFAULT_RECLAIM_ANGLE
        if ppm3 is None:
            ppm3 = 1.0
        if circular is None:
            circular = False
        if eight is None:
            eight = fitted_eight_likelihood(ppm3)
        if bulkdensity is None:
            bulkdensity = 1.0
        if dropheight is None:
            dropheight = 0.5 * bed_size_z
        if detailed is None:
            detailed = False
        if reclaimincrement is None:
            reclaimincrement = DEFAULT_RECLAIM_INCREMENT

        self.simulator = "cone" if cone else "formula" if formula else "lattice" if lattice else "detailed" if detailed else "fast"
        self.ppm3 = ppm3

        seed_argument, options = optional_arguments(seed, lattice, record_particles, cone, detailed_parameters, formula, eight_height_scale)
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

    def drop_grid(self) -> tuple[float, float]:
        return drop_grid(self.simulator, self.ppm3, self.bed_size_x, self.bed_size_z)

    def stack_reclaim(self, material_deposition: MaterialDeposition) -> Material:
        """
        Stack material according to material deposition and reclaim into new blended material.
        :param material_deposition: material and deposition data, stacked as continuous drops on the simulator's grid
        :return: reclaimed material
        """

        drops = material_deposition.drops(*self.drop_grid())
        data_dict = self.stack_reclaim_array(drops.to_numpy(), drops.columns.to_list())

        # Extract reclaimer speed from deposition meta
        reclaim_x_per_s = material_deposition.deposition.meta.reclaim_x_per_s

        # calculate timestamp column from x positions
        data_dict["timestamp"] = data_dict["x"] / reclaim_x_per_s

        # reorganize reclaimed material into pandas DataFrame
        data = DataFrame(data_dict)

        return Material.from_data(data, category="reclaimed")

    def get_heights(self, finish_stacking: bool = True):
        """Heights of the stockpile, after the material in motion came to rest; without finish_stacking the current state, so that looking
        at the stockpile while stacking does not pause the stacker of the detailed simulation (bindings from 2026.3 on)"""
        if finish_stacking:
            return self.bsl.get_heights()
        return self.bsl.get_heights(finish_stacking=False)

    def get_particles(self) -> dict:
        """Particles stacked so far as numpy arrays (position and size n x 3, parameters n x columns), needs record_particles."""
        return self.bsl.get_particles()
