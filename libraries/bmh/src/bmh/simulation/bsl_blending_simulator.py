import math

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
            reclaimangle = 45.0
        if ppm3 is None:
            ppm3 = 1.0
        if circular is None:
            circular = False
        if eight is None:
            eight = 0.87
        if bulkdensity is None:
            bulkdensity = 1.0
        if dropheight is None:
            dropheight = 0.5 * bed_size_z
        if detailed is None:
            detailed = False
        if reclaimincrement is None:
            reclaimincrement = 1.0 / math.sqrt(ppm3)

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

    def stack_reclaim(self, material_deposition: MaterialDeposition) -> Material:
        """
        Stack material according to material deposition and reclaim into new blended material.
        :param material_deposition: material and deposition data
        :return: reclaimed material
        """

        # stack all data
        self.bsl.stack_list(
            material_deposition.data.to_numpy(),
            material_deposition.data.columns.to_list(),
        )

        # reclaim stacked material
        data_dict = self.bsl.reclaim()

        # Extract reclaimer speed from deposition meta
        reclaim_x_per_s = material_deposition.deposition.meta.reclaim_x_per_s

        # calculate timestamp column from x positions
        data_dict["timestamp"] = [v / reclaim_x_per_s for v in data_dict["x"]]

        # reorganize reclaimed material into pandas DataFrame
        data = DataFrame(data_dict)

        return Material.from_data(data, category="reclaimed")

    def get_heights(self):
        return self.bsl.get_heights()

    def get_particles(self) -> dict:
        """Particles stacked so far as numpy arrays (position and size n x 3, parameters n x columns), needs record_particles."""
        return self.bsl.get_particles()
