"""
Python reference implementation of the lattice simulator.

LatticeSimulator ports the hexsim proof of concept (Code/hexsim/sim.py) close to its original and unoptimized: particles on a hexagonal
close-packed (hcp) lattice come to rest when all three lattice sites below them are filled, the lattice is compressed vertically to reach
a given angle of repose. BlendingSimulatorLatticeLib of BlendingSimulator implements the same rules in C++ and is tested against it.

It implements the BlendingSimulator interface of bmh. Reclaiming follows the C++ library: a particle whose base is at height y above
ground position x is reclaimed at position x - y / tan(reclaim angle), positions are clamped to the bed. Its volume is spread over its
size along the reclaim position, as the C++ library spreads its slices of particle size over finer reclaim increments.
"""

import math
from collections.abc import Iterator

import numpy as np
from bmh.simulation.blending_simulator import BlendingSimulator


class ParticleBuffer:
    """Collects stacked volume until it suffices for particles of a fixed volume, like AveragedParameters in the C++ library."""

    def __init__(self, particle_volume: float):
        self.particle_volume = particle_volume
        self.volume = 0.0
        self.sums: list[float] = []

    def push(self, volume: float, parameters: list[float]) -> None:
        if not self.sums:
            self.sums = [0.0] * len(parameters)
        self.volume += volume
        for i, p in enumerate(parameters):
            self.sums[i] += volume * p

    def pop_all(self) -> Iterator[list[float]]:
        """Parameters of every complete particle, the volume weighted average of the buffered material."""
        # The tolerance keeps the last particle of material that adds up to whole particles despite rounding errors
        while self.volume >= self.particle_volume * (1.0 - 1e-9):
            values = [s / self.volume for s in self.sums]
            remaining = max(0.0, self.volume - self.particle_volume)
            self.sums = [s * remaining / self.volume for s in self.sums]
            self.volume = remaining
            yield values


class ParticleSimulator(BlendingSimulator):
    """Common stacking, particle bookkeeping and reclaiming of the particle simulators."""

    # Particle shape for visualization: "box" or "sphere"
    shape = "box"

    def __init__(self, bed_size_x: float, bed_size_z: float, *, ppm3: float, reclaim_angle: float = 45.0, reclaim_increment: float = 1.0):
        super().__init__(bed_size_x, bed_size_z)
        self.particle_volume = 1.0 / ppm3
        self.tan_reclaim_angle = math.tan(math.radians(reclaim_angle))
        self.reclaim_increment = reclaim_increment
        self.buffer = ParticleBuffer(self.particle_volume)
        # Center x, base height y and center z of every particle in m
        self.positions: list[tuple[float, float, float]] = []
        self.parameters: list[list[float]] = []
        # Stacking time of every particle
        self.times: list[float] = []
        self.lost_particles = 0

    @property
    def particle_size(self) -> float:
        """Edge length of a box or horizontal diameter of a sphere in m."""
        raise NotImplementedError()

    @property
    def particle_height(self) -> float:
        """Vertical extent of a particle in m."""
        return self.particle_size

    def drop(self, x: float, z: float) -> tuple[float, float, float] | None:
        """Drop a particle at (x, z), returning its resting position (center x, base y, center z) or None if it did not fit on the bed."""
        raise NotImplementedError()

    def stack(self, timestamp: float, x: float, z: float, volume: float, parameter: list[float]) -> None:
        self.buffer.push(volume, parameter)
        for values in self.buffer.pop_all():
            position = self.drop(x, z)
            if position is None:
                self.lost_particles += 1
            else:
                self.positions.append(position)
                self.parameters.append(values)
                self.times.append(timestamp)

    def reclaim_intervals(self) -> np.ndarray:
        """Start of the reclaim interval of every particle: a particle spans its size along the reclaim position, clamped to the bed."""
        positions = np.asarray(self.positions, dtype=float).reshape(-1, 3)
        size = self.particle_size
        reclaim_position = positions[:, 0] - positions[:, 1] / self.tan_reclaim_angle
        return np.clip(reclaim_position - 0.5 * size, 0.0, self.bed_size_x - size)

    def reclaim(self) -> list[list[float | list[float]]]:
        slice_count = math.ceil(self.bed_size_x / self.reclaim_increment - 1e-9)
        parameter_count = len(self.parameters[0]) if self.parameters else 0
        volumes = np.zeros(slice_count)
        sums = np.zeros((slice_count, parameter_count))
        if self.positions:
            # Split the volume of every particle over the slices its interval overlaps, like the C++ library splits its native slices
            starts = self.reclaim_intervals()
            ends = starts + self.particle_size
            parameters = np.asarray(self.parameters)
            first = np.floor(starts / self.reclaim_increment).astype(int)
            for k in range(math.ceil(self.particle_size / self.reclaim_increment) + 1):
                index = first + k
                overlap = np.minimum(ends, (index + 1) * self.reclaim_increment) - np.maximum(starts, index * self.reclaim_increment)
                volume = self.particle_volume * np.clip(overlap, 0.0, None) / self.particle_size
                valid = (volume > 0) & (index < slice_count)
                np.add.at(volumes, index[valid], volume[valid])
                np.add.at(sums, index[valid], volume[valid, None] * parameters[valid])
        qualities = np.divide(sums, volumes[:, None], out=np.zeros_like(sums), where=volumes[:, None] > 0)
        # Like the C++ library, each slice is reported at the reclaimer position after reclaiming it
        return [[(i + 1) * self.reclaim_increment, float(volumes[i]), qualities[i].tolist()] for i in range(slice_count)]


class LatticeSimulator(ParticleSimulator):
    """
    Port of hexsim: spheres on a hexagonal close-packed (hcp) lattice, layers of triangular lattices stacked ABAB.

    Lattice coordinates (xi, yi, zi) are chosen by hexsim such that (xi, yi - 1, zi) is always one of the three sites below, so filled
    sites form contiguous columns and a height per (xi, zi) describes the pile. Differences to hexsim: the bed is a rectangle in the world,
    distances are measured in the world instead of in mixed lattice units, and the face-centered cubic (fcc) variant is left out because
    its supports all lie towards +x and +z, which makes its piles lean.

    The resting rule only depends on which sites are filled, so compressing the lattice vertically keeps every pile and scales the tangent
    of all slopes by the same factor. Piles of the uncompressed lattice are hexagonal pyramids with the volume of a cone of
    NATIVE_ANGLE_OF_REPOSE (derived in the Entropy note Lattice Simulation); the compression maps that to angle_of_repose. The spheres
    become spheroids of height vertical_scale times their diameter and keep the particle volume.
    """

    shape = "sphere"

    # Slope of the cone with the height and volume of a pile on the uncompressed lattice, 60.89° (faces 62.06°, edges 58.52°)
    NATIVE_TAN_ANGLE_OF_REPOSE = 4.0 / 3.0 * math.sqrt(math.pi / math.sqrt(3.0))
    NATIVE_ANGLE_OF_REPOSE = math.degrees(math.atan(NATIVE_TAN_ANGLE_OF_REPOSE))

    def __init__(self, bed_size_x: float, bed_size_z: float, *, ppm3: float, angle_of_repose: float | None = 45.0, **kwargs):
        super().__init__(bed_size_x, bed_size_z, ppm3=ppm3, **kwargs)
        if angle_of_repose is None:
            self.vertical_scale = 1.0
        else:
            self.vertical_scale = math.tan(math.radians(angle_of_repose)) / self.NATIVE_TAN_ANGLE_OF_REPOSE
        # A close-packed sphere of diameter a occupies a / sqrt(2) cubed, compressed by the vertical scale; that is the particle volume
        self.a = (math.sqrt(2.0) * self.particle_volume / self.vertical_scale) ** (1.0 / 3.0)
        self.row_distance = self.a * math.sqrt(3.0) / 2.0
        self.layer_distance = self.a * math.sqrt(2.0 / 3.0) * self.vertical_scale
        self.heights: dict[tuple[int, int], int] = {}
        self.max_height = 0

    @property
    def particle_size(self) -> float:
        return self.a

    @property
    def particle_height(self) -> float:
        return self.a * self.vertical_scale

    def coordinates_to_units(self, xi: int, yi: int, zi: int) -> tuple[float, float]:
        """Horizontal position in units of the lattice: x in sphere diameters, z in rows."""
        return xi + (zi % 2) / 2.0 + (yi % 2) / 2.0, zi + (yi % 2) / 3.0

    def units_to_coordinates(self, xu: float, zu: float, yi: int = 0) -> tuple[int, int, int]:
        zi = round(zu - (yi % 2) / 3.0)
        return round(xu - (yi % 2) / 2.0 - (zi % 2) / 2.0), yi, zi

    def coordinates_to_world(self, xi: int, yi: int, zi: int) -> tuple[float, float, float]:
        """Sphere center x, base y and center z in m."""
        xu, zu = self.coordinates_to_units(xi, yi, zi)
        return (xu + 0.5) * self.a, yi * self.layer_distance, zu * self.row_distance + 0.5 * self.a

    def world_to_units(self, x: float, z: float) -> tuple[float, float]:
        return x / self.a - 0.5, (z - 0.5 * self.a) / self.row_distance

    def valid_coordinates(self, xi: int, yi: int, zi: int) -> bool:
        if yi < 0:
            return False
        x, _, z = self.coordinates_to_world(xi, yi, zi)
        return 0.0 <= x < self.bed_size_x and 0.0 <= z < self.bed_size_z

    def is_filled(self, xi: int, yi: int, zi: int) -> bool:
        # Sites outside of the bed act like walls
        if not self.valid_coordinates(xi, yi, zi):
            return True
        return self.heights.get((xi, zi), 0) > yi

    def get_spaces_below(self, xi: int, yi: int, zi: int) -> list[tuple[int, int, int]]:
        ys = yi % 2
        zs = zi % 2
        return [
            (xi + ys + zs - ys * zs - 1, yi - 1, zi + ys - 1),
            (xi + ys, yi - 1, zi),
            (xi + ys + zs * ys - 1, yi - 1, zi + ys),
        ]

    def get_free_spaces_below(self, xi: int, yi: int, zi: int) -> list[tuple[int, int, int]]:
        return [s for s in self.get_spaces_below(xi, yi, zi) if not self.is_filled(*s)]

    def drop_coordinates(self, xu: float, zu: float) -> tuple[int, int, int] | None:
        # Look for the first site from above that is free and touches the pile
        y = self.max_height
        while True:
            xi, yi, zi = self.units_to_coordinates(xu, zu, yi=y)
            if self.valid_coordinates(xi, yi, zi):
                if self.is_filled(xi, yi, zi):
                    return None
                if len(self.get_free_spaces_below(xi, yi, zi)) < 3:
                    break
            elif y == 0:
                return None
            y -= 1

        # Fall until all three sites below are filled, preferring the site closest to the drop position
        while True:
            free = self.get_free_spaces_below(xi, yi, zi)
            if not free:
                return xi, yi, zi
            if len(free) == 1:
                xi, yi, zi = free[0]
            else:

                def distance(site: tuple[int, int, int]) -> float:
                    sxu, szu = self.coordinates_to_units(*site)
                    return math.hypot((xu - sxu) * self.a, (zu - szu) * self.row_distance)

                xi, yi, zi = min(free, key=distance)

    def drop(self, x: float, z: float) -> tuple[float, float, float] | None:
        site = self.drop_coordinates(*self.world_to_units(x, z))
        if site is None:
            return None
        xi, yi, zi = site
        self.heights[(xi, zi)] = yi + 1
        self.max_height = max(self.max_height, yi + 1)
        return self.coordinates_to_world(xi, yi, zi)
