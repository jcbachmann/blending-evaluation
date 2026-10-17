"""
Blending Simulator Lib for Python
"""

from __future__ import annotations

import collections.abc
import typing

import numpy
import numpy.typing

__all__: list[str] = ["BlendingSimulatorLib", "continuous_drops", "simulate_batch", "simulator_version"]

class BlendingSimulatorLib:
    def __init__(
        self,
        heap_world_size_x: typing.SupportsFloat | typing.SupportsIndex,
        heap_world_size_z: typing.SupportsFloat | typing.SupportsIndex,
        reclaim_angle: typing.SupportsFloat | typing.SupportsIndex,
        particles_per_cubic_meter: typing.SupportsFloat | typing.SupportsIndex,
        circular: bool,
        eight_likelihood: typing.SupportsFloat | typing.SupportsIndex,
        bulk_density_factor: typing.SupportsFloat | typing.SupportsIndex,
        drop_height: typing.SupportsFloat | typing.SupportsIndex,
        detailed: bool,
        reclaim_increment: typing.SupportsFloat | typing.SupportsIndex,
        seed: typing.SupportsInt | typing.SupportsIndex | None = None,
        lattice: bool = False,
        record_particles: bool = False,
        cone: bool = False,
    ) -> None: ...
    def get_heights(self) -> list[list[float]]: ...
    def get_particles(self) -> dict: ...
    def reclaim(self) -> dict: ...
    def stack(
        self,
        arg0: typing.SupportsFloat | typing.SupportsIndex,
        arg1: typing.SupportsFloat | typing.SupportsIndex,
        arg2: typing.SupportsFloat | typing.SupportsIndex,
        arg3: typing.SupportsFloat | typing.SupportsIndex,
        arg4: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex],
    ) -> None: ...
    def stack_list(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], arg1: collections.abc.Sequence[str]) -> None: ...

def simulate_batch(
    data: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    columns: collections.abc.Sequence[str],
    heap_world_size_x: typing.SupportsFloat | typing.SupportsIndex,
    heap_world_size_z: typing.SupportsFloat | typing.SupportsIndex,
    reclaim_angle: typing.SupportsFloat | typing.SupportsIndex,
    particles_per_cubic_meter: typing.SupportsFloat | typing.SupportsIndex,
    circular: bool,
    eight_likelihood: typing.SupportsFloat | typing.SupportsIndex,
    reclaim_increment: typing.SupportsFloat | typing.SupportsIndex,
    seed: typing.SupportsInt | typing.SupportsIndex | None = None,
    lattice: bool = False,
    threads: typing.SupportsInt | typing.SupportsIndex = 1,
    cone: bool = False,
) -> dict: ...
def continuous_drops(
    volumes: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex],
    parameters: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    portion_times: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex],
    path_times: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex],
    path_x: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex],
    path_z: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex],
    cell_x: typing.SupportsFloat | typing.SupportsIndex,
    cell_z: typing.SupportsFloat | typing.SupportsIndex,
) -> numpy.typing.NDArray[numpy.float64]: ...

simulator_version: str
