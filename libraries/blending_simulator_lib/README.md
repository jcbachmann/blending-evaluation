# Blending Simulator Lib (blending_simulator_lib)

[![PyPI version](https://img.shields.io/pypi/v/blending_simulator_lib.svg)](https://pypi.org/project/blending_simulator_lib/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Simulation library for Bulk Material Homogenization, implemented in C++.

## Usage

```python
from blending_simulator_lib import BlendingSimulatorLib

sim = BlendingSimulatorLib(
    60.0,  # heap_world_size_x: bed length in m
    20.0,  # heap_world_size_z: bed depth in m
    45.0,  # reclaim_angle in degrees
    8.0,  # particles_per_cubic_meter
    False,  # circular
    0.87,  # eight_likelihood (fast simulation)
    1.0,  # bulk_density_factor (detailed simulation)
    10.0,  # drop_height in m (detailed simulation)
    False,  # detailed
    1.0,  # reclaim_increment in m
    seed=1,  # optional, random if not given
    lattice=True,  # lattice simulation on a hexagonal close-packed lattice (angle of repose 45°) instead of the fast one
    record_particles=True,  # keep every particle for get_particles()
    # cone=True would choose the cone simulation instead: no particles, every stacked row one portion forming a 45° cone over the pile;
    # formula=True the formula simulation, the cone simulation reduced to the stacker line (all rows at one z, no circular beds)
)
sim.stack_list(data, ["timestamp", "x", "z", "volume", "quality"])  # numpy array, one row per increment; other columns are parameters
sim.get_heights()  # height map, list of rows along z
sim.get_particles()  # dict of numpy arrays: position (n x 3 centers), size (n x 3), parameters (n x parameters) and columns
sim.reclaim()  # dict of x, volume and the parameters per reclaimed slice
```

The simulators are described in the README of the `blending-simulation` repository. `bmh.simulation.BslBlendingSimulator` wraps this
class in the `BlendingSimulator` interface of `bmh`.

`simulate_batch(data, columns, ..., lattice=False, threads=1, cone=False, formula=False)` stacks and reclaims many inputs in one call.
The formula simulation needs a simulator release after 2026.3; bindings built against an older one raise a `RuntimeError` for
`formula=True`.
`continuous_drops(volumes, parameters, portion_times, path_times, path_x, path_z, cell_x, cell_z)` turns a material that flows while the
stacker moves along a piecewise linear path into rows (x, z, volume, parameters): one drop per grid cell the path crosses, with the volume
of the time spent over it (`bmh.helpers.continuous_drops` wraps it, with the cell sizes of each simulator in `drop_grid`).

## Development

After making changes to the module, you need to regenerate the stubs. To regenerate the stubs, execute this command from
the package directory:

```bash
uv run pybind11-stubgen blending_simulator_lib._blending_simulator_lib -o src
```

The C++ simulator is fetched from the `blending-simulation` repository at the tag set in `CMakeLists.txt`. To build against a local
checkout instead, for example while changing both, point CMake's `FetchContent` to it:

```bash
SKBUILD_CMAKE_DEFINE="FETCHCONTENT_SOURCE_DIR_BLENDING-SIMULATION=/path/to/BlendingSimulator" uv sync --reinstall-package blending_simulator_lib
```
