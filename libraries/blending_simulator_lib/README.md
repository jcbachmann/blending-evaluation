# Blending Simulator Lib (blending_simulator_lib)

[![PyPI version](https://img.shields.io/pypi/v/blending_simulator_lib.svg)](https://pypi.org/project/blending_simulator_lib/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Simulation library for Bulk Material Homogenization, implemented in C++.

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
