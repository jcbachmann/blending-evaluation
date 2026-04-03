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
