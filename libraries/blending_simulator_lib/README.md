# Blending Simulator Lib for Python

This package wraps and tests the Blending Simulator code written in C++.

## Stubs

After making changes to the module, you need to regenerate the stubs. To regenerate the stubs, execute this command from
the package directory:

```bash
uv run pybind11-stubgen blending_simulator_lib._blending_simulator_lib -o src
```
