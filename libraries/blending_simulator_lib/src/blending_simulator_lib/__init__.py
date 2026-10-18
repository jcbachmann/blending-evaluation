from . import _blending_simulator_lib
from ._blending_simulator_lib import BlendingSimulatorLib, continuous_drops, simulate_batch, simulator_version

__all__ = [
    "BlendingSimulatorLib",
    "continuous_drops",
    "simulate_batch",
    "simulator_version",
]

# Simulators from 2026.4 on fit the fast simulation's eight likelihood to the particle size
if hasattr(_blending_simulator_lib, "fitted_eight_likelihood"):
    fitted_eight_likelihood = _blending_simulator_lib.fitted_eight_likelihood
    __all__.append("fitted_eight_likelihood")
