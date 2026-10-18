"""The stockpile setup of the surrogate models and optimizations in bmh_ml, and the core limits of any bed.

The bed is 59 m long and 20 m deep and holds 2,500 m³, stacked over one day. The stacker travels along the middle of the bed's depth
and within the core of the stockpile: a stockpile with a 45° angle of repose as deep as the bed has the radius bed_size_z / 2, so
its core reaches from x = bed_size_z / 2 to bed_size_x - bed_size_z / 2 and its toes reach the ends of the bed. The stacking time does
not influence a simulation, it only sets the time scale of the material and deposition timestamps.
"""

BED_SIZE_X: int = 59
BED_SIZE_Z: int = 20
TOTAL_VOLUME: int = 2500
STACKING_TIME: int = 24 * 60 * 60  # seconds


def get_core_limits(bed_size_x: float, bed_size_z: float) -> tuple[float, float]:
    """The ends x_min and x_max of the stockpile's core on a bed, the limits of the stacker positions."""
    return 0.5 * bed_size_z, bed_size_x - 0.5 * bed_size_z


X_MIN, X_MAX = get_core_limits(BED_SIZE_X, BED_SIZE_Z)
