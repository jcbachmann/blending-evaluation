import math
from typing import Tuple


def get_bed_size(
        volume: float,
        *,
        max_stockpile_height: float = 20.0,
        volume_factor: float = 1.25
) -> Tuple[float, float]:
    max_volume = volume * volume_factor
    core_length = (max_volume - math.pi * math.pow(max_stockpile_height, 3) / 3) / math.pow(max_stockpile_height, 2)
    bed_size_x = core_length + 2 * max_stockpile_height
    bed_size_z = 2 * max_stockpile_height
    return bed_size_x, bed_size_z
