import numpy as np
import pandas as pd

from bmh.benchmark.material_deposition import Deposition, Material, MaterialDeposition
from bmh.simulation.bsl_blending_simulator import BslBlendingSimulator


def generate_material_deposition(quality: list[float], total_volume: float, x_position: list[float], bed_size_x: int, bed_size_z: int) -> MaterialDeposition:
    """Helper function to generate material deposition from quality and x-position lists.

    This helper function generates a MaterialDeposition object from the provided lists for quality and x-position. It serves as a simple entrypoint into
    blending simulation. It falls back on simple defaults: All variables are automatically matched in size. Timestamps are linearly spaced between 0 and max
    timestamp of 24 hours in seconds. The volume of is divided equally among the qualities. The reclaim rate is set to 1 per second. The z-position is set to be
    in the middle of the bed. Material quality is interpolated to up to 1000 chunks depending on the total volume, with each chunk having a size of 100 units.

    :param quality: List of material qualities.
    :param total_volume: Total volume of material to be deposited.
    :param x_position: List of x-positions for material deposition.
    :param bed_size_x: Integer value for bed size in x-direction.
    :param bed_size_z: Integer value for bed size in z-direction.

    :return: MaterialDeposition object.
    """
    max_timestamp = 24 * 60 * 60
    chunk_size = 100
    chunks = min(max(1, int(total_volume // chunk_size)), 1000)
    n = len(quality)
    i_quality = np.linspace(0, n - 1, n)
    i_interpolated = np.linspace(0, n - 1, chunks)
    interpolated_quality = np.interp(i_interpolated, i_quality, quality)

    return MaterialDeposition(
        material=Material.from_data(
            pd.DataFrame(
                {
                    "timestamp": np.linspace(0, max_timestamp, len(interpolated_quality)),
                    "volume": [total_volume / len(interpolated_quality)] * len(interpolated_quality),
                    "quality": interpolated_quality,
                }
            )
        ),
        deposition=Deposition.from_data(
            data=pd.DataFrame(
                {
                    "timestamp": np.linspace(0, max_timestamp, len(x_position)),
                    "x": x_position,
                    "z": [0.5 * bed_size_z] * len(x_position),
                }
            ),
            bed_size_x=bed_size_x,
            bed_size_z=bed_size_z,
            reclaim_x_per_s=bed_size_x / max_timestamp,
        ),
    )


def bsl_stack_reclaim(material_deposition: MaterialDeposition) -> Material:
    """Helper function to run blending simulation on material deposition generating reclaimed material."""
    return BslBlendingSimulator(
        bed_size_x=material_deposition.deposition.meta.bed_size_x,
        bed_size_z=material_deposition.deposition.meta.bed_size_z,
    ).stack_reclaim(material_deposition)
