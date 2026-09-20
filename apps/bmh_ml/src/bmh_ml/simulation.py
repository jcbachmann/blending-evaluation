import math

import numpy as np
import pandas as pd
from bmh.benchmark.material_deposition import Deposition, Material, MaterialDeposition
from bmh.helpers.math import stdev
from bmh.helpers.stockpile_math import get_stockpile_height, get_stockpile_slice_volume
from bmh.simulation.bsl_blending_simulator import BslBlendingSimulator


def weighted_std(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return math.sqrt(variance)


def get_volume_stdev(reclaimed: Material, x_max: float, x_min: float) -> float:
    ideal_df = reclaimed.data.copy()
    ideal_height = get_stockpile_height(volume=ideal_df["volume"].sum(), core_length=x_max - x_min)
    ideal_df["x_diff"] = (ideal_df["x"] - ideal_df["x"].shift(1)).fillna(0.0)
    ideal_df["volume"] = ideal_df.apply(
        lambda row: get_stockpile_slice_volume(
            x=row["x"],
            core_length=x_max - x_min,
            height=ideal_height,
            x_min=x_min,
            x_diff=row["x_diff"],
        ),
        axis=1,
    )

    return stdev((ideal_df["volume"] - reclaimed.data["volume"]).values)


def generate_material(
    material_variables: list[float],
    total_volume: float,
) -> Material:
    max_timestamp = 86400  # Local constant as it has no influence on the results
    return Material.from_data(
        pd.DataFrame(
            {
                "timestamp": np.linspace(0, max_timestamp, len(material_variables)),
                "volume": [total_volume / len(material_variables)] * len(material_variables),
                "quality": material_variables,
            }
        )
    )


def generate_deposition(
    deposition_variables: list[float],
    bed_size_x: float,
    bed_size_z: float,
) -> Deposition:
    max_timestamp = 86400  # Local constant as it has no influence on the results
    return Deposition.from_data(
        data=pd.DataFrame(
            {
                "timestamp": np.linspace(0, max_timestamp, len(deposition_variables)),
                "x": deposition_variables,
                "z": [0.5 * bed_size_z] * len(deposition_variables),
            }
        ),
        bed_size_x=bed_size_x,
        bed_size_z=bed_size_z,
        reclaim_x_per_s=bed_size_x / max_timestamp,
    )


def evaluate_sim(
    material_variables,
    deposition_variables,
    bed_size_x: float,
    bed_size_z: float,
    total_volume: float,
) -> tuple[float, float]:
    x_min = bed_size_z * 0.5
    x_max = bed_size_x - x_min

    material_deposition = MaterialDeposition(
        material=generate_material(
            material_variables,
            total_volume=total_volume,
        ),
        deposition=generate_deposition(
            deposition_variables,
            bed_size_x=bed_size_x,
            bed_size_z=bed_size_z,
        ),
    )

    sim = BslBlendingSimulator(bed_size_x=bed_size_x, bed_size_z=bed_size_z)
    reclaimed_material = sim.stack_reclaim(material_deposition)

    f1 = weighted_std(
        values=reclaimed_material.data["quality"],
        weights=reclaimed_material.data["volume"],
    )
    f2 = get_volume_stdev(
        reclaimed=reclaimed_material,
        x_min=x_min,
        x_max=x_max,
    )

    return f1, f2
