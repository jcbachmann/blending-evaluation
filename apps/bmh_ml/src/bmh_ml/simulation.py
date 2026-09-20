import numpy as np
import pandas as pd
from bmh.benchmark.material_deposition import Deposition, Material, MaterialDeposition
from bmh.helpers.reclaimed_material_evaluator import ReclaimedMaterialEvaluator
from bmh.simulation.bsl_blending_simulator import BslBlendingSimulator


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

    evaluator = ReclaimedMaterialEvaluator(reclaimed=reclaimed_material, x_min=x_min, x_max=x_max)
    f1 = evaluator.get_single_parameter_stdev("quality")
    f2 = evaluator.get_volume_stdev()

    return f1, f2
