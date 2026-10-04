"""The fast blending simulator as the pipeline uses it, at a detail level (`SimulatorSettings`) that the bundles record."""

import numpy as np
import pandas as pd
from bmh.benchmark.material_deposition import Deposition, Material, MaterialDeposition
from bmh.helpers.reclaimed_material_evaluator import ReclaimedMaterialEvaluator
from bmh.simulation.bsl_blending_simulator import BslBlendingSimulator

from bmh_ml.settings import get_simulator_settings


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
    f1, f2, _ = evaluate_sim_with_profile(material_variables, deposition_variables, bed_size_x, bed_size_z, total_volume)
    return f1, f2


def evaluate_sim_with_profile(
    material_variables,
    deposition_variables,
    bed_size_x: float,
    bed_size_z: float,
    total_volume: float,
) -> tuple[float, float, np.ndarray]:
    """F1, F2 and the reclaimed profile they are computed from: the volume and the quality of each reclaimed slice, shape (2, slices)."""
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

    settings = get_simulator_settings()
    sim = BslBlendingSimulator(bed_size_x=bed_size_x, bed_size_z=bed_size_z, ppm3=settings.ppm3, reclaimincrement=settings.reclaim_increment)
    reclaimed_material = sim.stack_reclaim(material_deposition)

    evaluator = ReclaimedMaterialEvaluator(reclaimed=reclaimed_material, x_min=x_min, x_max=x_max)
    f1 = evaluator.get_single_parameter_stdev("quality")
    f2 = evaluator.get_volume_stdev()
    profile = np.vstack([reclaimed_material.data["volume"].to_numpy(), reclaimed_material.data["quality"].to_numpy()])
    return f1, f2, profile
