#!/usr/bin/env python
import logging
import os

from bmh.benchmark.data import BenchmarkData
from bmh.benchmark.material_deposition import DepositionMeta
from bmh.optimization.optimization import DepositionOptimizer

from bmh_apps.helpers.bed_size import get_bed_size


def main():
    logging.basicConfig(level=logging.DEBUG)

    benchmark_path = os.environ.get("BMH_BENCHMARK_PATH")
    if not benchmark_path:
        raise SystemExit("Environment variable BMH_BENCHMARK_PATH must point to the benchmark directory")

    benchmark = BenchmarkData(base_path=benchmark_path)
    benchmark.read_base()
    material_meta = benchmark.get_material_meta("scenario-mining-aae0")
    material = material_meta.get_material()
    bed_size_x, bed_size_z = get_bed_size(volume=material_meta.volume)
    deposition_meta = DepositionMeta.create_empty(
        bed_size_x=bed_size_x,
        bed_size_z=bed_size_z,
        reclaim_x_per_s=1.0,
    )
    optimizer = DepositionOptimizer(
        deposition_meta=deposition_meta,
        x_min=0.5 * bed_size_z,
        x_max=bed_size_x - 0.5 * bed_size_z,
        population_size=500,
        max_evaluations=20000,
        offspring_size=30,
        v_max=1.0,
        parameter_labels=material.get_parameter_columns(),
        objectives=["F1/Ash (%)", "F1/Sulphur (%)", "F2"],
    )
    optimizer.run(
        material=material,
        variables=100,
        population_generator=None,
    )


if __name__ == "__main__":
    main()
