#!/usr/bin/env python

import hydra
from omegaconf import DictConfig

from bmh.benchmark.data import BenchmarkData
from bmh.benchmark.material_deposition import DepositionMeta
from bmh.optimization.optimization import DepositionOptimizer
from bmh_apps.helpers.bed_size import get_bed_size


# FIXME I can't get structured config schema to work with multiprocessing
# @dataclass
# class OptimizationConfig:
#     population_size: int = 100
#     offspring_size: int = 10
#     max_evaluations: int = 25000
#     variable_count: int = 30
#     objectives: List[str] = MISSING


# @dataclass
# class Config:
#     benchmark_path: str = MISSING
#     material_identifier: str = MISSING
#     optimization: OptimizationConfig = OptimizationConfig()


# cs = ConfigStore.instance()
# cs.store(name="base_config", node=Config)
# cs.store(group="optimization", name="base_optimization_config", node=OptimizationConfig)

# cs.store(group="optimization/objectives", name="base_objective_config", node=ObjectiveConfig)


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    benchmark = BenchmarkData(cfg.benchmark_path)
    benchmark.read_base()
    material_meta = benchmark.get_material_meta(cfg.material_identifier)
    material = material_meta.get_material()
    bed_size_x, bed_size_z = get_bed_size(volume=material_meta.volume)
    x_min = 0.5 * bed_size_z
    x_max = bed_size_x - x_min
    deposition_meta = DepositionMeta.create_empty(
        bed_size_x=bed_size_x,
        bed_size_z=bed_size_z,
        reclaim_x_per_s=1.0,
    )
    optimizer = DepositionOptimizer(
        deposition_meta=deposition_meta,
        x_min=x_min,
        x_max=x_max,
        population_size=cfg.optimization.population_size,
        max_evaluations=cfg.optimization.max_evaluations,
        offspring_size=cfg.optimization.offspring_size,
        v_max=cfg.system.v_max,
        parameter_labels=material.get_parameter_columns(),
        plot_server_str=cfg.plot_server,
        ppm3=cfg.simulation.ppm3,
        objectives=cfg.optimization.objectives,
        reference_front_file=cfg.reference_front_file,
    )
    optimizer.run(
        material=material,
        variables=cfg.optimization.variable_count,
        # population_generator=MultiGenerator([
        # mixed starting population with standard strategies (Chevron), random solutions and memory
        # good start but no extraordinary solutions
        # (RandomGenerator(), 5),
        # (FullSpeedGenerator(), 1),
        # (RandomEndGenerator(), 5),
        # (FixedRandomSpeedGenerator(), 5),
        # (RandomSpeedGenerator(), 1)
        # ]),
    )


if __name__ == '__main__':
    main()
