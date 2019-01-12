#!/usr/bin/env python

import argparse

from bmh.benchmark.data import BenchmarkData
from bmh.benchmark.material_deposition import DepositionMeta
from bmh.helpers.identifiers import get_identifier
from bmh.optimization.optimization import DepositionOptimizer

from ..helpers.bed_size import get_bed_size
from ..helpers.configure_logging import configure_logging


def main(path: str, material_identifier: str, verbose: bool):
    configure_logging(verbose=verbose)
    benchmark = BenchmarkData()
    benchmark.read_base(path)
    material_meta = benchmark.get_material_meta(material_identifier)
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
        population_size=300,
        max_evaluations=25000,
        offspring_size=30,
        v_max=0.1
    )
    optimizer.run(
        material=material,
        variables=100 + 1
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Optimize material deposition for a given material curve'
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--path', default='.', help='Simulator benchmark path')
    parser.add_argument('--material', type=str, default='generated_2Y45', help='Material curve identifier')
    args = parser.parse_args()

    main(path=args.path, material_identifier=get_identifier(args.material), verbose=args.verbose)
