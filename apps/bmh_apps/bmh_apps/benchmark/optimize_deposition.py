#!/usr/bin/env python

import argparse
import math

from bmh.benchmark.data import BenchmarkData
from bmh.helpers.identifiers import get_identifier
from bmh.optimization.optimization import optimize_deposition

from bmh_apps.helpers.configure_logging import configure_logging


def get_bed_size(volume: float, max_stockpile_height: float = 20.0):
    max_volume = volume * 1.25
    core_length = (max_volume - math.pi * math.pow(max_stockpile_height, 3) / 3) / math.pow(max_stockpile_height, 2)
    bed_size_x = core_length + 2 * max_stockpile_height
    bed_size_z = 2 * max_stockpile_height
    return bed_size_x, bed_size_z


def main(path: str, material_identifier: str):
    configure_logging(verbose=False)
    benchmark = BenchmarkData()
    benchmark.read_base(path)
    material_meta = benchmark.get_material_meta(material_identifier)
    material = material_meta.get_material()
    bed_size_x, bed_size_z = get_bed_size(volume=material_meta.volume)
    optimize_deposition(
        bed_size_x=bed_size_x,
        bed_size_z=bed_size_z,
        material=material,
        variables=30 + 1,
        population_size=250,
        max_evaluations=25000
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Optimize material deposition for a given material curve'
    )
    parser.add_argument('--path', default='.', help='Simulator benchmark path')
    parser.add_argument('--material', type=str, default='generated_2Y45', help='Material curve identifier')
    args = parser.parse_args()

    main(path=args.path, material_identifier=get_identifier(args.material))
