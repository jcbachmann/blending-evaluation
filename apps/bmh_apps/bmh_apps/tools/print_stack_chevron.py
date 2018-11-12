#!/usr/bin/env python
import argparse

from bmh_apps.chevron.chevron_path import chevron_path
from bmh_apps.helpers.material_path_io import print_merged_material_path, read_material, merge_material_path


def main(args):
    material_path = merge_material_path(
        length=args.length,
        depth=args.depth,
        material=read_material(args.material),
        path=chevron_path(args.layers),
    )
    print_merged_material_path(material_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='chevron stacker')
    parser.add_argument('--length', type=float, default=300, help='Blending bed length')
    parser.add_argument('--depth', type=float, default=50, help='Blending bed depth')
    parser.add_argument('--material', type=str, default='test_material.txt', help='Material input file')
    parser.add_argument('--layers', type=float, default=5, help='Amount of layers placed using Chevron stacking')

    main(parser.parse_args())
