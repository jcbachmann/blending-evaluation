#!/usr/bin/env python
import argparse

from ..helpers.material_path_io import read_material, read_path, print_merged_material_path, merge_material_path


def main(args):
    material_path = merge_material_path(
        length=args.length,
        depth=args.depth,
        material=read_material(args.material),
        path=read_path(args.path),
    )
    print_merged_material_path(material_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='height map evaluator')
    parser.add_argument('--length', type=float, default=300, help='Blending bed length')
    parser.add_argument('--depth', type=float, default=50, help='Blending bed depth')
    parser.add_argument('--material', type=str, default='test_material.txt', help='Material input file')
    parser.add_argument('--path', type=str, default='test_path.txt', help='Stacker traverse path file (normalized 0-1)')

    main(parser.parse_args())
