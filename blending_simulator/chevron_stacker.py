#!/usr/bin/env python
import argparse

import pandas as pd

from blending_simulator.stacker import stack_with_printer


def chevron_path(layers: float) -> pd.DataFrame:
    path = [[float(f), float(f % 2)] for f in range(0, int(layers) + 1)]
    if layers - int(layers) > 0:
        # incomplete layer
        path.append([layers, (layers - int(layers)) if int(layers) % 2 == 0 else (1 - (layers - int(layers)))])
    return pd.DataFrame(data=path, columns=['part', 'path'])


def main(args):
    stack_with_printer(
        length=args.length,
        depth=args.depth,
        material=args.material,
        stacker_path=chevron_path(args.layers)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='chevron stacker')
    parser.add_argument('--length', type=float, required=True, help='Blending bed length')
    parser.add_argument('--depth', type=float, required=True, help='Blending bed depth')
    parser.add_argument('--material', type=str, required=True, help='Material input file')
    parser.add_argument('--layers', type=float, required=True, help='Speed')

    main(parser.parse_args())
