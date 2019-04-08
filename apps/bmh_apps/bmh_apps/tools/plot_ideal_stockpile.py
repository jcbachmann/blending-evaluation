#!/usr/bin/env python
import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
from bmh.helpers.stockpile_math import get_stockpile_volume, get_stockpile_height, get_stockpile_slice_volume


def plot_ideal_stockpile(x_min: float, core_length: float, height: float):
    logger = logging.getLogger(__name__)

    slices = 10000
    bed_size_x = core_length + 2 * x_min
    x_diff = bed_size_x / (slices - 1)  # -1 required, linspace includes start and end!
    x = np.linspace(0.0, bed_size_x, slices)

    volumes = [get_stockpile_slice_volume(f, core_length, height, x_min, x_diff) for f in x]

    logger.info('Bed Size X: %.1f m', bed_size_x)
    logger.info('Height: %.1f m', height)
    logger.info('Core Length: %.1f m', core_length)
    logger.info('Total Volume: %.1f m³', sum(volumes))
    logger.info('Computed Volume: %.1f m³', get_stockpile_volume(height, core_length))
    logger.info('Computed Height: %.1f m³', get_stockpile_height(sum(volumes), core_length))

    plt.plot(x, volumes, label=f'Volume per slice (x diff {x_diff:.02f}m)')

    plt.xlabel('Reclaimer Position')
    plt.ylabel('Volume')
    plt.title('Ideal Stockpile')
    plt.legend()
    plt.show()


def main(args):
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s [%(module)s]: %(message)s'
    )

    plot_ideal_stockpile(args.x_min, args.core_length, args.height)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot ideal stockpile')
    parser.add_argument('--x_min', type=float, default=25, help='core length of the stockpile')
    parser.add_argument('--core_length', type=float, default=130, help='core length of the stockpile')
    parser.add_argument('--height', type=float, default=10, help='height of the stockpile')
    parser.add_argument('--verbose', '-v', type=bool, default=False, help='enable verbose logging')

    main(parser.parse_args())
