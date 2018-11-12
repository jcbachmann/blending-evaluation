import argparse

import matplotlib.pyplot as plt

from bmh_apps.roundness.roundness_evaluator import RoundnessEvaluator


def main(args):
    i = 0
    for input_file in args.input_files:
        e = RoundnessEvaluator(args.dist_seg_size, args.angle_seg_count)
        e.add_from_file(input_file)
        e.plot(input_file[:-4], i / (len(args.input_files) - 1) if i > 0 else 0)
        i += 1

    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate height map roundness from height map input files')
    parser.add_argument('input_files', type=str, help='Height map file', nargs='+')
    parser.add_argument('--dist_seg_size', type=float, default=5.0, help='Size of a single distance segment')
    parser.add_argument('--angle_seg_count', type=int, default=9, help='Amount of angle segments in range 0-45°')

    main(parser.parse_args())
