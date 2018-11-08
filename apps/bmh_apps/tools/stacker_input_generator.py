#!/usr/bin/env python
import argparse
import math
import signal
import sys


class Generator:
    finish = False

    def __init__(self, args):
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # Blending bed parameters
        self.length = args.length
        self.depth = args.depth

        # Material flow
        self.m3_per_second = 1.5

        # Material parameter output period
        self.t_diff = 1

        # Only 75% of the available depth should be used in average
        self.depth_fill_factor = 0.75

        self.circular = args.circular

    def run_linear(self):
        # Stacker path parameters
        min_pos = self.depth / 2
        max_pos = self.length - self.depth / 2

        # Total volume in cubic meters
        m3_center = (max_pos - min_pos) * pow(self.depth_fill_factor * self.depth / 2, 2)
        m3_cone = math.pi * pow(self.depth_fill_factor * self.depth / 2, 3) / 3
        m3_total = m3_center + m3_cone

        pos_period = 2111
        t = 0
        while not self.finish and m3_total > 0:
            # Position
            distance = (t % pos_period) / pos_period
            if int(t / pos_period) % 2:
                distance = 1 - distance
            x = distance * (max_pos - min_pos) + min_pos
            z = self.depth / 2

            # Mixture
            red_part = 0.5 + 0.5 * math.sin((t * 0.001) * 2 * math.pi)
            blue_part = 0.5 + 0.5 * math.sin((t * 0.001 + 0.333) * 2 * math.pi)
            yellow_part = 0.5 + 0.5 * math.sin((t * 0.001 + 0.666) * 2 * math.pi)
            m3_this_time = min(self.m3_per_second * self.t_diff, m3_total)

            if red_part > blue_part:
                if red_part > yellow_part:
                    sys.stdout.write(f'{t} {x} {z} {m3_this_time} 1 0 0\n')
                else:
                    sys.stdout.write(f'{t} {x} {z} {m3_this_time} 0 0 1\n')
            else:
                if blue_part > yellow_part:
                    sys.stdout.write(f'{t} {x} {z} {m3_this_time} 0 1 0\n')
                else:
                    sys.stdout.write(f'{t} {x} {z} {m3_this_time} 0 0 1\n')

            t += self.t_diff
            m3_total -= m3_this_time

        sys.stdout.close()

    def run_circular(self):
        # Stacker path parameters
        l = min(self.length, self.depth) / 2
        r = self.depth_fill_factor * l / 2

        a_min = 40
        a_max = 300

        # Total volume in cubic meters
        m3_total = math.pi / 3 * (pow(0.5 * l + r, 3) - 2 * pow(0.5 * l, 3) + pow(0.5 * l - r, 3))
        m3_total *= (a_max - a_min) / 360
        m3_pile = 0.9 * math.pi / 3 * pow(r, 3)
        m3_afterpile = m3_total - m3_pile

        sub_period = 2111
        a_sub = 50
        t_mat = 0
        t_sub = 0
        m3_stacked = 0

        while not self.finish and m3_stacked < m3_total:
            # Position
            sub = (t_sub % sub_period) / sub_period
            if int(t_sub / sub_period) % 2:
                sub = 1 - sub
            if m3_stacked > m3_pile:
                a = max(a_min,
                        min((m3_stacked - m3_pile) / m3_afterpile * (a_max - a_min - 20) + a_min + sub * a_sub, a_max))
            else:
                a = a_min
            r = a / 360 * 2 * math.pi - 0.5 * math.pi
            x = 0.5 * l * math.sin(r) + self.length / 2
            z = 0.5 * l * math.cos(r) + self.depth / 2

            # Mixture
            red_part = 0.5 + 0.5 * math.sin((t_mat * 0.001) * 2 * math.pi)
            blue_part = 0.5 + 0.5 * math.sin((t_mat * 0.001 + 0.333) * 2 * math.pi)
            yellow_part = 0.5 + 0.5 * math.sin((t_mat * 0.001 + 0.666) * 2 * math.pi)
            m3_this_time = min(self.m3_per_second * self.t_diff, m3_total - m3_stacked)

            if red_part > blue_part:
                if red_part > yellow_part:
                    sys.stdout.write(f'{t_mat} {x} {z} {m3_this_time} 1 0 0\n')
                else:
                    sys.stdout.write(f'{t_mat} {x} {z} {m3_this_time} 0 0 1\n')
            else:
                if blue_part > yellow_part:
                    sys.stdout.write(f'{t_mat} {x} {z} {m3_this_time} 0 1 0\n')
                else:
                    sys.stdout.write(f'{t_mat} {x} {z} {m3_this_time} 0 0 1\n')

            t_mat += self.t_diff
            if m3_stacked > m3_pile:
                t_sub += self.t_diff
            m3_stacked += m3_this_time

        sys.stdout.close()

    def run(self):
        self.status('Starting generator')
        try:
            if self.circular:
                self.run_circular()
            else:
                self.run_linear()
        except IOError:
            self.status('Stopping generator due to IOError')
            self.finish = True
        self.status('Generator stopped')

    @staticmethod
    def status(msg):
        print("[generator] " + msg, file=sys.stderr)

    def signal_handler(self, _signum, _frame):
        self.status('Stopping generator')
        self.finish = True


def main(args):
    Generator(args).run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='height map evaluator')
    parser.add_argument('--length', type=float, required=True, help='Blending bed length')
    parser.add_argument('--depth', type=float, required=True, help='Blending bed depth')
    parser.add_argument('--circular', action='store_true', help='Generate output for circular blending bed')

    main(parser.parse_args())
