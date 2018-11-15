#!/usr/bin/env python
import argparse
import logging
import os

from data_explorer import app

from solution_explorer import solution, graphs, testlets


def main(args):
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s [%(module)s]: %(message)s'
    )

    solutions, meta = solution.read_solutions(args.directory)

    path_detail_canvas = graphs.PathDetailCanvas()

    app.execute(
        path=os.path.abspath(args.directory),
        entry_list=solutions,
        testlet_list=[
            testlets.ObjectiveTestlet(0, meta.objective_maximums[0], label='Quality Stdev'),
            testlets.ObjectiveTestlet(1, meta.objective_maximums[1], label='Volume Stdev'),
            testlets.PathDistanceTestlet(meta.variables_count),
            # testlets.PathGraphTestlet()
        ],
        main_figures=[
            graphs.ObjectivesScatterCanvas(meta.objectives, meta.all_variables, meta.all_objectives,
                                           path_detail_canvas.plot_selection),
            path_detail_canvas
        ],
        label=args.directory
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Homogenization Optimization Results Explorer')
    parser.add_argument('directory', type=str, help='path to solutions to be explored')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='log everything')

    main(parser.parse_args())
