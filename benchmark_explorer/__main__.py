#!/usr/bin/env python

import argparse
import os
from typing import List

from benchmark_explorer import testlets
from benchmark_explorer.evaluation import Evaluation
from data_explorer import app
from data_explorer.testlet import Testlet
from simulator_benchmark.evaluate_benchmark_data import read_references, get_identifier


def read_evaluation(evaluation_path: str) -> Evaluation:
    evaluation = Evaluation(get_identifier(evaluation_path), evaluation_path)
    evaluation.references = read_references(evaluation_path)
    return evaluation


def main(args):
    standard = read_evaluation(args.standard)
    evaluations = [read_evaluation(s) for s in args.evaluations]

    static_testlets: List[Testlet] = [
        testlets.MaterialIdentifierTestlet(),
        testlets.DepositionIdentifierTestlet()
    ]

    dynamic_testlets: List[Testlet] = [
        testlets.CorrelationTestlet(e) for e in evaluations
    ]

    app.execute(
        path=os.path.abspath(args.path),
        entry_list=[s for _, s in standard.references.items()],
        testlet_list=static_testlets + dynamic_testlets,
        label='Benchmark Explorer',
        verbose=args.verbose
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark Explorer'
    )

    parser.add_argument('--path', default='.', type=str,
                        help='output path for images')
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
                        help='enables verbose logging')
    parser.add_argument('--standard', default='./standard', type=str,
                        help='standard which all other results are compared to')
    parser.add_argument('evaluations', type=str, nargs='+',
                        help='paths to evaluations which are compared against standard')

    main(parser.parse_args())
