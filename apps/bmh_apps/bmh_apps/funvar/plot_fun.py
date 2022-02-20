import argparse
import logging

import plotly.express as px

from .fun_var_results import FunVarResults


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    results = FunVarResults.from_files(args.filename)

    if len(results.fun_columns) == 3:
        px.scatter_3d(
            results.df,
            x=results.fun_columns[0],
            y=results.fun_columns[1],
            z=results.fun_columns[2],
            color='run'
        ).show()
    elif len(results.fun_columns) == 2:
        px.scatter(
            results.df,
            x=results.fun_columns[0],
            y=results.fun_columns[1],
            color='run'
        ).show()
    else:
        raise Exception("Invalid number of columns")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, nargs='+')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose logging')
    return parser.parse_args()


if __name__ == '__main__':
    main(get_args())
