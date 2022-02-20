import argparse
import logging

import plotly.express as px

from .fun_var_results import FunVarResults


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    results = FunVarResults.from_files(args.filename)
    df = results.df.melt(
        id_vars=results.misc_columns,
        value_vars=results.var_columns,
        var_name='var',
        value_name='value'
    )

    fig = px.line(df, x='var', y='value', line_group='run_individual', color='run')
    fig.update_traces(opacity=0.05)
    fig.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, nargs='+')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose logging')
    return parser.parse_args()


if __name__ == '__main__':
    main(get_args())
