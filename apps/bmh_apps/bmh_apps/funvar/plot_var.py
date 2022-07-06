import argparse
import logging

import plotly.express as px

from .fun_var_results import FunVarResults


def get_opacity(count: int) -> float:
    offset = -5
    factor = 6
    power = 0.8
    return 0.05 + min(factor * pow(1 / max(count - offset, 1), power), 0.95)


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    results = FunVarResults.from_files(args.filename)
    if results.len() == 0:
        logging.warning('No results found')
        return

    df = results.df.melt(
        id_vars=results.misc_columns,
        value_vars=results.var_columns,
        var_name='var',
        value_name='value'
    )

    fig = px.line(df, x='var', y='value', line_group='run_individual', color='run')
    fig.update_traces(opacity=get_opacity(results.len()))
    fig.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, nargs='+')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose logging')
    return parser.parse_args()


if __name__ == '__main__':
    main(get_args())
