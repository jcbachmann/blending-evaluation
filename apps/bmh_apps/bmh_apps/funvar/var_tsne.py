import argparse
import logging

import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE

from .fun_var_math import filter_relevant_region, filter_efficient_front
from .fun_var_results import FunVarResults


def filter_data(df: pd.DataFrame, fun_columns: list[str]):
    df = filter_relevant_region(df, fun_columns)
    logging.info(f'Region filtered dataframe has {df.shape[0]} rows')

    df = filter_efficient_front(df, fun_columns)
    logging.info(f'Efficient front has {df.shape[0]} entries')

    return df


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    results = FunVarResults.from_files(args.filename)
    df = filter_data(results.df, results.fun_columns)

    # Prepare data for clustering
    df = df[results.var_columns].reset_index(drop=True)

    tsne = TSNE()
    tsne_results = tsne.fit_transform(df)
    print(tsne.get_params())
    print(tsne_results)

    df['tsne-1'] = tsne_results[:, 0]
    df['tsne-2'] = tsne_results[:, 1]
    fig = px.scatter(df, x='tsne-1', y='tsne-2', color='cluster', title='TSNE')
    fig.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, nargs='+')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose logging')
    return parser.parse_args()


if __name__ == '__main__':
    main(get_args())
