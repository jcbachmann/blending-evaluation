import argparse
import logging
import math
import time
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly import express as px
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .fun_var_math import filter_efficient_front
from .fun_var_results import FunVarResults


def test_cluster_counts(df: pd.DataFrame):
    model = KMeans()

    def get_silhouette_score(n_clusters):
        logging.debug(f'Computing silhouette score for {n_clusters} clusters')
        model.set_params(n_clusters=n_clusters)
        model.fit(df)
        score = metrics.silhouette_score(df, model.labels_)
        logging.info(f'Silhouette score for {n_clusters} clusters: {score}')
        return score

    cluster_counts = range(2, math.ceil(df.shape[0] / 2.5), math.ceil(df.shape[0] / 2.5 / 30))
    silhouette_scores = [get_silhouette_score(cluster_count) for cluster_count in cluster_counts]
    fig = px.bar(
        x=cluster_counts,
        y=silhouette_scores,
        title='Silhouette Score vs. Cluster Count',
        labels=dict(x='Cluster Count', y='Silhouette Score')
    )
    fig.show()

    optimal_cluster_count = max(zip(silhouette_scores, cluster_counts))[1]
    return optimal_cluster_count


def test_pca(df: pd.DataFrame):
    def get_explained_variation(n: int):
        logging.debug(f'Computing PCA with {n} components')
        pca = PCA(n_components=n)
        pca.fit_transform(df)
        cum_explained_variation = np.sum(pca.explained_variance_ratio_)
        logging.info(f'Cumulative explained variation for {n} components: {cum_explained_variation}')
        return cum_explained_variation

    n_options = list(range(1, min(df.shape[0], df.shape[1])))
    explained_variations = [get_explained_variation(n) for n in n_options]
    fig = px.bar(
        x=n_options,
        y=explained_variations,
        title='Cumulative Explained Variation vs. Number of Components',
        labels=dict(x='Number of Components', y='Cumulative Explained Variation')
    )
    fig.show()


def pca_2_components(df: pd.DataFrame):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df)
    logging.info(
        f'PCA for 2 components results in cumulative variance explained of: '
        f'{np.sum(pca.explained_variance_ratio_)}'
    )
    return pca, pca_result


def do_kmeans(df: pd.DataFrame, cluster_count: int, title: str):
    kmeans = KMeans(n_clusters=cluster_count)
    kmeans.fit(df)
    centroids = kmeans.cluster_centers_

    df = df.copy()
    df['solution'] = df.index
    df['cluster'] = kmeans.labels_
    df = df.melt(
        id_vars=['solution', 'cluster'],
        var_name='var',
        value_name='value'
    )

    fig = px.line(df, x='var', y='value', line_group='solution', color='cluster')
    fig.update_layout(
        title=title if title else 'Clustered Solutions',
        xaxis_title='Variable',
        yaxis_title='Position',
    )
    fig.write_html(f"{title or 'clustered-solutions'}-{int(time.time())}.html")
    fig.show()

    df = pd.DataFrame(centroids)
    df['cluster'] = df.index
    df = df.melt(
        id_vars='cluster',
        var_name='var',
        value_name='value'
    )
    fig = px.line(df, x='var', y='value', color='cluster')
    fig.update_layout(
        title=f'Centroids for Clustered Solutions',
        xaxis_title='Variable',
        yaxis_title='Position',
    )
    fig.show()

    return centroids, kmeans.labels_


def plot_clusters(cluster_data, centroids, labels, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cluster_data[:, 0],
        y=cluster_data[:, 1],
        mode='markers',
        name='Results',
        text=labels
    ))
    fig.add_trace(go.Scatter(
        x=centroids[:, 0],
        y=centroids[:, 1],
        mode='markers+text',
        name='Centroids',
        marker=dict(color='red'),
        text=list(range(centroids.shape[0])),
        textposition='top center',
    ))
    fig.update_layout(
        title=title if title else 'Clusters',
    )
    fig.write_html(f"{title or 'clusters'}-{int(time.time())}.html")
    fig.show()


def filter_data(df: pd.DataFrame, fun_columns: List[str]):
    df = filter_efficient_front(df, fun_columns)
    logging.info(f'Efficient front has {df.shape[0]} entries')

    return df


def cluster_in_variable_space(df: pd.DataFrame, results: FunVarResults):
    # Prepare data for clustering
    df = df[results.var_columns].reset_index(drop=True)

    test_pca(df)
    optimal_cluster_count = test_cluster_counts(df)
    centroids, kmeans_labels = do_kmeans(
        df,
        cluster_count=optimal_cluster_count,
        title='clustered-solutions-in-variable-space'
    )

    # Visualize clustering
    pca, pca_result = pca_2_components(df)
    centroids_pca = pca.transform(centroids)
    plot_clusters(
        cluster_data=pca_result,
        centroids=centroids_pca,
        labels=kmeans_labels,
        title='clusters-in-variable-space'
    )


def cluster_in_objective_space(df: pd.DataFrame, results: FunVarResults):
    # Prepare data for clustering
    df_fun = df[results.fun_columns[:3]].reset_index(drop=True)

    centroids, kmeans_labels = do_kmeans(df_fun, cluster_count=5, title='clustered-solutions-in-objective-space')

    fig = px.scatter_3d(
        df_fun,
        x=results.fun_columns[0],
        y=results.fun_columns[1],
        z=results.fun_columns[2],
        color=kmeans_labels,
    )
    fig.add_trace(go.Scatter3d(
        x=centroids[:, 0],
        y=centroids[:, 1],
        z=centroids[:, 2],
        mode='markers+text',
        name='Centroids',
        marker=dict(color='red'),
        text=list(range(centroids.shape[0])),
        textposition='top center',
    ))
    fig.update_layout(
        title='Clusters in Objective Space',
    )
    fig.write_html(f"clusters-in-objective-space-{int(time.time())}.html")
    fig.show()

    df = df[results.var_columns].reset_index(drop=True)
    df['solution'] = df.index
    df['cluster'] = kmeans_labels
    df = df.melt(
        id_vars=['solution', 'cluster'],
        var_name='var',
        value_name='value'
    )

    fig = px.line(df, x='var', y='value', line_group='solution', color='cluster')
    fig.update_layout(
        title=f'Solutions Clustered in Objective Space',
        xaxis_title='Variable',
        yaxis_title='Position',
    )
    fig.write_html(f"solutions-clustered-in-objective-space-{int(time.time())}.html")
    fig.show()


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    results = FunVarResults.from_files(args.filename)
    df = filter_data(results.df, results.fun_columns)

    cluster_in_variable_space(df, results)
    cluster_in_objective_space(df, results)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, nargs='+')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose logging')
    return parser.parse_args()


if __name__ == '__main__':
    main(get_args())
