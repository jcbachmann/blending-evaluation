import argparse
import logging

import pandas as pd
import plotly.express as px


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    logging.info("Loading training data")
    training_data = pd.read_csv("data/training_data.csv")
    logging.info("Training data loaded")

    fig = px.scatter(training_data, x="f1", y="f2")
    fig.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
