import argparse
import logging

import plotly.express as px

from bmh_ml.training_data import load_training_data


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    logging.info("Loading training data")
    training_data = load_training_data()
    logging.info("Training data loaded")

    fig = px.scatter(training_data, x="f1", y="f2")
    fig.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
