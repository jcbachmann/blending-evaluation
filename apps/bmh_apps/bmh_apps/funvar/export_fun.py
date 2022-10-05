import argparse
import logging
import os
import uuid

from .fun_var_math import filter_efficient_front
from .fun_var_results import FunVarResults


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    results = FunVarResults.from_files(args.filename, fun_only=True)
    label = results.label

    if args.non_dominated:
        results.df = filter_efficient_front(results.df, results.fun_columns)
        label += " (non-dominated)"

    file_path = os.path.join(
        os.path.commonpath(results.df['file_path'].to_list()),
        f"{label} {str(uuid.uuid4())[:4]}.FUN"
    )

    logging.info(f"Exporting {len(results.df)} FUN values to {file_path}")
    results.df[results.fun_columns].to_csv(file_path, index=False, sep=' ')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, nargs='+')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose logging')
    parser.add_argument('--non-dominated', action='store_true', default=False, help='Show only non-dominated solutions')
    return parser.parse_args()


if __name__ == '__main__':
    main(get_args())
