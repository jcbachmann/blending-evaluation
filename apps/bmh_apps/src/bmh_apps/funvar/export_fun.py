import argparse
import logging
import os
import uuid

from bmh_apps.funvar.fun_var_math import filter_efficient_front
from bmh_apps.funvar.fun_var_results import FunVarResults


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, nargs="+")
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--non-dominated", action="store_true", help="Show only non-dominated solutions")
    parser.add_argument("--skip-random-suffix", action="store_true", help="Skip adding a random suffix to the output file name")
    return parser.parse_args()


def get_file_name(label: str, suffix: str) -> str:
    # The label has full control over the name, a separator is only needed when a random suffix follows it
    return " ".join(part for part in (label, suffix) if part) + ".FUN"


def main(args: argparse.Namespace | None = None):
    if args is None:
        # Entry point of the export_fun script
        args = get_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    results = FunVarResults.from_files(args.filename, fun_only=True)
    label = args.label if args.label else results.label
    common_path = os.path.commonpath(results.df["file_path"].to_list())

    if args.non_dominated:
        results.df = filter_efficient_front(results.df, results.fun_columns)
        label = f"{label} (non-dominated)".strip()

    suffix = "" if args.skip_random_suffix else str(uuid.uuid4())[:4]
    file_path = os.path.join(common_path, get_file_name(label, suffix))

    logging.info(f"Exporting {len(results.df)} FUN values to {file_path}")
    results.df[results.fun_columns].to_csv(file_path, index=False, sep=" ")


if __name__ == "__main__":
    main()
