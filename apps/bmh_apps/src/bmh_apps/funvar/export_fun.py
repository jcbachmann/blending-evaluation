import argparse
import logging
import os
import re
import uuid

from bmh_apps.funvar.fun_var_math import filter_efficient_front
from bmh_apps.funvar.fun_var_results import FunVarResults, get_filename_without_extension


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, nargs="+")
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--non-dominated", action="store_true", help="Show only non-dominated solutions")
    parser.add_argument(
        "--skip-random-suffix", action="store_true", help="Skip adding a random suffix to the output file name, an existing file is overwritten"
    )
    return parser.parse_args()


def get_file_name(label: str, suffix: str) -> str:
    # The label has full control over the name, a separator is only needed when a random suffix follows it
    name = " ".join(part for part in (label, suffix) if part)
    # A path separator in the label would place the file in another directory
    return re.sub(r"[/\\]", "_", name) + ".FUN"


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
    fun_path = os.path.join(common_path, get_file_name(label, suffix))
    obj_path = get_filename_without_extension(fun_path) + "OBJ"

    logging.info(f"Exporting {len(results.df)} FUN values to {fun_path}")
    # No header, as jMetal reads FUN files as plain numbers (e.g. as reference front). The objective names are stored in the OBJ file next to it.
    results.df[results.fun_columns].to_csv(fun_path, index=False, header=False, sep=" ")
    with open(obj_path, "w") as f:
        f.write(f"{results.fun_columns}\n")


if __name__ == "__main__":
    main()
