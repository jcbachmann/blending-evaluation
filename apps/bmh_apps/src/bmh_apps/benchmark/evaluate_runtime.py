import argparse
import logging
import re
from datetime import datetime
from glob import glob
from pathlib import Path

import pandas as pd
import plotly.express as px

TIMESTAMP_PATTERN = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)]")


def get_timestamp(match: re.Match[str]) -> datetime:
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")


def evaluate_log_file_runtime(file_path: str) -> float:
    """Runtime in hours between the first and the last log line that starts with a timestamp."""
    first_match = last_match = None
    with open(file_path) as f:
        for line in f:
            # Only match the lines here, parsing every timestamp would be slow for large logs
            match = TIMESTAMP_PATTERN.match(line)
            if match is None:
                continue  # e.g. continuation lines of multi-line messages
            if first_match is None:
                first_match = match
            last_match = match
    if first_match is None or last_match is None:
        raise ValueError(f"No log line with a timestamp found in {file_path}")
    return (get_timestamp(last_match) - get_timestamp(first_match)).total_seconds() / 3600


def parse_run_directory(directory: str) -> dict[str, str]:
    """Parse the name of a hydra sweep directory like `optimization.max_evaluations=1000,+run=3`."""
    return dict(assignment.split("=", 1) for assignment in directory.split(",") if "=" in assignment)


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    values = []

    for logfile_glob in args.logfile:
        for logfile in glob(logfile_glob):
            variables = parse_run_directory(Path(logfile).parent.name)
            if args.parameter not in variables or "+run" not in variables:
                logging.warning(f"Skipping {logfile}: directory name has no {args.parameter} and +run assignment")
                continue
            try:
                runtime = evaluate_log_file_runtime(logfile)
            except ValueError as e:
                logging.warning(f"Skipping {logfile}: {e}")
                continue
            values.append(
                {
                    args.parameter: float(variables[args.parameter]),
                    "Runtime": runtime,
                    "Run": variables["+run"],
                }
            )

    if not values:
        logging.error("No usable log files found")
        return

    runtimes = pd.DataFrame(values).sort_values(by=["Run"])

    fig = px.box(
        runtimes,
        x=args.parameter,
        y="Runtime",
        title=f"Influence of {args.parameter} on Runtime",
    )
    fig.update_layout(
        xaxis_title=args.parameter,
        yaxis_title="Runtime (h)",
        dragmode="select",
    )
    fig.show()


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate optimization over parameter runtime based on log files")
    parser.add_argument("logfile", type=str, nargs="+")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose logging")
    parser.add_argument("--parameter", type=str, required=True, help="Parameter name")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
