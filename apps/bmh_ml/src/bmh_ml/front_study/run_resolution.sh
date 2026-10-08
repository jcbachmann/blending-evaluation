#!/usr/bin/env bash
# Recomputes the resolution ladder alone for one T6 material of bundle S2-v1 (default: index 2): the page resolution-material-<i>.html,
# the figure resolution-figure-material-<i>.{pdf,svg,png} and their data, in $BMH_ML_FRONT_STUDY_OUT (default
# workdir/front-interpolation/out). About 15 minutes on 14 cores (i9-9900K): verify 2 min, resolution 12 min, time_levels 1 min (run it on
# an otherwise idle machine). Independent of the interpolation steps.
# Usage (from anywhere): ENV=/path/to/env apps/bmh_ml/src/bmh_ml/front_study/run_resolution.sh [material_index]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
MATERIAL="${1:-2}"
: "${ENV:?set ENV to an environment made with: UV_PROJECT_ENVIRONMENT=\$ENV uv sync --locked --all-packages (never the repository .venv)}"
export BMH_ML_STORE="${BMH_ML_STORE:-$REPO/workdir/bmh-ml-store}"
cd "$REPO"
OUT="${BMH_ML_FRONT_STUDY_OUT:-workdir/front-interpolation/out}"
rm -f "$OUT/clouds-material-$MATERIAL.npz"  # resolution alone resumes an interrupted run; a full run starts fresh
mkdir -p workdir/logs
for step in verify resolution time_levels; do
    echo "== $step"
    "$HERE/watchdog.sh" "workdir/logs/temp-front-study-$step.log" "$ENV/bin/python" -m "bmh_ml.front_study.$step" "$MATERIAL"
done
for step in build_resolution_page plot_resolution_figure; do
    echo "== $step"
    "$ENV/bin/python" -m "bmh_ml.front_study.$step" "$MATERIAL"
done
