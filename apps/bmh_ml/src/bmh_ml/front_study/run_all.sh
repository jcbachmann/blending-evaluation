#!/usr/bin/env bash
# Recomputes both charts of the front study for one T6 material of bundle S2-v1 (default: index 2, the third material): every step,
# both pages and the figure, into $BMH_ML_FRONT_STUDY_OUT (default workdir/front-interpolation/out). About 1.5 hours on 14 cores
# (i9-9900K): verify 2 min, interpolate 6 min, refine 17 min, resolution 12 min, deepen 56 min, confirm_beyond 2 min, time_levels 1 min.
# Every simulation step runs under watchdog.sh, which logs the CPU temperature to workdir/logs/ and stops a step at 95 °C or more.
# Usage (from anywhere): ENV=/path/to/env apps/bmh_ml/src/bmh_ml/front_study/run_all.sh [material_index]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
MATERIAL="${1:-2}"
: "${ENV:?set ENV to an environment made with: UV_PROJECT_ENVIRONMENT=\$ENV uv sync --locked --all-packages (never the repository .venv)}"
export BMH_ML_STORE="${BMH_ML_STORE:-$REPO/workdir/bmh-ml-store}"
cd "$REPO"
OUT="${BMH_ML_FRONT_STUDY_OUT:-workdir/front-interpolation/out}"
# a full run starts these from scratch; refine, resolution and deepen alone resume where they were interrupted
rm -f "$OUT/refined-material-$MATERIAL.npz" "$OUT/clouds-material-$MATERIAL.npz" "$OUT/deep-material-$MATERIAL.npz"
mkdir -p workdir/logs
for step in verify interpolate refine resolution deepen confirm_beyond time_levels; do
    echo "== $step"
    "$HERE/watchdog.sh" "workdir/logs/temp-front-study-$step.log" "$ENV/bin/python" -m "bmh_ml.front_study.$step" "$MATERIAL"
done
for step in build_interpolation_page build_resolution_page plot_resolution_figure; do
    echo "== $step"
    "$ENV/bin/python" -m "bmh_ml.front_study.$step" "$MATERIAL"
done
