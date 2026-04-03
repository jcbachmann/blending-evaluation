#!/bin/bash
source ../../../../.venv/bin/activate
python -m bmh_apps.benchmark.optimize_deposition -m '+run=range(33)' '+experiment=mining-f1-f2-f4' optimization.max_evaluations=1000000 optimization.precondition_population=true

