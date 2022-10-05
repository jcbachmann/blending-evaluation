#!/bin/bash
source ../../../../.venv/bin/activate
python -m bmh_apps.benchmark.optimize_deposition -m '+run=range(66)' '+experiment=mining-f1-2-f2' optimization.max_evaluations=1000000 optimization.precondition_population=false

