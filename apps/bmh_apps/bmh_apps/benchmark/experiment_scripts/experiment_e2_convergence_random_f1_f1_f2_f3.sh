#!/bin/bash
source ../../../../.venv/bin/activate
python -m bmh_apps.benchmark.optimize_deposition -m '+run=range(11)' '+experiment=mining-f1-f1-f2-f3' optimization.max_evaluations=10000000

