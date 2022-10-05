#!/bin/bash
source ../../../../.venv/bin/activate
python -m bmh_apps.benchmark.optimize_deposition -m '+run=range(5)' '+experiment=mining-f1-f1-f2' optimization.max_evaluations=1000000 optimization.population_size=500 optimization.offspring_size=30 optimization.variable_count='range(10,310,15)'

