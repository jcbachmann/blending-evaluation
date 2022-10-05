#!/bin/bash
source ../../../../.venv/bin/activate
python -m bmh_apps.benchmark.optimize_deposition -m '+run=range(31)' '+experiment=mining-f2' optimization.max_evaluations=1000000 optimization.population_size=500 optimization.offspring_size=30

