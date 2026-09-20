# BMH Apps

BMH Apps

## Configuration

Scripts that need machine specific locations read them from environment variables instead of hardcoded paths:

| Variable | Used by | Meaning |
|---|---|---|
| `BMH_BENCHMARK_PATH` | `bmh_apps.benchmark.optimize_deposition`, `bmh_apps.benchmark.optimize_deposition_simple` | Directory of the simulator benchmark (materials, depositions, simulators) |
| `BMH_EXPERIMENTS_PATH` | `bmh_apps.funvar.*` (`plot_fun`, `export_fun`, ...) | Root directory searched for experiment directories, so experiments can be given by id (e.g. `E1234abcd`) instead of a path |

## Comparing optimization runs

`plot_eaf` compares groups of independent optimization runs with empirical attainment functions (EAFs). Runs are grouped by their varying parameters, so for example the experiments with random and with preconditioned start populations can be given by id or path:

```shell
plot_eaf E1234abcd E5678cdef
```

It exports two plots: the best, median and worst attainment surface of each group and, for exactly two groups, the EAF difference. Use `--drop-columns` to reduce more than two objectives to two, `--compare A B` to select two of several groups (the group names are the parameters, e.g. `precondition=true`) `--intervals` to change the resolution of the difference and `--labels` to give the groups readable names in the legend, e.g. `--labels random preconditioned` (in the order of the legend, that is the order of `--compare` or otherwise alphabetical). If the groups have different numbers of runs, the first runs (by path) of both groups are compared.

