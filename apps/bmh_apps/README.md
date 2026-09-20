# BMH Apps

BMH Apps

## Configuration

Scripts that need machine specific locations read them from environment variables instead of hardcoded paths:

| Variable | Used by | Meaning |
|---|---|---|
| `BMH_BENCHMARK_PATH` | `bmh_apps.benchmark.optimize_deposition_simple` | Directory of the simulator benchmark (materials, depositions, simulators) |
