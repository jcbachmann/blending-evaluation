# BMH ML

Machine learning experiments for the blending simulator: an LSTM surrogate model of the simulator objectives (F1 homogenization, F2 reclaim volume deviation), compared with optimization on the real simulator.

Installing this package pulls in TensorFlow, so it is not part of the default workspace install. To install all workspace packages including this one:

```shell
uv sync --all-packages
```

`uv sync --package bmh_ml` would remove the other packages from the environment. The commands below use `uv run --package bmh_ml`, which installs what the scripts need when they are started.

## Usage

The scripts read and write `data/` and `output/` relative to the working directory (both are git-ignored), so run all of them from the same directory. Any directory works, for example one outside of the repository: `uv run --project <path of the repository> --package bmh_ml python -m bmh_ml.<script>`. The typical order is:

```shell
uv run --package bmh_ml python -m bmh_ml.generate_training_data                              # data/training_data.csv
uv run --package bmh_ml python -m bmh_ml.plot_training_data                                  # optional: F1/F2 scatter of the training data
uv run --package bmh_ml python -m bmh_ml.train_lstm_model --model-set random_training_data   # data/scaler_<set>.pkl, data/lstm_model_f{1,2}_<set>.keras
uv run --package bmh_ml python -m bmh_ml.evaluate_model                                      # LSTM predictions vs. simulator, prints R2 and the error
uv run --package bmh_ml python -m bmh_ml.optimize_model                                      # NSGA-III on the LSTM, writes output/lstm_model/
uv run --package bmh_ml python -m bmh_ml.optimize_simulation                                 # NSGA-III on the simulator, writes output/simulation/
uv run --package bmh_ml python -m bmh_ml.transform_optimization_results                      # recompute each result with the other model and plot
uv run --package bmh_ml python -m bmh_ml.plot_optimization_results 'output/simulation/*.json'
```

`transform_optimization_results` exports images with Plotly's `kaleido`, which needs a Chrome installation.

## Model sets

The LSTM models are stored in **model sets**: the scaler and the models for F1 and F2 that were trained together on one dataset, in
`data/scaler_<set>.pkl` and `data/lstm_model_f{1,2}_<set>.keras`. Several sets can exist next to each other, for example one trained on a
dataset where every row has its own random material and deposition and one trained on a dataset with 50 depositions per material.
The scaler belongs to the models it was fitted with, so it is part of the set.

* `train_lstm_model` needs `--model-set` and `--training-data` (default `data/training_data.csv`). The name is required on purpose, so
  a set is never trained or overwritten by accident and models are never stored under the name of another dataset.
* The scripts that use models (`optimize_model`, `evaluate_model`, `transform_optimization_results`) take `--model-set`. Its default is
  `random_training_data`, the set of the experiments made so far (`DEFAULT_MODEL_SET` in `settings.py`). This is a deliberate choice, not a leftover.
  Results on the surrogate record the set they were made with.

The file `training_data_random.csv` of the first experiments has other column names (`y1`, `y2`, `x1` ... `x70`) than the current data
(`f1`, `f2`, `m1` ... `m50`, `d1` ... `d20`). It can select the material (`--training-data`, the columns are used by position), but
`train_lstm_model` cannot read it.

## Design of the optimization experiments

`optimize_model` and `optimize_simulation` optimize the deposition for **one fixed material**: the first row of the training data
(`--training-data`). This is deliberate. The material is not varied, the runs are **replicates** that repeat the optimization with
different random seeds, to cover the randomness of the algorithm (and, for the simulation, of the simulation itself).

* The settings of the runs, the number of evaluations and the population size, come from a Latin hypercube design (`--runs`,
  `--evaluations`, `--population-sizes`). The replicate number counts the runs with the same settings. As the design draws the settings,
  their number of replicates varies (3 to 7 with the defaults).
* The seed of a run is the design seed plus the run id. The optimization on the surrogate gives the same result again for the same
  seed. The simulation is random itself, so its results differ slightly from run to run.
* A result file `output/<model>/run_<id>_evals<n>_pop<n>_replicate<n>.json` contains the fronts (`objectives`, `variables`), the fixed
  `material_variables`, and the `parameters` of the run (`run_id`, `replicate`, `population_size`, `n_evaluations`, and the `model_set`
  for results on the surrogate). Older result files (`instance_<n>_run_<n>_pop<n>.json`) have an `instance_id` that is the same as a
  replicate number and do not contain the material of the simulation results.
* `transform_optimization_results` recomputes every result for the material it was made for. Older results of the simulation do not
  know it, they are recomputed for the first row of `--training-data` and a warning says so. Its cache files (`*_recomputed_*.json`)
  are only used for the same material and model set.

## Other options

The optimization scripts accept `--runs 4 --evaluations 400 1000 --population-sizes 10 20 40` for a quick test. The simulation is evaluated
in parallel by one process pool for all runs. `train_lstm_model` takes `--epochs` (default 100).

The training data is checked when it is loaded: a truncated last row (e.g. from an interrupted or size limited write) is ignored with a warning, other incomplete rows are an error, and a warning is logged if the file has fewer rows than `training_data_params.json` says were generated.
