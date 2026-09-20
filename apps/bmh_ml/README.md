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
uv run --package bmh_ml python -m bmh_ml.generate_training_data          # data/training_data.csv
uv run --package bmh_ml python -m bmh_ml.plot_training_data              # optional: F1/F2 scatter of the training data
uv run --package bmh_ml python -m bmh_ml.train_lstm_model                # data/scaler.pkl, data/lstm_model_f{1,2}_random_training_data.keras
uv run --package bmh_ml python -m bmh_ml.evaluate_model                  # LSTM predictions vs. simulator
uv run --package bmh_ml python -m bmh_ml.optimize_model                  # NSGA-III on the LSTM, writes output/lstm_model/
uv run --package bmh_ml python -m bmh_ml.optimize_simulation             # NSGA-III on the simulator, writes output/simulation/
uv run --package bmh_ml python -m bmh_ml.transform_optimization_results  # recompute each result with the other model and plot
uv run --package bmh_ml python -m bmh_ml.plot_optimization_results 'output/simulation/*.json'
```

`transform_optimization_results` exports images with Plotly's `kaleido`, which needs a Chrome installation.

The optimization scripts run a Latin hypercube design of algorithm settings: `--runs` (default 30), `--evaluations` (default 20000 and 100000) and `--population-sizes` (default 50, 100 and 200). A quick test is possible with `--runs 4 --evaluations 400 1000 --population-sizes 10 20 40`. The simulation is evaluated in parallel by one process pool for all runs. `train_lstm_model` takes `--epochs` (default 100).

The training data is checked when it is loaded: a truncated last row (e.g. from an interrupted or size limited write) is ignored with a warning, other incomplete rows are an error, and a warning is logged if the file has fewer rows than `training_data_params.json` says were generated.

