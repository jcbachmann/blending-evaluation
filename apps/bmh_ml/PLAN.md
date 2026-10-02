# Plan: find the best models for F1 and F2

Status file for a multi-day effort. **If you resume this work (a new session, or after a break): read sections 0, 5 and 9 first.**
Results themselves are never kept in this file, they live in the experiment tracking store (section 4.5); this file holds the plan, the
decisions and short conclusions.

## 0. How to resume

1. Read section 9 (progress log), the first unchecked milestone in section 5 and the open proposals of section 10.
2. `git log --oneline -20` shows what was done since. The repository and the tracking store are the memory, not the chat.
3. Before declaring anything done, run the same checks as CI: `uv sync --locked --all-extras --dev`, `uv run pytest`, `uv run ruff check .`,
   `uv run ruff format --check .`, in a fresh environment outside the repository (`UV_PROJECT_ENVIRONMENT=/tmp/ci-env`), never in the
   repository's `.venv` (a lesson of this project: tests that only pass in a developer environment broke `master` once).
4. Since 2026-09-24 the store is `workdir/bmh-ml-store/` in the repository (moved with the file sync from the first machine), so set
   `BMH_ML_STORE` to it before running anything, or a new empty store starts at `~/bmh-ml-store`. After moving a store, run
   `workdir/fix-mlflow-store-paths.py` once: MLflow keeps absolute artifact paths in its database.

## 1. Goal and success criteria

**Goal:** models that predict the two objectives of the blending simulator, F1 (homogenization, standard deviation of the reclaimed
quality) and F2 (deviation of the reclaimed volume from the ideal stockpile), as well as reasonably possible, and a pipeline that
makes every attempt persisted and comparable without manual bookkeeping.

"As well as possible" is not one number. A model is judged on four things, reported side by side, and the final choice is made
with you on that table:

| Aspect | Question | Measured by |
|---|---|---|
| Accuracy where it matters | Is the model right in the region the optimizer visits, not just on average? | error on the *operating region* test set, relative to the simulator's own noise |
| Usefulness for optimization | Does optimizing the model give solutions that are good *in the simulator*? | transfer test: optimize the model, re-simulate the front, compare with the simulator-optimized front |
| Robustness | Does it avoid impossible or exploitable predictions? | share of negative predictions, bias in the tail of good solutions, stress test set |
| Cost | Is it worth having a model at all? | inference throughput compared with the simulator (section 2) |

**Two scopes**, tracked as separate tasks: **S1 fixed material** (input: the 20 deposition positions; this is what your experiments
optimize) and **S2 general material** (input: material curve of 50 values and deposition; what the current trainer does).

## 2. Facts (measured on 2026-09-20, this machine)

An ad hoc script produced these numbers; `python -m bmh_ml.build_bundle` now prints the same table for every bundle (M1.2). Treat the numbers here as the starting point, not as final.

**The simulator is noisy, which caps every model.** The same input simulated 30 times:

| Region | F1 noise sd | F1 spread sd | best possible R2 (F1) | F2 noise sd | F2 spread sd | best possible R2 (F2) |
|---|---|---|---|---|---|---|
| fixed material, random depositions | 0.0080 | 0.152 | 0.997 | 0.137 | 3.81 | 0.999 |
| fixed material, **optimized** depositions | 0.0093 | 0.035 | **0.929** | 0.249 | 3.51 | 0.995 |
| random material and deposition | 0.0071 | 0.208 | 0.999 | 0.145 | 4.46 | 0.999 |

So a model cannot be better than these ceilings, and in the region that matters, F1 differences below about 0.01 are noise. Test labels
should be averaged over repeated simulations, and errors are best read in units of the noise sd.

**The training data does not cover the region the optimizer works in.** Fixed material, simulator values:

| | F1 | F2 |
|---|---|---|
| random depositions (5 / 50 / 95 %) | 0.295 / 0.483 / 0.762 | 16.9 / 24.1 / 31.9 |
| your optimized fronts (min / median / max) | 0.051 / 0.126 / 0.466 | 2.9 / 4.4 / 24.3 |

96 % of the optimized solutions are below the 1st percentile of random data in F1, 96 % in F2, and **100 % in at least one**. The
surrogate optimizations you ran predict **negative F2 (impossible) for 77 % of their 2,450 front solutions**: the model extrapolates
exactly where it is used and the optimizer exploits its errors.

**The "LSTM" is a dense network.** Every sample is a sequence of length 1 (`reshape((n, 1, 70))`), so the LSTM sees one time step and
does no sequence modeling.

**A surrogate is currently not faster than the simulator.** One optimization run with 100,000 evaluations took 106 s on the LSTM
surrogate and 49 s on the simulator (16 cores). The simulator needs about 2 ms per evaluation. The surrogate has to earn its place by
batched inference speed, by being smooth and denoised, or by being differentiable, which is why cost is part of the criteria.

**Data is cheap.** 250,000 rows are generated in about 2 minutes (16 cores). The bottleneck is not the amount of data but where it is
sampled and which model is used.

**Hardware:** 16 CPU cores, 38 GB RAM, no GPU. Everything below must be reasonable on the CPU. Since 2026-09-24 the work continues on an
Intel i9-9900K (8 cores, 16 hardware threads), 62 GB RAM and an AMD Radeon RX 5600/5700 (Navi 10, gfx1010) GPU; the numbers above are
from the first machine. ROCm is not installed, so TensorFlow runs on the CPU; Navi 10 is not on AMD's official ROCm support list, so using
the GPU would need an unofficial setup (section 10).

## 3. Decisions

Made now, with the reason. Say so if one of them should be different; most are cheap to change early and expensive late.

| Decision | Choice | Why, and what else was considered |
|---|---|---|
| Experiment tracking and comparison UI | **MLflow** (local, SQLite backend, file artifacts) | Runs, parameters, metrics, artifacts and a comparison UI without an account; works with any framework; has a model registry. Verified here: installs on Python 3.12, the UI server starts. Weights & Biases is the most polished but is built around a cloud account (a self-hosted server exists but is heavier); TensorBoard shows and overlays curves but has no table of runs with parameters, artifacts and a model registry; Aim is a capable tracker with a smaller ecosystem and no registry; DVC focuses on data and pipeline versioning. |
| Where the store lives | `~/bmh-ml-store`, overridable by `BMH_ML_STORE` | Models, predictions and datasets grow to gigabytes; they must not sit in the synced folder. |
| Model frameworks | Keras 3 (already installed), scikit-learn, LightGBM. PyTorch only if an approach needs it. | The pipeline talks to models through one small interface, so a framework is an implementation detail and can be added later. |
| Configuration | Command line `--param key=value` overrides (values read as YAML) on the defaults of the model classes; no configuration framework | Changed from the first plan (OmegaConf configs): the defaults live next to the code that uses them, a run logs the complete parameters anyway, and sweeps (Optuna) are driven from Python. The Hydra runner would change the working directory and create output folders, which fights with a pipeline that has its own store. |
| Hyperparameter search | Optuna used directly, every trial a nested MLflow run | More control than a sweeper plugin, and the results end up in the same UI. |
| Labels | single simulations for training, **repeat-averaged (16x) for validation and test** | Training data is cheap, but noisy test labels would hide the differences between good models. |
| Test sets | **frozen and versioned**, model selection only on validation data | Otherwise many attempts overfit the test set unnoticed. |
| Dependencies and CI | heavy packages only in `bmh_ml`, tests behind `importorskip`; pure logic (metrics, splits, config) tested in CI | CI installs only the dev group; it must stay green on all four operating systems. |

**Decisions for you.** I proceed with the default if you say nothing:

1. **Scope priority.** Default: track S1 and S2, treat S1 as primary for the transfer test because your experiments use one material.
2. **What "best" means.** Default: no single score yet. I report the four aspects of section 1 and we choose the shortlist criterion together
   once the first baselines are in.
3. **Store location** `~/bmh-ml-store`, as above.
4. **New dependencies:** `mlflow`, `lightgbm`, `optuna` (later maybe `torch`). All are large, none is needed by the rest of the repository.
5. **Compute:** CPU only for now. The current machine has an AMD GPU (Navi 10) without ROCm; see section 10 for whether it is worth
   setting up.

## 4. Architecture of the pipeline

### 4.1 Layout (new modules in `bmh_ml`)

```
bmh_ml/
  datasets/     builders (random, operating region, structured, materials), manifest, splits, frozen test sets
  models/       one interface (fit, predict, save, load) and the model families
  evaluation/   metrics, noise ceiling, plots, transfer test, report
  tracking/     MLflow conventions: experiments, run naming, logging of config, data, git state, artifacts
  conf/         Hydra configs: dataset, model, training, sweep
  cli           train, evaluate, sweep, report, ui
```

### 4.2 Datasets and test sets

A dataset is generated from a **manifest** (generator, parameters, seed, simulator settings, code version) and stored with a content
hash in `<store>/datasets/<id>/`. The manifest makes it reproducible and every run records the dataset id.

Frozen test sets (each with denoised labels and its noise ceiling):

| Set | Content | Purpose |
|---|---|---|
| **T1** in distribution | random depositions (S1: the fixed material, S2: random materials) | the number people usually report |
| **T2** operating region | solutions of simulator-optimized fronts of runs never used for training | **the set that decides**: where the optimizer works |
| **T3** unseen materials | new random materials | generalization of S2 |
| **T4** realistic materials | scenario materials of the benchmark, resampled to 50 values | later; realism check |
| **T5** stress | extreme depositions (edges, constant positions, jumps) | robustness, exploitable errors |

Training data families (backlog B1, B3): random, **operating region** (from optimizer runs, refined in a loop), structured deposition
families (ramps, sawtooth, chevron-like, smooth random paths) and perturbations around good solutions.

### 4.3 Model interface and families

`fit(train, validation, config)`, `predict(x)`, `save`, `load`, plus `describe()` (parameters, size). Families: mean predictor and ridge
(sanity floor), LightGBM, MLP, the existing model as a reproduction baseline, then whatever the backlog produces. Outputs are always
positive (F1, F2 cannot be negative): see B2.

### 4.4 Evaluation protocol

For each model, each test set and each objective:

* **Accuracy:** RMSE, MAE, R2, and the same **relative to the noise ceiling** (RMSE divided by the noise sd, so 1 means perfect
  up to noise). This normalized RMSE is my proposal for the primary accuracy metric.
* **Tail metrics:** the same on the best 10 % of the true values (what the optimizer looks for), and the bias there (optimism).
* **Ranking:** Spearman correlation and pairwise ordering accuracy, within the operating region.
* **Robustness:** share of negative or out-of-range predictions on all sets.
* **Cost:** throughput at batch sizes 100 (one optimizer generation) and 10,000, model size, training time.
* **Transfer test (shortlisted models only, a few minutes each):** NSGA-III on the model with a fixed budget and seeds, re-simulate the
  final front with denoised labels, compare with the simulator-optimized reference front (hypervolume ratio and IGD+).

All predictions are stored as artifacts, so new metrics can be computed for old runs without retraining.

### 4.5 Tracking conventions

* One MLflow **experiment per scope** (`S1-fixed-material`, `S2-general-material`), one **run per training**, sweeps as nested runs.
* Logged per run: full config, dataset ids, git commit and dirty flag, hash of `uv.lock`, seeds, hardware, all metrics namespaced
  `<set>/<objective>/<metric>` (for example `T2/F1/nrmse`), learning curves per epoch, model, predictions, plots, wall time.
* **Model registry** with aliases such as `champion-S1`; the scripts of this repository will load models from an alias
  (milestone M4). The existing *model sets* stay supported.
* `python -m bmh_ml.ui` starts the UI on the store. `python -m bmh_ml.report` writes a leaderboard (markdown and HTML) from the store,
  so the comparison never needs manual bookkeeping.

### 4.6 Reproducibility and testing

Seeds everywhere, dataset ids and code versions in every run; a re-run of a run reproduces its metrics within a tolerance that is
itself tested. Unit tests use tiny synthetic data. CI must stay green (section 0).

## 5. Roadmap

Durations are **human-equivalent working days**, the effort a person would need to write, run and debug this. They are not my wall-clock time:
that is set by compute (data generation, training runs, sweeps) and by your review and decisions, not by writing code. I report real elapsed
time and what dominated it when a milestone is done.

### M0 - facts, plan, tooling check (today) - done when this file is committed
- [x] measure noise, operating region, hardware, speed
- [x] validate that MLflow works here (SQLite store, UI, API)
- [x] write this plan

### M1 - training and evaluation pipeline (about 2 days)
- [x] M1.1 module structure, optional dependencies, CI-safe tests (the config skeleton moved to M1.4, where the configs are used)
- [x] M1.2 dataset builder, manifest and hash, frozen test sets T1, T2, T3, T5, and the noise-ceiling tool (replaces the ad hoc numbers)
- [x] M1.3 model interface and baselines: mean, ridge, LightGBM, MLP, reproduction of the existing model
- [x] M1.4 evaluation suite (metrics, plots), MLflow logging, `train` and `evaluate` commands
- [x] M1.5 `report` (leaderboard), `ui`, documentation in the README
- **Acceptance met 2026-09-20:** from a clean checkout, three documented commands (`build_bundle`, `train`, `report`/`ui`) train the
  baselines and show them in the UI and in the leaderboard, with noise ceilings next to every number; a second run reproduces the
  metrics exactly; CI is green.

### M2 - the data and the deployment check (about 1 to 2 days)
- [x] operating-region data and the refinement loop (optimize the model, simulate, add the results, retrain)
- [x] transfer test as a pipeline stage
- [x] Optuna sweeps as nested runs
- **Acceptance:** the effect of the training distribution on T2 is measured and documented; the transfer test runs end to end.
- **Acceptance met 2026-09-24** (section 9): the effect of the training distribution is measured on T2, `valop` and in the transfer test,
  with three training seeds and a control of the same size; the transfer test runs end to end, alone and inside training, refinement and
  sweeps. The finding that matters for M3: the training distribution changes usefulness for optimization (transfer test) much more than
  accuracy on T2, and the operating region is specific to each model.

### M3 - modeling iterations (about 3 to 5 days, open ended)
Work through the backlog (section 6) in order of expected value. Every experiment: a hypothesis, the runs in MLflow, and a short entry in
`FINDINGS.md` (what was tried, the numbers, what we learned). Check in with you at the end of each backlog item that changed the picture.
- **Stop rule for an idea:** it either beats the current champion on T2 relative to the noise ceiling, or it is documented as tried.

### M4 - champions into the repository (about 1 day)
- [ ] registry aliases `champion-S1`, `champion-S2`; the scripts load models from an alias
- [ ] fast batched inference for the optimizer; end-to-end check with the real optimization scripts

### M5 - report and cleanup (about 1 day)
- [ ] final comparison table (accuracy, transfer, robustness, cost) and the recommendation
- [ ] remove what is obsolete (old trainer paths), update documentation

## 6. Experiment backlog

Ordered by expected value. "H" is the hypothesis, to be confirmed or rejected by numbers.

| # | Idea | H and how to test |
|---|---|---|
| B1 | **Operating-region training data** by the refinement loop | H: the largest single gain, because today the model extrapolates in exactly the region of use (77 % negative F2). Compare T2 error with random-only data of the same size. |
| B2 | **Positive outputs and log scale**, loss weighted to the good region | H: F2 spans 3 to 24 and F1 0.05 to 0.5, so relative errors matter; fixes impossible predictions by construction. |
| B3 | **Structured deposition data** (ramps, sawtooth, smooth paths) and features that describe the pile (deposited volume per bed section) | H: cheaper coverage of good regions than pure random sampling; F2 is a geometric property of the deposition. |
| B4 | **Structure of F1.** Quality is a passive additive quantity, so the reclaimed quality should be a linear mixing of the material, controlled by the deposition (F1 = spread of `W(deposition) x material`). A network that predicts `W` from the deposition and computes F1 analytically | H: far better generalization across materials (S2). First test linearity with the simulator at high particle density, where noise is small. |
| B5 | **Architectures:** wider or deeper MLPs, residual, 1D convolution and GRU over the real sequences (material and deposition are ordered), attention; deep ensembles | H: proper sequence models beat the length-1 "LSTM"; ensembles give uncertainty against exploitation. |
| B6 | **Boosting (LightGBM) and Gaussian processes** for S1 (20 dimensions) | H: strong on small, well-placed data; a GP gives uncertainty and can be extremely accurate with a few thousand points. |
| B7 | **Predict the reclaimed profile** (volume and quality per slice) and compute F1 and F2 analytically from it | H: much richer supervision than two numbers. Needs the generator to store profiles. |
| B8 | **Data and noise:** learning curves over the data size, repeated labels, heteroscedastic loss | H: shows what more data buys and whether label noise limits training. |
| B9 | **Inference speed:** vectorized numpy or torch models, export formats | H: batched inference can be much faster than a Keras `predict` per generation, which decides whether a surrogate is worth using. |
| B10 | **Gradient-based optimization** through a differentiable model | H: a smooth surrogate allows methods the noisy simulator cannot. |
| B11 | **Uncertainty guards** (penalize ensemble disagreement in the optimizer) | H: removes exploitation of model errors. |
| B12 | **Multi-fidelity** (cheap noisy simulations at low particle density as extra data) | H: more data per second of compute; only if B8 says data is the limit. |

## 7. Risks

| Risk | Mitigation |
|---|---|
| Overfitting the test sets through many attempts | frozen test sets, selection on validation data only, one final untouched set for the last report |
| Reading noise as signal | noise ceiling next to every metric, denoised labels, differences smaller than the noise are not conclusions |
| The surrogate is not worth it (section 2, speed) | cost is a criterion from the start; B9 tests early whether batched inference changes this |
| Heavy dependencies break CI or the other packages | optional dependencies, `importorskip`, CI-equivalent check before every push |
| The store grows without bound | it lives outside the synced folder; the report shows the size; old runs can be archived |
| Time sinks in sweeps | budgets per sweep, the stop rule of M3 |

## 8. Working agreements

* Every result exists in the tracking store or in the repository; nothing lives only in a chat message.
* Small commits with one topic each, message style as in the history. I do not push, you do.
* I verify with the CI-equivalent commands and on real data before I say something works, and I say what I did not verify.
* I decide small things and record them here; I ask about anything that changes the meaning of "best" or costs you time or money.
* Long runs run in the background with a log; a run that changes a conclusion is reported when it finishes.

## 9. Progress log

| Date | What |
|---|---|
| 2026-09-20 | M0: measurements, MLflow check, plan written |
| 2026-09-20 | M1.1 store location and tracking dependencies; M1.2 seeded generators, dataset store with manifest and content hash, frozen bundles, `build_bundle` command with the test sets T1, T2, T2s, T3, T5 and the noise ceiling table. Checked on the real data (small S1 bundle, about 50 s for 5,800 labeled inputs, noise levels as measured in M0). |
| 2026-09-20 | M1.3 model interface (`fit`, `predict`, `save`, `load`, one estimator per objective, F2 on the deposition only) with the mean, ridge, LightGBM, Keras MLP and the reproduction of the first LSTM (`legacy_lstm`); models are looked up by name and imported lazily. |
| 2026-09-20 | M1.4 metrics (errors, tail, ranking, negative rate, noise-normalized), evaluation on the bundle sets, throughput, plots, MLflow logging (params, dataset ids, code version, metrics, model, predictions), `train` and `evaluate_run` commands. Found and fixed on real data: LightGBM with all 16 threads was 10 to 40 times slower than with 8. First numbers on a 2,000 row S1 bundle: the operating-region set T2 already shows R2 below 0 for F2 for every model, as the plan predicted. |
| 2026-09-20 | M1.5 `report` (leaderboard with noise ceilings, markdown and HTML), `ui`, README. Checked with the real MLflow UI: health and API answer, the runs are listed. Open for the M1 acceptance: the run on full-size bundles. |
| 2026-09-20 | M1 acceptance on the full-size bundle S1-v1 (250,000 training rows, validation 20,000 x 8 repeats, T1 5,000, T2 544, T2s 2,450, T5 231, all x 16 repeats; building it took about 8 minutes). Baselines trained: mean, ridge, LightGBM, MLP. A second LightGBM run with the same seed gave identical values for all 123 metrics. Results (R2, noise ceiling about 0.9998): MLP is best with T1 0.907 (F1) and 0.972 (F2); LightGBM 0.827 and 0.951; ridge and mean have no skill. **In the operating region the F2 models have almost none:** MLP T2 F2 R2 0.02, on the surrogate fronts T2s -12.5. So random-data accuracy hides the problem, as expected, and B1 (operating-region training data) is the first thing to test in M2. The legacy LSTM run (100 epochs, batch 32) took about 29 minutes on 250,000 rows; its R2 (T1 0.781/0.929, T2 F2 -0.96) is behind both the new MLP and LightGBM, so the new MLP is already the best baseline. **M1 is complete, all its checkboxes and its acceptance criterion are met.** |
| 2026-09-24 | New machine (section 2). Store moved to `workdir/bmh-ml-store/`, MLflow paths rewritten, all 6 runs and their models load. LightGBM threads re-measured: 16 threads 1.6 times slower than 8 (not 10 to 40 times as on the first machine), 8 and 12 fastest, default kept. Keras `predict` on a batch of 100 took 79 ms against 6 ms calling the model directly, fixed (matters for every optimizer on a Keras model). M2 code: bundles with added training datasets and extra validation sets, transfer test (`transfer`, `train --transfer`), refinement loop (`refine`), Optuna sweeps (`sweep`), transfer columns in the leaderboard. |
| 2026-09-24 | **M2 transfer test and refinement loop, measured.** Yardstick: the 30 optimization runs on the simulator reach a hypervolume ratio of 0.57 on average (0.18 to 1.00) against the best known front (non-dominated T2, 16-repeat labels). Baselines on S1-v1: MLP 0.63 (0.25 per run), legacy LSTM 0.43 (0.17), LightGBM and ridge 0: all find fronts whose predictions are far too good (MLP: F2 bias -10, 47 % of the solutions with a negative prediction). Refinement loop, MLP, 4 rounds of 10 NSGA-III runs, front solutions plus 4 perturbations each (12,355 rows added to 250,000, about 25 minutes), bundle `S1-v1-rmlp` with the operating-region validation set `valop` (1,012 solutions, 8 repeats). **Three training seeds each:** transfer hypervolume ratio random data 0.63/0.71/0.62 (per run 0.25/0.41/0.24), refined 0.84/0.88/0.76 (per run 0.64/0.66/0.46), control with the same number of random rows added 0.73/0.47/0.51 (per run 0.50/0.24/0.26). Negative predictions on the found solutions: 47-69 % random, 0 % refined, 29-59 % control; F2 bias -9 to -14, -0.6 to -2.4, -8 to -12. **So the location of the data matters, not its amount: the refined models transfer better with every seed, and no longer exploit impossible F2 values.** Not solved: F1 stays optimistic on the found solutions (bias -0.10 to -0.17 in all variants), and the accuracy on T2 (the simulator's own optimized region) does not improve, F2 R2 even drops (baseline 0.02 to 0.37, refined about -0.48), with a large spread across seeds; T2 accuracy and usefulness for the model's own optimization are different things. `valop` F2 nrmse 32.6 (baseline) to 14.1 (round 4), control 28.7, F1 not improved; `valop` favors the last round by construction. |
| 2026-09-24 | **Hardware and tracking.** CPU use measured with a benchmark on a copy of the store (percent of the 16 hardware threads; 50 % is about all 8 physical cores): 14 NSGA-III runs on the MLP 40.3 s at 17 % in one process, 13.6 s at 66 % in 14 processes; 3 MLP trainings 108 s at 33 % one after another, 60 s at 56 % in 3 processes; 4 LightGBM sweep trials 28.7 s at 47 %, 21.1 s at 61 % with 2 workers. A single MLP training used only about a third of the CPU, so F1 and F2 are now fitted in two processes at the same time (MLP 39 to 31 s for 10 epochs including process start, LightGBM 25 s at 81 % CPU with bit-identical results). The optimizations of the transfer test and the refinement loop run in parallel, `train --seed ... --workers`, `sweep --workers`. The GPU (Navi 10) stays unused: the MLP is too small to be limited by arithmetic. MLflow: every run gets a generated description (summary line, purpose, links to related runs, datasets, parameters, results, reproduce command), datasets are logged as inputs, saved views for the UI (`tracking/views.py`: leaderboard, comparison charts, refinement rounds, sweeps), runs start before the training so their duration is real, and adding results later no longer reopens a run (the end times of 5 runs reopened earlier were repaired). |
| 2026-09-24 | **LightGBM refinement and the first sweep.** LightGBM refinement loop (from the S1-v1 baseline, 4 rounds, one seed): transfer hypervolume ratio 0 (round 0), 0.02, 0.31, 0.24, **0.74** (round 4, per run 0.26), control 0.43; F2 bias -17.6 to -2.9 (control -20.9), negative predictions 56 % to 0 % (control 73 %). Sweep `lgbm-rmlp-1` (LightGBM, 20 trials on `S1-v1-rmlp`, objective the mean nrmse of F1 and F2 on `val` and `valop`, 20 minutes): best 8.86 (num_leaves 490, learning_rate 0.04, min_child_samples 142) against about 10 for the defaults; `valop` F1 nrmse 5.8 (the refined MLP 7.8), F2 16.7 (MLP 14.1). **But its transfer test is poor: 0.63 (per run 0.16), F2 bias -12.4.** Two lessons: the refinement data of one model does not cover the regions another model's optimizer exploits (the operating region is model-specific), and validation accuracy, even on `valop`, does not predict usefulness for optimization, so sweeps need the transfer test in their objective. |
| 2026-09-24 | **M2 complete** (acceptance in section 5). Decisions taken with you today: Chevron with 19 passes as the reference for relative objectives, reordering the work by quality instead of the publication deadline. Open: the F2 normalization (section 10). |
| 2026-09-24 | **Chevron in the transfer test.** Reference: the 19-pass Chevron (alternating ends), 64 simulations, cached per material in the store (`references/`): F1 0.535, F2 12.67. New metrics (`transfer/chevron_*`, all objectives divided by Chevron's, reference point (1, 1)): hypervolume beyond Chevron (`chevron_hv`, pooled and per run), share of the solutions better than Chevron in both objectives (`chevron_beaten_rate`), best relative F1 and F2, the same for the simulator's best front and for the predictions. `transfer --recompute` computed them for all 21 earlier tests from the stored solutions, no new simulations. Results: the 30 simulator runs beat Chevron with 65 % of their solutions on average (0 to 100 %), hypervolume beyond Chevron per run 0.155 (0 to 0.445), their pooled best front 0.449. Refined MLP (3 seeds): 98 to 100 % beat Chevron, 0.47 to 0.49 pooled over its 5 runs, 0.35 to 0.41 per run. MLP on random data: 42 to 82 %, 0.40 to 0.42, 0.24 to 0.33; control 64 to 85 %, 0.37 to 0.45, 0.27 to 0.37; refined LightGBM 97 %, 0.41, 0.29; legacy LSTM 10 %, 0.32, 0.20. **Caveats before this is a claim:** the per-run comparison is not budget-matched (a model run uses 20,000 cheap model evaluations but the model was trained on about 262,000 simulations; the simulator runs used 20,000 to 100,000 simulations each, with older settings), and the ranking by hypervolume beyond Chevron differs from the ranking by the reference-normalized hypervolume ratio, because they weight different regions of the front. The simulation-budget experiment (section 10, finding 4) is what makes the comparison fair. |
| 2026-10-01 | **M3 started: predicting the reclaimed profile.** The simulator returns 60 reclaimed slices (volume and quality) and F1 and F2 are exact functions of them (checked: recomputed objectives equal the labels). The F of a mean profile is slightly below the mean F of noisy simulations; the gap, measured with 16 repeats, is about 0.001 for F1 and 0.02 (random) to 0.11 (optimized) for F2, i.e. an added constant of about 0.001 and 1.1 in F^2. Datasets can now carry profiles (float16), `build_bundle --profiles --new-val`, model `profile_mlp`. Bundle **S1-v2**: the inputs of S1-v1 simulated again with profiles, new validation set (same inputs), the frozen test sets of S1-v1 (2 minutes to build). Store: LightGBM models now stored compressed (the sweep had grown the store to 3 GB; 54 boosters 3.02 to 1.19 GB, identical predictions). **First comparison** (same architecture 512x4, same simulations, 3 seeds each, all with the transfer test): F2 from the profile is far better where it matters: T2s F2 R2 0.74 to 0.84 against -13 to -15 for the scalar MLP, transfer hypervolume ratio 0.66 / 0.99 / 0.70 against 0.60 / 0.59 / 0, beats Chevron 93 to 100 % against 4 to 54 %, no negative predictions (scalar 32 to 59 %), F2 bias on the found solutions +0.4 to +1.3 against -10 to -16, **without any refinement** (the refined scalar MLP reached 0.83 and 99 %). F1 from the profile is poor (T1 R2 0.45 to 0.64 against 0.89; the small errors of every slice's quality add up in the spread). The first noise correction, fitted on training labels, absorbed model error (10 to 26 instead of about 1); replaced by a correction measured on the repeated validation data alone (S1-v2: 0.0008 and 1.12, matching the direct measurement). Next run: one network predicting profile and objectives, F1 from the direct output, F2 from the profile, plus the ablation with both objectives from the direct outputs. |
| 2026-10-02 | **M3: the hybrid profile model.** One network (512x4) predicts the 120 profile values and F1 and F2 directly; F1 is taken from the direct output, F2 is computed from the predicted profile with the noise correction measured on the repeated validation data (0.0008 for F1, 1.12 for F2). Means of 3 seeds on S1-v2, same simulations for every variant (scalar MLP / profile only / both objectives from the direct outputs / **hybrid**): validation F1 nrmse 5.35 / 11.5 / 3.73 / **3.73**; F2 nrmse 5.84 / 4.68 / 4.04 / 4.39; T2 F1 R2 0.90 / -0.05 / 0.93 / **0.93**; T2 F2 R2 0.02 / -0.34 / 0.48 / **0.53**; T2s F2 R2 -14.2 / 0.78 / -4.0 / **0.82**; transfer hypervolume ratio 0.40 / 0.78 / 0.64 / **0.99** (per run 0.15 / 0.53 / 0.41 / **0.76**; the simulator's own runs 0.57); hypervolume beyond Chevron 0.30 / 0.43 / 0.43 / **0.525** (the simulator's best front 0.449); beats Chevron 28 / 97 / 98 / 97 %; negative predictions 45 / 0 / 0 / 0 %; F2 bias on found solutions -13.8 / +0.7 / -3.6 / **-0.13**; F1 bias -0.16 / -0.06 / -0.09 / -0.06. **Conclusions:** training on the profiles makes the direct F1 output about 30 % more accurate; F2 must be computed from the profile, a direct F2 output of the same network is still exploited by the optimizer (and validation accuracy again ranks it best, the transfer test does not). Without refinement and without extra simulations the hybrid model is the best surrogate so far. Hardware: the CPU overheated (92 to 100 C) and crashed the machine twice; after the cooling fix 85 to 96 C under full load, with a watchdog that stops the work at 97 C (`workdir/logs/`). |
| 2026-10-02 | **Refinement loop for the hybrid model** (`S1-v2-rhybrid`, round 0 = hybrid seed 1, 4 rounds of about 2,900 rows each with profiles, random control; one seed). Accuracy where the optimizer goes improves clearly: `valop` nrmse F1 8.9 to 4.9, F2 7.9 to 5.0 (control 6.4 / 9.2); T2 F2 R2 0.51 to 0.64-0.74 (control 0.36). The transfer test does not move: hypervolume ratio 1.05 / 1.00 / 0.89 / 1.05 / 0.88 for rounds 0 to 4, control 0.81, within the seed spread of the hybrid model (0.91 to 1.05); beats Chevron 85 to 100 % throughout; F1 bias on the found solutions stays about -0.05 in all rounds, F2 bias drifts from +0.2 to -1.5. **So the hybrid model is already about as useful for optimization as the transfer test can tell without refinement; refinement buys accuracy in the operating region, not better optimization results.** The remaining weakness is the F1 optimism (-0.05), and the cost question (simulation budget) is still open. The loop hung once in the control simulation (fork of a process running TensorFlow threads); simulation workers now start with spawn, and the loop was completed with the same code path. |
| 2026-10-02 | **Simulation budget (S1).** New baseline `transfer --simulator-baseline`: NSGA-III (population 100, 5 seeds) directly on the simulator, fronts re-simulated 16 times and scored like the transfer test; every run now records `budget/simulations`. Per optimization run (hypervolume ratio against the old T2 reference front / beyond Chevron / share beating Chevron): simulator with 2,000 simulations 0.51 / 0.33 / 67 %, 5,000 0.87 / 0.49 / 87 %, **10,000 1.05 / 0.56 / 98 %**, 25,000 1.11 / 0.59, 100,000 1.18 / 0.61. Hybrid profile model (3 seeds each) trained on 10k / 25k / 50k / 100k random simulations (budget incl. validation 12k / 30k / 60k / 120k): 0.38 / 0.36 / 0.54 / 0.56 per run, beyond Chevron 0.30 / 0.32 / 0.39 / 0.40; with 250k (S1-v2, 410k with validation) 0.76 / 0.44. **For one material the surrogate does not pay: about 15 s of NSGA-III on the simulator (10,000 simulations at 1.3 ms) beats the best surrogate trained on 410,000 simulations.** The old reference front (T2, 30 runs of the earlier experiments) is weaker than fresh simulator runs, which exceed a ratio of 1; the reference should become the best front of these runs. Where a surrogate can pay: many materials (train once, optimize a new material without simulations), real time, and expensive simulators; this is the next step (S2), compared per new material with simulator NSGA-III at a given budget. |
| 2026-10-02 | **S2, many materials: first result.** Bundle S2-v1 (15 minutes): 500,000 rows with a random material each (smoothed random walks between 5 and 10) and profiles, validation 20,000 x 8, T1, T3 (20 unseen materials x 50), T5, T2 (the thesis material's simulator fronts) and the new **T6: reference fronts for 8 unseen materials** (3 NSGA-III runs of 25,000 simulations each on the simulator, 43 to 83 front solutions per material, 16-repeat labels). The transfer test now runs per material and averages (`hv_ratio_material_min` is the worst material). Hybrid model, 3 seeds (25 minutes each, 3 in parallel): accurate on random inputs and on unseen materials (T3 R2 0.93 for F1, 0.97 for F2), but not where the optimizer goes: T6 F1 R2 -12 to -13, F2 about 0, T2 F1 0.73-0.82, F2 0.27-0.47. **Transfer on the 8 unseen materials: hypervolume ratio per run 0.03 to 0.05, beats Chevron 43 to 53 % (worst material 0 to 3 %), F1 bias -0.06 to -0.10; NSGA-III directly on the simulator with 5,000 simulations: 0.16 and 75 %, with 10,000: 0.47 and 90 % (worst material 84 %).** The general-material model repeats the S1 story before refinement, more strongly: random-data accuracy hides an operating region it has never seen, now for every material. Candidate remedies: refinement over many materials (each round optimizes the model for new random training materials), F1 relative to the material's own spread (Chevron of the material), and models that read the material curve as a sequence. |

## 10. Review against the domain definitions (2026-09-24)

Read in the Entropy vault: the objective definitions (`Concepts/Optimization/Objectives/`, `Concepts/Stockpile Math/`), the student
research questions and the notes on the students' ML work (`Projects/Students/`), and `IEEE SSCI 2027 Submission.md`. Compared with the code:
`bmh.helpers.reclaimed_material_evaluator` and the thesis optimizer `bmh.optimization.homogenization_problem`.

**Findings**

1. **bmh_ml predicts the absolute objectives, the thesis optimizes relative ones.** bmh_ml's F1 is the volume-weighted standard deviation of
   the reclaimed quality, and F2 is the standard deviation of the difference between the reclaimed volume per slice and the reclaim volume of
   the ideal stockpile. These are the numerators of the thesis objectives. The thesis F1 is the *homogenization efficiency ratio*, this
   standard deviation divided by that of full-speed Chevron stacking of the same material (1 = as good as Chevron, hypervolume reference
   point 1). The thesis optimizer divides F2 by one sixth of the volume per slice ("worst acceptable"); the vault has "relative F2 against
   Chevron" as an open TODO.
   - For S1 (one material) the difference is a constant factor, so the models and M1/M2 accuracy numbers are unaffected. The reporting is
     not: fronts, hypervolumes and the transfer test should also be given relative to Chevron, so they compare with the thesis experiments
     and the students' results.
   - For S2 (any material) it matters for the model: absolute F1 scales with the spread of the material, a ratio to Chevron of the same
     material does not. S2 should predict the ratio, with the Chevron denominator simulated with repeats (it is noisy too).
2. **Chevron is the baseline everyone asks about, and the transfer test does not report it yet.** Keith Kumar re-simulated 8,388 fronts
   of MLP and XGBoost surrogates: the simulated values were worse than predicted for 79 to 89 % of the points, and only 4.6 % of them
   dominated Chevron. The same exploitation shows here (section 9). The transfer test should add the Chevron point and the share of
   found solutions that dominate it in the simulator.
3. **The student research questions are this milestone's questions.** "How does the front change when the solutions found by the ML model
   are re-evaluated with the simulation?" is the transfer test. "How much does a different material curve affect the results?" needs the
   transfer test on several materials (S2, or S1 bundles for other materials). "How can a front be compared with Chevron?" is finding 2.
   Keith's CSVs (fronts with predicted and simulated objectives) can check a re-implementation, not replace it: they come without code.
4. **Cost should be counted in simulations, not only in prediction speed.** The surrogate here was trained on 250,000 simulations plus
   about 3,000 per refinement round; one optimization run on the simulator uses 20,000 to 100,000. A claim that a surrogate helps
   optimization needs the hypervolume reached per simulation spent, including the training data. The experiment that answers it: the
   refinement loop from a small random base (for example 10,000 or 25,000 rows) against simulator-only optimization with the same number of
   simulations. This joins B1 and B8.
5. **F2 is geometry and F1 is linear mixing, both by definition.** F2 compares the reclaimed volume curve with the ideal stockpile (half
   cones and a triangular core at 45 degrees, closed form in the vault), and the volume per slice is a function of the deposition only. F1 is
   the weighted spread of reclaimed quality, which is a volume-weighted average of the input material per slice. This supports B3/B7 (predict
   the reclaimed volume and the mixing weights per slice, compute F1 and F2 from them) and connects to the student topic "compare a
   mathematical formulation with the simulation".
6. **The publication deadline sets the order.** The IEEE SSCI 2027 poster abstract (250 words) is due 1 November 2026. What it needs from
   here: the transfer-test result with a spread over training seeds, the refinement loop against its control, and the Chevron comparison.
   Hyperparameter tuning and architectures (M3) do not change that story and come after.
7. **GPU:** the RX 5600/5700 (Navi 10) could speed up the Keras models, but ROCm is not installed and Navi 10 would need an unofficial
   setup. The models are small (the MLP trains in about 2.5 minutes on the CPU), so it only pays off for B5 (larger models, ensembles).
   Installing ROCm is a system change and your decision.

**Decisions of 2026-09-24:** Chevron with 19 passes (alternating ends, the 20 variables kept) is the reference; relative objectives are for
normalization and for talking about improvement. The order of the work is chosen for the quality of the results, the publication deadline
is ignored for now. The "one sixth" of the thesis optimizer comes from commit `ceb217d` (2019): the standard deviation of two slices at 7/6
and 5/6 of the mean slice volume, a tolerance of about 17 % per slice, not derived. F2 depends only on the deposition, so its Chevron value
is a constant (12.67 here, 64 repeats; F1 0.535); **F2 is divided by Chevron's F2 as well (confirmed 2026-09-24)**, so (1, 1) means
"as good as Chevron" in both objectives.

**Proposed changes** (the first applies next):

- ~~M2: add the Chevron reference to the transfer test~~ done 2026-09-24 (`evaluation/chevron.py`, section 9); the transfer test is
  repeated over training seeds before an effect is stated.
- New M2b before M3, for the abstract: the simulation-budget experiment of finding 4, and the transfer test on a few other materials
  (finding 3).
- S2: predict F1 relative to Chevron of the same material (finding 1). Needs a decision on the Chevron schedule in the 20-variable
  encoding (for example alternating between the two ends of the bed) and on the F2 normalization (Chevron's F2, or the thesis's one
  sixth of the volume per slice).
- Backlog: move B3/B7 (geometric features, predicting per-slice profiles) up, next to B1.
- Sweeps: optionally include the transfer test in the objective (1 to 1.5 minutes per trial), as validation accuracy did not predict it.
- Refinement per model: every model family needs its own refinement loop (or data from several models' optimizers).

**Finding of 2026-10-01, open: the ideal stockpile of F2.** F2 uses `bmh.helpers.stockpile_math.get_ideal_stockpile_volumes` (the formula of the vault note "Ideal Stockpile", the cut area evaluated at one point per slice). The newer derivation in the vault (`Concepts/Stockpile Math/Ideal Stockpile Derivation`, four phases, the cut area integrated over each slice) gives about one slice later ramps (slice 3: 4.4 against 20.4; plateau 53.6 in both) and sums exactly to the total volume (the point rule gives 2503.5 for 2500). The simulator departs from both: even Chevron (mean of 64 runs) reclaims nothing before slice 7, about 4 slices later than either ideal, and overshoots to about 62 in slices 12 to 14, so part of every F2 value is this start offset. To clarify with you: is the offset expected (reclaimer or slice coordinates of the simulator), and should F2 move to the integrated derivation? Because datasets now store the profiles, F2 can be recomputed for any definition without simulating again. Cause, measured on the pile itself (16 Chevron runs, `get_heights`): the ideal has the full ridge height 7.3 from x = 10 to 49 (the
stacker's travel limits) with half cones out to 2.7 and 56.3; the simulated pile starts at x = 5, reaches full height only at about x = 14,
falls off from about x = 43 and ends at about 53. Its ends taper inward over 4 to 5 cells instead of forming a cone around the turning
points. The simulator reclaims a particle at column x and height h in slice x - h (the same lean as the derivation), and the binding labels
the slice collected between positions k-1 and k as x = k (one slice of offset).
