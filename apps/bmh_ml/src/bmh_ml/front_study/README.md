# Front study: interpolation web and resolution ladder

Two charts of 2026-10-08 on one simulator-optimized Pareto front: the **interpolation web** (sections 1 to 6) and the **resolution
ladder** (section 7, made for use in publications). The code is this package, `bmh_ml.front_study`; every step is run from the repository
root as `python -m bmh_ml.front_study.<step> <material_index>`. All results go to the git-ignored directory
`workdir/front-interpolation/out` (or `$BMH_ML_FRONT_STUDY_OUT`), called `out/` below.

The interpolation web: a simulator-optimized Pareto front is taken, and every pair of its solutions is connected by a straight line
in deposition-variable space. Every line is re-evaluated with the simulator and drawn in objective space (F1 against F2). Each line on its
own is almost transparent, but together they show how the objectives behave between optimized deposition schedules. Every point of a
line is the mean of 32 simulations; every solution is drawn at the mean of 128 single simulations, which surround it as a noise cloud and
show how far simulator noise alone moves it.

* **The chart:** `out/front-interpolation-material-2.html`, one self-contained file (2.0 MB, all data embedded, the 411,000 line points
  compressed). Open it in any browser, no server needed. It loads the IBM Plex fonts from Google Fonts and falls back to system fonts
  when offline.
* **The data** in plain formats: `out/solutions-material-2.csv`, `out/pairs-material-2.csv`, `out/lines-material-2.csv.gz`,
  `out/data-material-2.json` (section 4).
* **Recomputing everything:** `run_all.sh` in this directory (section 3), about 1.5 hours on 14 cores.

## 1. Which data was used

| | |
|---|---|
| Result | Bundle `S2-v1` of the `bmh_ml` store, test set **T6** (dataset `S2-v1-T6-c6b60ab04f`, built 2026-10-02 with code `e9b1bc0`). T6 holds the final fronts of NSGA-III runs directly on the simulator for 8 unseen random materials, with labels that are the mean of 16 simulations per solution. Read from the store `workdir/bmh-ml-store` (read only, nothing written to it). |
| Material | **Material index 2** (0-based, the third of the 8, in generation order). Chosen because its front is the widest relative to the simulator noise: 83 final solutions of the three runs, 23 non-dominated on the stored labels, F1 0.106 to 0.328, F2 3.4 to 12.1. All 83 are used, not only the non-dominated ones. The material is 50 quality values, a random walk in 5 to 10 (steps ±1, Gaussian smoothing σ 0.75), regenerated exactly from the bundle seed: `np.random.SeedSequence(1).spawn(7)[6]`, then `random_materials(8, rng)[2]`. Total volume 2,500 m³. |
| Stockpile | Bed 59 × 20 m. 20 deposition variables: the stacker position x ∈ [10, 49] m at z = 10 m, evenly spaced in time over 86,400 s. Reclaim speed 59 m per 86,400 s. Constants in `apps/bmh_ml/src/bmh_ml/settings.py`, simulation in `simulation.py`. |
| Simulator | Fast `BslBlendingSimulator` (`blending_simulator_lib`) at level L1: `ppm3` = 1, reclaim step 1 m, reclaim angle 45°, eight-likelihood 0.87 (the defaults), 60 reclaimed slices. T6 was simulated with simulator release v2026.1; everything here was simulated with v2026.2 (the tag in `libraries/blending_simulator_lib/CMakeLists.txt`), whose fast simulator only refactors v2026.1 (same model, different random number stream). Both are unseeded: every simulation is random. |
| Objectives | F1: volume-weighted standard deviation of the reclaimed quality. F2: standard deviation of the reclaimed volume per slice from the ideal stockpile, m³. Both minimized, evaluated between x = 10 and 49 m (`bmh.helpers.reclaimed_material_evaluator`). |
| Optimizer of the result | pymoo 0.6.1.6 NSGA-III, population 100, 250 generations (25,000 simulations per run), 100 uniform reference directions, SBX (p 0.9, η 15), polynomial mutation (η 20), random start, duplicates eliminated, seeds 1, 2, 3 (`bmh_ml.experiment.optimize`, `evaluation/transfer.py: run_simulator_fronts_many`). |
| Code | The results in `out/` were computed on 2026-10-08 against `BlendingEvaluation` commit `dbe2be3`, in a separate environment (`uv sync --locked --all-packages`, Python 3.12), on an Intel i9-9900K with 14 worker processes, by the same scripts as standalone files in `workdir/front-interpolation/`. They then moved into this package; rebuilding the pages and exports from the package reproduced every data file (CSV, JSON contents) exactly. |

## 2. How the chart was generated

The steps, in order (modules of this package; each takes the material index as its only argument):

1. **`verify.py`: is the result recomputable?** Regenerates the 8 T6 materials from the bundle seed and checks them against the stored
   ones. Simulates the 83 stored solutions again (16 repeats, like the stored labels) and compares. Runs the three NSGA-III optimizations
   again with the same settings and labels their fronts with 16 repeats. Simulates Chevron stacking (19 passes, 64 simulations) as a
   reference. Writes `out/verify-material-2.json` (numbers) and `.npz` (arrays). About 2 minutes.
2. **`interpolate.py`: the lines.** For every pair (a, b) of the 83 solutions (3,403 pairs), the depositions a + t (b − a) for
   t = 1/40 … 39/40, each simulated 8 times (the mean is the point). The end points (t = 0 and 1) are the solutions themselves with their
   pooled labels: the stored 16 and the fresh 16 simulations of step 1, 32 in all (replaced in step 5). The deposition box is convex, so
   every point is a valid deposition. 1.06 million simulations, 6 minutes. Writes `out/lines-material-2.npz`.
3. **`refine.py`: more points where a line changes.** Adaptive bisection, 6 rounds. In each round, every segment between two
   neighbouring points of a line whose objectives differ by more than 4 noise units gets its midpoint, simulated 8 times like the others.
   A noise unit is the standard deviation of the difference of two 8-simulation means (F1 0.0055, F2 0.100 m³, measured from the spread
   of all points); F1 and F2 are combined as a Euclidean length in these units, so pure noise exceeds 4 in about 1 of 3,000 segments.
   Segments are not split below 1/2560 of a line (64 times finer than step 2). Flat stretches keep the spacing of step 2. Rounds added
   59,255 / 59,192 / 49,992 / 40,291 / 33,368 / 29,223 points, 271,321 in all (404,038 interior points, 3 times step 2; 2.2 million
   simulations), 17 minutes. The budget of 1.2 million points was not needed: the refinement stops at the finest spacing, where 26,625
   segments still jump by more than 4 units (steps, see section 6). The state is saved after every round to
   `out/refined-material-2.npz`, and a rerun of `refine.py` alone resumes from it.
4. **`resolution.py`: the noise clouds** (and the resolution ladder, section 7). Every solution and Chevron simulated 128 times at six
   particle resolutions, keeping every single result; the level L1 (`ppm3` = 1, the resolution of the optimization and of the lines)
   provides the clouds of this chart. **The solutions are drawn at the mean of their 128 L1 simulations**, and the front is the
   non-dominated part of these means: **23 solutions** (253 pairs among them). 12 minutes. Writes `out/clouds-material-2.npz`.
5. **`deepen.py`: 32 simulations per point.** Every interior point of the refined lines is simulated 24 more times and pooled with its 8
   (mean and standard deviation of all 32), which halves its noise: standard error now F1 0.0021, F2 0.035 m³ (before 0.0039 and
   0.071). The end points of every line become the solutions' 128-simulation means. 9.7 million simulations, 56 minutes, in chunks of
   40,000 points saved as they finish (a rerun resumes). Writes `out/deep-material-2.npz`. Averaging removes noise, not the steps of
   the simulator (section 6): with less noise the lines show their staircase more clearly.
6. **`confirm_beyond.py`: the points that seem to beat the front.** 6,498 interpolated points (32 simulations each) were dominated by no
   final solution. As they were picked among 404,038 noisy points, part of that is luck; each is simulated again 32 times and kept as
   *confirmed* if it is still non-dominated. 4,108 are. 131 s. Writes `out/beyond-material-2.npz`.
7. **`build_interpolation_page.py`: the page and the exports.** Numbers the solutions by F1 (ids #1 to #83 shown in the chart), computes
   the summary statistics, embeds everything in `templates/interpolation.html` (replacing the placeholder `/*DATA*/null`) and writes the
   CSV and JSON exports.
   Plain Python with numpy, no `bmh_ml` needed, a few seconds.

Not part of the chart: **`seeded_scan.py`** scans one base segment (t = 0.25 to 0.275 of the longest line between two front solutions,
pair 1803, 82 m) in 1,000 steps with a fixed simulator seed, which makes every simulation deterministic (seeds 1 and 2). It shows the
shape of the response without noise (section 6). Writes `out/seeded-scan-material-2-seed{1,2}.npy` (columns t, F1, F2). Seconds.

**The reference: Chevron at maximum speed.** The stacker alternates between the two ends of the bed at every one of the 20 deposition
steps (x = 10, 49, 10, …, 49), so the pile has 19 layers, the number of deposition variables minus one
(`bmh_ml.evaluation.chevron.chevron_deposition`). The chart uses the mean of its 128 L1 simulations of step 4 with the material of the
front: F1 0.768, F2 12.66 m³ (step 1's 64 simulations gave 0.769 and 12.63). The chart can show the objectives relative to it
(1 = as good as Chevron), the convention of the `bmh_ml` transfer test. On this material every front solution is far
better than Chevron in F1 (0.14 to 0.43 of it) and better in F2 (0.27 to 0.95). The interpolation lines never reach Chevron's F1 (at
most 0.68 of it), but 58 % of the lines between front solutions (39 % of all lines) cross above Chevron's F2 somewhere along the way.

**A reproduction of results with point drops.** The study was made before the simulators of `bmh` dropped the material continuously
along the stacker path and before F2 integrated the ideal stockpile over each slice. It keeps the conditions of that time on purpose:
every material portion is stacked at the stacker position of its timestamp (drops "point"), F2 compares with the ideal stockpile at the
slice positions (rule "point"), and the reference is the constant-speed Chevron above, which the steps take directly from
`chevron_deposition()` instead of the Chevron with stops and shrinking turning points that new results are measured against
(`bmh_ml.evaluation.chevron.chevron_path`). `front_study.common.get_study_settings` holds these settings: `resolution.py` and
`time_levels.py` simulate with them, `run_all.sh` exports them (`BMH_ML_SIMULATOR`) for the other steps, and `verify.py` takes those
recorded with bundle `S2-v1`, which read as the same. A new study of the fronts would use the settings and the Chevron of the current
code instead.

## 3. Recomputing

Never use or sync the repository's `.venv`; make a separate environment once:

```shell
cd Code/BlendingEvaluation
UV_PROJECT_ENVIRONMENT=/some/dir/outside/env uv sync --locked --all-packages
ENV=/some/dir/outside/env apps/bmh_ml/src/bmh_ml/front_study/run_all.sh 2
# or one step at a time, e.g. after an interruption:
BMH_ML_STORE=$PWD/workdir/bmh-ml-store /some/dir/outside/env/bin/python -m bmh_ml.front_study.deepen 2
```

`run_all.sh` runs the simulation steps, both page builders and the figure for one material and overwrites `out/` (it deletes the saved states of
`refine.py`, `resolution.py` and `deepen.py` first, so they start from the new data; each of them alone resumes an interrupted run). `BMH_ML_STORE` defaults to `workdir/bmh-ml-store`, which must contain bundle
`S2-v1` and its datasets. Worker processes: `N_JOBS = 14` at the top of `verify.py`, `interpolate.py`, `refine.py`, `resolution.py`,
`deepen.py` and `confirm_beyond.py`. Every step runs under `watchdog.sh`, which logs the CPU package temperature every 10 s to
`workdir/logs/temp-front-study-<step>.log` and stops the step if it stays at 95 °C or more for 30 s. On the i9-9900K the full load
reached 86 °C on 2026-10-08. After an interruption, run the interrupted step alone (it resumes), then the later steps.

**Results differ from run to run.** The simulator is random and unseeded, so every rerun gives slightly different numbers (within the
noise reported below), and the rerun of the optimization in step 1 finds a different front each time. For bit-identical reruns, the
simulator seed of v2026.2 (`BslBlendingSimulator(..., seed=...)`) would have to be passed through `bmh_ml.simulation`, which it is not.

**Another material or bundle.** The material index (0 to 7) is the script argument; the page is named after it. Another bundle with a T6
set: change `load_bundle("S2-v1")` in `verify.py` (the bundle's detail level is applied by `use_bundle_simulator`, so an `S2-L4` front
would be simulated at `ppm3` = 64, about 50 times slower per simulation). Larger fronts grow the work with the square of the number of
solutions; `SEGMENTS` and `REPEATS` at the top of `interpolate.py` set the points per line and the simulations per point, and
`THRESHOLD`, `GRID`, `BUDGET` and `MAX_ROUNDS` at the top of `refine.py` the refinement, `TARGET` in `deepen.py` the simulations per
point, `REPEATS` and `LEVELS` in `resolution.py` the clouds.

## 4. Files and formats

| File | Content |
|---|---|
| `verify.py`, `interpolate.py`, `refine.py`, `resolution.py`, `deepen.py`, `confirm_beyond.py`, `time_levels.py` | The simulation steps (`python -m bmh_ml.front_study.<step>`). |
| `seeded_scan.py` | The deterministic scan of one segment (section 2). |
| `common.py` | The output directory and Pareto dominance, shared by the steps. |
| `build_interpolation_page.py`, `templates/interpolation.html` | The page builder and the page (HTML, CSS, JS, no libraries; the data placeholder is `/*DATA*/null`). |
| `run_all.sh`, `watchdog.sh` | All steps for one material; the temperature watchdog every simulation step runs under. |
| `../tests/test_front_study.py` | Unit tests of the dominance, the pooling of simulations, the line encoding, the refinement rule and the quantization. |
| `out/front-interpolation-material-2.html` | **The chart.** |
| `out/solutions-material-2.csv` | One row per final solution (83): `row` (index used in all other files), `id` (number shown in the chart, by F1), `front` (1 if on the Pareto front used here), `f1`, `f2` (mean of the 128 simulations of the cloud at L1), `f1_sd_single`, `f2_sd_single` (standard deviation of one simulation, from the same 128), `x1` … `x20` (the deposition schedule, stacker position in m at the 20 time steps). |
| `out/pairs-material-2.csv` | One row per line (3,403): `pair`, `row_a`, `row_b` (rows of the solutions file), `id_a`, `id_b`, `both_front` (1 for the 253 lines between front solutions), `distance_m` (Euclidean distance of the two schedules), `points` (points of the refined line, 41 to 270). |
| `out/lines-material-2.csv.gz` | Every point of every line (410,844 rows, sorted by pair and t): `pair`, `position` (0 … 2560), `t` = position / 2560 (0 at solution a, 1 at b), `f1`, `f2` (mean of the simulations), `f1_sd`, `f2_sd` (standard deviation of the single simulations, empty at the end points), `simulations` (32, or 128 at the end points), `round` (0 for the 41 points of step 2, 1 … 6 for the refinement round that added the point). The deposition of a point is `x_a + t (x_b − x_a)`. |
| `out/data-material-2.json` | Exactly what the page embeds: `labels`, `id`, `front`, `deposition`, `material`, `pairs`, `lines` (all refined points as gzipped, base64 encoded little-endian int16: the number of points of each pair, then per point the position on the grid, F1 × 10⁴ and F2 × 10³, each as the difference to the previous point of the same pair; `build_interpolation_page.encode_lines`, decoded by `decodeLines` in the page), `clouds` (per solution, its 128 single simulations at L1, F1 × 10⁴, F2 × 10³ as integers), `beyond` (`pair` and `y`, the 32-simulation objectives, of the confirmed points), `chevron` (the reference, F1 and F2 unquantized), `chevron_cloud` (its 128 single simulations, quantized), `check` (the verification numbers), `stats`. |
| `out/lines-material-2.npz` | The 41-point lines of step 2 at full precision (8 simulations per point): `material` (50), `deposition` (83 × 20), `labels` (83 × 2, pooled 32 simulations), `front` (22 solutions on these labels, superseded by step 4), `pairs` (3,403 × 2), `t` (41), `lines` (3,403 × 41 × 2), `interior_sd` (3,403 × 39 × 2), `chevron`. |
| `out/refined-material-2.npz` | The refined lines of step 3 (8 simulations per point), one entry per point: `pair`, `position`, `f` (n × 2), `sd` (n × 2, NaN at the end points), `simulations`, `round`. |
| `out/deep-material-2.npz` | The lines of the chart (step 5, 32 simulations per point), same arrays as `refined-material-2.npz`. |
| `out/verify-material-2.json`, `.npz` | Verification numbers; arrays `stored`, `stored_sd` (T6 labels), `fresh`, `fresh_sd` (16 new simulations), `rerun_deposition`, `rerun`, `rerun_seed` (the repeated optimization), `chevron`. |
| `out/beyond-material-2.npz` | The candidates of step 6: `index` (into `deep-material-2.npz`), `pair`, `position`, `y` (32 fresh simulations), `confirmed`. |
| `out/clouds-material-2.npz` | Every single simulation of every solution and of Chevron at six resolutions (section 7); level L1 is the cloud of this chart. |
| `out/seeded-scan-material-2-seed{1,2}.npy` | 1,001 × 3 (t, F1, F2): the deterministic scan of `seeded_scan.py`. |

Reading the data without the page, for example (from `out/`):

```python
import numpy as np, pandas as pd
solutions = pd.read_csv("out/solutions-material-2.csv")
pairs = pd.read_csv("out/pairs-material-2.csv")
lines = pd.read_csv("out/lines-material-2.csv.gz")   # pandas reads gzip directly
front_lines = lines[lines.pair.isin(pairs.pair[pairs.both_front == 1])]
one = lines[lines.pair == 1803]                       # one line: F1, F2 over t
d = np.load("out/lines-material-2.npz")               # d["lines"][p, k] = (F1, F2) at t = d["t"][k] on pair d["pairs"][p]
```

## 5. Using the chart

* **Axes relative to Chevron / absolute** (default relative): tick labels, tooltips and readouts in multiples of the Chevron reference,
  or in the simulator's units (F1, F2 in m³). The dashed lines mark Chevron's F1 and F2; the shaded area is worse than Chevron in at least
  one objective; the dark diamond is Chevron with its noise cloud. When Chevron lies outside the view, an arrow at the edge points to it.
* **Zoom:** scroll wheel at the cursor (with Shift only F1, with Alt only F2), drag to pan, Shift-drag a box to zoom into it, pinch and
  drag on touch screens, `+`, `-`, `0` with the chart focused. The buttons above the chart zoom in and out, fit the front with its noise
  clouds (*Front*), show all lines (*All*, also a double-click) or extend the view to include Chevron (*With Chevron*). The zoom factor is
  shown next to them; the smallest view is about a fifth of the noise of one simulation.
* **Lines between front solutions / all final solutions**: the 253 lines among the 23 front solutions, or all 3,403 among the 83 final
  solutions of the three runs (hollow circles: dominated ones; their clouds are shown in this mode too).
* **Color uniform / by schedule distance**: by the Euclidean distance between the two deposition schedules (dark: near, light: far in
  the light theme; the reverse in the dark theme). Near pairs are drawn on top.
* **Line opacity**: the transparency of the faint lines (2 % suits all pairs, 10 % the front; refined lines zigzag through the noise of
  neighbouring points, so they carry more ink than straight segments).
* **Single simulations**: the noise cloud of every solution (128 single simulations at L1; the dot is their mean). The points of the
  lines have no clouds; each is the mean of 32 simulations.
* **Interpolated points that beat the front**: the 4,108 confirmed points, drawn at their objectives from 32 fresh simulations.
* **Hover** a solution: its lines and its cloud are highlighted. **Click** (or choose under *Pin*) to pin it, then move along one of its
  lines: the tooltip and the panel show the point (t, F1, F2), the partner solution, the schedule distance, both deposition schedules and
  the blended one at t, and F1 and F2 along the whole line against the straight chord, with a tick for every simulated point (dense ticks:
  where the refinement added points). **Esc** or a click on empty space releases it.
* Below the chart: the main findings, the verification table and the parameters.

The F1 axis of the initial view ends at 0.55, so Chevron (F1 0.769) is to the right of it; *With Chevron* brings it into view.

## 6. Results

**Verification** (step 1):

* All 8 T6 materials regenerate exactly from the bundle seed.
* The 83 stored solutions simulated again (16 each): correlation with the stored labels 0.994 (F1) and 0.999 (F2), mean difference
  −0.0003 (F1) and +0.018 m³ (F2), mean absolute difference 0.0031 and 0.080. Differences in standard errors: sd 1.09 (F1) and 1.21 (F2)
  where 1 is expected, |z| < 2 for 93 %. So the stored labels are reproduced within the simulator noise; F2 scatters slightly more than
  the noise estimate predicts.
* Fronts: 23 non-dominated solutions on the stored labels, 19 on the fresh ones, 17 in both. Hypervolume (normalized to the stored
  front, reference point 1.1): stored 1.044, the same solutions with fresh labels 1.033, the fresh front 1.035.
* The three optimizations run again (78 s): 56 final solutions, a front of 12, hypervolume 1.174 (runs 1.14 / 1.16 / 0.97). A
  comparable, here slightly better front: run-to-run variation, as the optimizer code is unchanged since the T6 build.

**What the lines show** (on the fast simulator at L1, deliberately cheap; to be checked at a higher detail level before relying on it):

* **Blending two good schedules mostly gives a worse one.** Halfway between two front solutions the median F2 is 12.8 m³, against
  4.7 m³ on the straight chord between their objectives (F1 0.246 against 0.179). The lines bow far up and to the right; F2 suffers most.
  (With the front of the 32-simulation labels, 22 solutions, the same median was 9.7 m³: the statistic depends on which solutions
  form the front.)
* **The front itself depends on the number of simulations.** Represented by the means of 128 new simulations instead of the earlier 32,
  7 solutions join the front and 6 leave it (22 before, 23 now). The two estimates agree within their noise (differences of 1.06
  standard errors in spread), but neighbouring front solutions lie only about 0.005 apart in F1, three standard errors of a
  32-simulation mean.
* **Some interpolated points beat the front:** 4,108 of 404,038 stay non-dominated by every final solution after 32 fresh simulations
  (2,905 with 8-simulation lines, 2,059 before the refinement), and together they dominate all 23 front solutions.
* **They come from near-identical schedules:** median schedule distance 3.0 m for the pairs that produced them, 57 m over all pairs. The
  interpolation acts as a local search around a front that had not converged (consistent with the better rerun), not as a useful
  recombination of distant schedules.
* **The objectives change in steps, not smoothly.** When the refinement bisects a segment, the larger half carries 83 % of the jump in
  the median (a smooth change splits about evenly), and in 64 % of the cases only one half stays above the threshold; so the number of
  refined segments shrinks slowly from round to round (59,000 to 29,000) instead of collapsing, and 26,625 segments still jump at the
  finest spacing of 1/2560. The seeded scan shows why: with a fixed seed, F1 and F2 along 1/40 of an 82 m line change at only 18 of
  1,000 steps and are exactly constant in between, at the same 18 positions for both seeds. So the steps are deterministic, not noise:
  presumably particles changing their 1 m cell (at `ppm3` = 1 a particle is 1 m³) when a stacker position crosses a cell boundary, about
  700 along that line. One step changes F1 by up to 0.03 (three times the noise of one simulation) and F2 by up to 0.7 m³. The expected
  objectives of the L1 simulator are thus a staircase in the deposition variables, with zero gradient almost everywhere. This matters for
  gradient-based optimization through a model and for what a smooth surrogate can learn; at L4
  (0.25 m particles) the steps should be more numerous and smaller, not yet checked.
* **Noise:** one simulation scatters by about 0.0105 in F1 and 0.26 m³ in F2 around a solution (the clouds); a point of a line (32
  simulations) by about a sixth of that (standard error 0.0021 and 0.035 m³).

## 7. The resolution ladder (second chart)

**Question.** How do the optimized solutions, and their robustness against simulator noise, change with the particle resolution of
the simulator? The front was optimized at L1 (`ppm3` = 1: one particle per m³, 1 m particles); the target simulations of the project are
far more detailed.

* **The chart:** [`out/resolution-material-2.html`](out/resolution-material-2.html), self-contained (0.7 MB), with a method,
  reproduction and figure-caption section at the bottom.
* **The static figure for print:** `out/resolution-figure-material-2.pdf` (vector, clouds rasterized), `.svg`, `.png` (300 dpi):
  (a) the clouds and means of all levels relative to Chevron, with the front of each level; (b) noise against `ppm3`; (c) cost against
  `ppm3`. A caption is on the page.
* **Reproducing:** `ENV=/some/env apps/bmh_ml/src/bmh_ml/front_study/run_resolution.sh 2` (about 15 minutes; independent of the
  interpolation steps): `verify.py` (the stored front and its material), `resolution.py` (the clouds), `time_levels.py` (the cost, run it on an idle
  machine), `build_resolution_page.py`, `plot_resolution_figure.py`.

**Setup.** The 83 final solutions of T6 material index 2 (section 1) and Chevron at maximum speed (19 layers), each simulated 128 times
at six levels of `ppm3` with the reclaim step fixed at 1 m (all levels reclaim the same 60 slices), reclaim angle 45°, eight-likelihood
0.87; 64,512 simulations. Levels: L0 0.125 (2 m particles, coarser than the optimization, added here), L1 1 (the optimization), L2 4,
L3 16, L4 64, L5 256 (0.16 m), named as in `bmh_ml`. Relative values divide by Chevron's mean at the same level.

**Statistics.** Point: the mean of the 128 simulations. ± one run: their standard deviation. Noise of a level: that standard deviation,
averaged over the 83 solutions. Front of a level: its non-dominated means. Shift: mean over the solutions of (level − L5). Rank
correlation: Spearman between the means of a level and of L5 over the 83 solutions.

| Level | ppm3 | particle edge, m | Chevron F1 | Chevron F2, m³ | noise F1 | noise F2, m³ | front | L1 front kept | shift F1 | shift F2, m³ | mean \|shift\| F2, m³ | rank corr. F1 / F2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 | 0.125 | 2.00 | 0.763 | 14.22 | 0.0385 | 0.488 | 5 | 3 of 23 | +0.116 | +3.06 | 3.06 | 0.13 / 0.82 |
| L1 | 1 | 1.00 | 0.768 | 12.66 | 0.0105 | 0.255 | 23 | 23 of 23 | −0.005 | +0.32 | 0.69 | 0.82 / 0.85 |
| L2 | 4 | 0.63 | 0.767 | 11.60 | 0.0056 | 0.102 | 16 | 5 of 23 | +0.002 | −0.44 | 0.61 | 0.89 / 0.92 |
| L3 | 16 | 0.40 | 0.789 | 11.46 | 0.0034 | 0.062 | 18 | 4 of 23 | −0.001 | −0.50 | 0.54 | 0.90 / 0.96 |
| L4 | 64 | 0.25 | 0.818 | 11.35 | 0.0019 | 0.037 | 14 | 4 of 23 | −0.001 | −0.30 | 0.32 | 0.95 / 0.98 |
| L5 | 256 | 0.16 | 0.833 | 11.51 | 0.0012 | 0.021 | 16 | 7 of 23 | 0 | 0 | 0 | 1 / 1 |

**Findings** (one material, the fast simulator; to be repeated on more materials before generalizing):

* **Noise falls steadily with resolution:** the standard deviation of one simulation falls by a factor of 1.6 to 1.9 for every fourfold
  increase of `ppm3` (F1 0.0105 at L1, 0.0012 at L5, 8.5 times less; F2 0.26 against 0.021 m³, 12 times less). At L1 one simulation scatters by about 1.4 % of
  Chevron's F1 and 2 % of its F2.
* **The front of the optimizer's resolution does not survive a finer simulation:** of the 23 solutions non-dominated at L1, 5 are
  non-dominated at L2, 4 at L3 and L4, and 7 at L5. The ranking of all 83 solutions agrees only partly with L5 (Spearman 0.82 for F1 and
  0.85 for F2 at L1, rising to 0.95 / 0.98 at L4).
* **F1 converges, F2 does not yet:** averaged over the solutions, F1 at L1 to L4 differs from L5 by 0.005 or less (a single solution by
  0.016 at L1 and 0.006 at L4). A solution's F2 differs from L5 by 0.69 m³ at L1 and still by 0.32 m³ at L4 (on average 0.30 m³ lower there,
  eight times the L4 noise of 0.037 m³). Chevron moves as well: F1 0.768 at L1,
  0.833 at L5; F2 12.66 and 11.51 m³. The earlier finding (L5 differs from L4 by 0.005 in F1 and 0.3 in F2) is confirmed on
  the optimized solutions.
* **L0 (2 m particles) is a different problem:** F1 rank correlation with L5 0.13, its front has 5 solutions; presumably because particles
  larger than the 1 m reclaim step make the reclaimed slices lumpy (not checked).
* **Cost** (one simulation in a single process, median, Intel i9-9900K, `time_levels.py`): L0 4.3 ms, L1 4.8 ms, L2 7.3 ms, L3 22.5 ms,
  L4 122 ms, L5 536 ms. From L2 on, every fourfold increase of `ppm3` costs 3.1 to 5.4 times as long; noise falls by a factor of 1.6
  to 1.9 for the same step.

**Files of the ladder.**

| File | Content |
|---|---|
| `resolution.py`, `time_levels.py`, `build_resolution_page.py`, `templates/resolution.html`, `plot_resolution_figure.py`, `run_resolution.sh` | The steps of the ladder. |
| `out/clouds-material-2.npz` | Every single simulation: `ppm3` and `names` (6 levels), `samples` (6 × 83 × 128 × 2: level, solution in the order of `verify-material-2.npz`, run, F1/F2), `chevron` (6 × 128 × 2), `seconds` (throughput per run). The L1 level is also used by the interpolation chart. |
| `out/resolution-material-2.csv` | One row per solution and level (498): `row`, `id` (as in the charts), `level`, `ppm3`, `f1_mean`, `f2_mean`, `f1_sd_single`, `f2_sd_single`, `simulations`, `nondominated_at_level`, `on_l1_front`. |
| `out/resolution-data-material-2.json` | What the page embeds: per-level summary (`levels`), `means`, `sds`, quantized `samples` (F1 × 10⁴, F2 × 10³), `chevron_samples`, `provenance`, `timing`. |
| `out/timing-material-2.json` | Single-process time per simulation and level: median, quartiles, runs, CPU. |
| `out/resolution-figure-material-2.{pdf,svg,png}` | The static figure. |
