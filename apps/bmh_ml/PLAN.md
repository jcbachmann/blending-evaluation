# bmh_ml: surrogate models for F1 and F2 (plan, state and how to continue)

The one document of this effort: goal, current state, next steps, facts, decisions, how to run things, conventions and the dated progress
log. Since 2026-10-03 it replaces the handoff `workdir/handoff-bmh-ml-2026-10-02.md` and the older workdir notes (resume prompt, store
transfer notes, data and export instructions). Measured results live in the MLflow store (section 7); this file holds the conclusions
with their numbers. Infrastructure that is not specific to this project (the offload hosts, the job queue) is described in
`~/.claude/CLAUDE.md` on the laptop.

## 0. How to resume

1. Read section 2 (state) and section 3 (next steps and open tasks). Section 9 has the measured results of every step, with dates.
2. `git log --oneline -20`. The repository, this file and the tracking store are the memory, not a chat.
3. Before running anything, read section 7: where the store and the MLflow server are, and how jobs are run on the offload hosts.
4. Domain definitions (F1, F2, ideal stockpile, Chevron, the students' findings) are in the Obsidian vault `Promotion/Entropy`
   (`Entropy/AGENTS.md` section 5, `Concepts/`); check them before explaining or relying on a domain quantity.
5. Before declaring anything done, run the CI-equivalent check (section 7.6) in a fresh environment.

## 1. Goal and how a model is judged

**Goal** (revised with Micha on 2026-10-04). Detailed stockpile simulations take too long for evolutionary optimization, which needs
thousands to hundreds of thousands of evaluations within minutes to hours. The question is whether **fast machine-learning models can
represent a slower, more detailed simulation well enough to replace it inside the optimizer** (NSGA-III over the 20 deposition positions),
for F1 (the volume-weighted standard deviation of the reclaimed quality) and F2 (the standard deviation of the reclaimed volume per
slice from the ideal stockpile). A model is trained for one stockpile setup and must be **material-independent**: evaluable on any
material and deposition of the same lengths and volumes (scope **S2**, 50 material values and 20 depositions). Models for one fixed
material (scope **S1**, the 20 depositions only) have no value of their own; their results are background. The proof of concept uses the
fast simulator at several detail levels (`ppm3`, section 5): cheap levels produce training data, a higher level plays the slow target.
And a pipeline that makes every attempt persisted and comparable without manual bookkeeping.

**The simulator is a deliberately cheap stand-in.** It takes about 1.3 ms per run only because of coarse settings (a high ppm3, so few
particles, and the simplest simulation type). The detailed simulation types take seconds to hours per run. This project is a proof of
concept: if machine-learning-based F1 and F2 work here, directly or through the reclaimed slice profile, the techniques are to be used
with training data from the detailed simulations. So:

* a surrogate is never dismissed because optimizing directly on this simulator is faster; 10,000 simulations per optimization are not
  affordable with the target simulators;
* methods are weighed by **how few training simulations they need** (learning curves), how well they **transfer** to the optimizer's
  region and to **new materials**, and whether their assumptions would hold for the detailed simulations.

**How a model is judged**, side by side, the final choice is made with Micha on that table:

| Aspect | Question | Measured by |
|---|---|---|
| Usefulness for optimization (**decides**) | Does optimizing the model give solutions that are good in the simulator? | transfer test: optimize the model, simulate what it finds (16 repeats), compare with simulator-optimized reference fronts and with Chevron |
| Accuracy where it matters | Is the model right where the optimizer goes? | errors on the operating-region sets (T2, T6, `valop`) relative to the simulator noise |
| Robustness | Impossible or exploitable predictions? | negative predictions, bias on the solutions found, stress set T5 |
| Data efficiency | How many simulations does the model need? | results over the training size (`budget/simulations` of every run) |

**From the vault and the students' work** (review of 2026-09-24): the thesis optimizes relative objectives (F1 divided by Chevron's
F1 of the same material; F2 divided by one sixth of the volume per slice, a tolerance of about 17 % per slice from commit `ceb217d`,
2019, not derived), bmh_ml predicts their absolute numerators. Keith Kumar re-simulated 8,388 front solutions of MLP and XGBoost
surrogates: 79 to 89 % were worse than predicted and only 4.6 % dominated Chevron, the same exploitation this pipeline measures. The
student research questions ("how does the front change when the model's solutions are re-simulated?", "how much does the material
matter?", "how to compare a front with Chevron?") are answered by the transfer test, its per-material form (T6) and the Chevron metrics.

Validation accuracy ranked models the wrong way three times (the best sweep trial, the direct-F2 ablation, S2 on random inputs), so the
transfer test decides. Objectives are also reported relative to Chevron stacking of the same material (19 passes, alternating ends):
(1, 1) means "as good as Chevron" in both objectives.

## 2. State (2026-10-03)

### S1, one material: solved

* Best model: **`profile_mlp`, the hybrid** (`models/keras_models.py`). One network predicts the 60-slice reclaimed profile (volume and
  quality per slice) and F1 and F2 directly; F1 comes from the direct output, F2 is **computed from the predicted profile** (the
  simulator's own formula, `evaluation/profile.py`) plus a noise correction measured on the repeated validation data. Bundle `S1-v2`,
  3 seeds: transfer hypervolume ratio 0.99 (per run 0.76; the old simulator runs 0.57), 97 % of the solutions found beat Chevron, no
  negative predictions, bias on the solutions found -0.13 (F2) and -0.06 (F1). A direct F2 output of the same network is exploited by the
  optimizer, so F2 must go through the profile.
* Refinement (adding the solutions the optimizer finds) was essential for the scalar MLP, but changes little for the hybrid.
* Data efficiency: trained on 10k / 25k / 50k / 100k / 250k random simulations, the hybrid reaches a transfer ratio per run of 0.38 /
  0.36 / 0.54 / 0.56 / 0.76. NSGA-III directly on this simulator reaches 1.05 with 10,000 simulations (about 15 s); with the target
  simulators that comparison turns around (section 1).
* The old reference front T2 (thesis experiments) is weaker than fresh simulator runs, which exceed a ratio of 1; the reference should
  become the best front of those runs.

### S2, any material: the open problem

* Bundle **`S2-v1`**: 500,000 rows with a random material each and profiles, validation 20,000 x 8, T1, T3 (20 unseen materials), T5, T2,
  and **T6: reference fronts for 8 unseen materials** (NSGA-III on the simulator, 16-repeat labels). The transfer test runs per material
  and averages; `hv_ratio_material_min` is the worst material.
* The hybrid (3 seeds) is accurate on random depositions of unseen materials (T3 R2 0.93 for F1, 0.97 for F2), but not on the optimized
  solutions of the 8 new materials: T6 F1 R2 -12 to -13, transfer ratio per run 0.03 to 0.05, beats Chevron 43 to 53 %. NSGA-III on the
  simulator: 0.16 with 5,000 simulations, 0.47 with 10,000.
* Refinement over many materials (`refine-S2-v1-rhybrid`, 20 new materials per round, rounds 1 to 3): T6 F2 R2 rises to 0.65, but F1 stays
  bad (T6 R2 -7 to -9.5) and the transfer does not improve. Round 4 and a second loop were killed by the temperature watchdog
  (2026-10-02, cooling fixed since); their bundles exist, the runs are closed as KILLED.
* **The reclaimed quality mixes the material linearly** (measured 2026-10-03, section 5): F1(a m + b) = |a| F1(m) and
  q(m1 + m2) = q(m1) + q(m2) within the noise. So E[F1^2] = m_c' B m_c exactly, with the centered material m_c and a matrix B that depends
  on the deposition only. Two models use this (3 seeds each on S2-v1, with the transfer test on T6):

  | Variant | T3 F1 R2 | T6 F1 R2 | transfer ratio per run | beats Chevron | F1 bias on found solutions |
  |---|---|---|---|---|---|
  | hybrid (baseline) | 0.93 to 0.94 | -12 to -13 | 0.031 to 0.045 | 43 to 53 % | -0.06 to -0.10 |
  | hybrid, material scaled (`material_scaling=true`) | 0.95 to 0.96 | -9 to -11 | 0.013 to 0.049 | 48 to 65 % | -0.03 to -0.04 |
  | `mixing_mlp` without quality output (`quality_weight=0`) | 0.91 to 0.94 | -14 to -22 | 0.036 to 0.048 | 40 to 53 % | -0.10 to -0.13 |
  | `mixing_mlp` with quality output (default) | 0.92 to 0.94 | -8 to -13 | 0.015 to 0.026 | 51 to 55 % | -0.12 to -0.13 |

  `material_scaling` standardizes each material by its own mean and spread and predicts F1 and the quality profile in those units.
  `mixing_mlp` predicts a factor L of B from the deposition (F1 = |L' m_c|, exact in the scale and offset of the material by
  construction), the volume profile for F2, and optionally the quality profile through predicted mixing weights.
* **Diagnosis:** the mixing model handles new materials exactly and still fails on T6 as badly, and F2 (deposition only) is negative on T6
  for every variant. So the problem is the optimized **depositions** outside the random training distribution, not the new materials: the
  S1 story again. What the mixing model learns about a deposition holds for every material, so refinement data should help it most.
  Material scaling makes F1 less optimistic (bias -0.03 instead of about -0.07), which is worth keeping.
* **Learning curve S2** (bundles `S2-b10k`, `S2-b30k`, `S2-b100k`: random training data of that size, validation a tenth of it x 2, the
  test sets of S2-v1; 500k is S2-v1; means of 3 seeds):

  | Model | T3 F1 R2 (10k / 30k / 100k / 500k) | T6 F2 R2 | transfer ratio per run | beats Chevron | F1 bias on found solutions |
  |---|---|---|---|---|---|
  | hybrid | 0.67 / 0.75 / 0.85 / 0.93 | -9.7 / -5.6 / -2.0 / 0.0 | 0.000 / 0.002 / 0.007 / 0.038 | 15 / 35 / 38 / 47 % | -0.32 / -0.26 / -0.21 / -0.07 |
  | hybrid, material scaled | **0.77 / 0.83 / 0.90 / 0.95** | -10.2 / -6.8 / -2.7 / -0.3 | 0.002 / 0.008 / 0.012 / 0.035 | 29 / 35 / 47 / 55 % | **-0.16 / -0.16 / -0.14 / -0.03** |
  | mixing, no quality output | 0.68 / 0.75 / 0.83 / 0.92 | -9.6 / -7.0 / -2.4 / -0.4 | 0.001 / 0.001 / 0.002 / 0.041 | 28 / 33 / 35 / 46 % | -0.17 / -0.19 / -0.18 / -0.11 |

  Material scaling is the most data-efficient on new materials (at 100k about as accurate as the plain hybrid at roughly 300k) and halves
  the F1 optimism at every size. The mixing model is not more data-efficient on random data, against the expectation: learning how a
  deposition mixes is the hard part, the exact treatment of the material does not shorten it; it is also the worst on T6 F1 (R2 -18 to
  -49). No model transfers on new materials from random data alone, even with 500k; F2 (deposition only) improves with data but reaches
  only R2 about 0 on optimized depositions. Data from the optimizer's region is needed, which the refinement loops test.

### Running

Nothing; the queues were stopped on 2026-10-03 at 22:00. Results of the refinement loops: section 9, last row.

## 3. Next steps and open tasks

### The evaluation plan (agreed 2026-10-04)

Everything is material-independent (S2). **After every phase the next one is re-planned in light of the results, not simply continued:**
the results and the adjusted next phase go to Micha before work goes on. `detailed=True` of the fast simulator is not used (in the
current configuration it is a concept, not a realistic simulation); the detail ladder uses `ppm3` with the reclaim step fixed at 1 m.
Effort: implementation in human-equivalent days, compute in wall time on both machines.

| Phase | Content | Implementation | Compute |
|---|---|---|---|
| **0 Foundations** | (a) The detail level (`ppm3`, reclaim step 1 m) as a setting of every dataset, test set, Chevron reference and transfer test; levels L1 = 1, L2 = 4, L3 = 16, L4 = 64 (target). (b) **Cross-fidelity study without ML** on new materials: random and optimized solutions (NSGA-III on L1 and on L4) evaluated at every level; rank correlations; fronts in both directions. This is the baseline "optimize on the cheap simulator". (c) Check the measuring instrument (the worst material is always 0). (d) Reference fronts and budget curves at L4. (e) Literature review (surrogate-assisted and offline model-based optimization, multi-fidelity, Pareto set learning) into the vault. | 2 days | about 1 day |
| **1 Multi-fidelity** | The material-scaled hybrid trained on L1 only, L4 only, and L1 plus few L4 (fine-tuning, or learning only the difference); learning curves over the number of L4 simulations; evaluated at L4 (T3, T6, transfer test) | 1.5 days | 1 day |
| **2 Surviving optimization** | F1 computed from the predicted quality profile (with linear mixing, material scaled) against direct F1; genuinely optimal solutions of many materials plus exploited ones in the training data; ensembles with an uncertainty penalty; conservative training; a small comparison of families and settings judged by the transfer test | 2.5 days | 1.5 days |
| **3 Few simulations per new material** | only if phases 1 and 2 leave zero-shot short: warm start from the general model, simulate the best candidates, update; against NSGA-III at L4 | 2 days | 1 day |
| **4 Solution generators** (exploratory) | a model proposes near-optimal depositions per material, directly and as the starting population of NSGA-III on the simulator; measured in simulations saved | 2 days | 1 day |
| **5 Synthesis** | SSCI abstract (by about 25 October), vault overview, this plan | 1 day | |

Leftover from 2026-10-03, decide in phase 0: the two material-scaled refinement loops `rrel-s2`, `rrel-s3` stopped after round 3 (they
continue with `--resume`); they ran at L1 and are only worth finishing if L1 results still matter after the cross-fidelity study.

### Open questions for Micha

1. **Ideal stockpile of F2** (finding of 2026-10-01, section 5): the simulated pile ends taper inward over 4 to 5 cells instead of forming
   the ideal half cones, so part of every F2 is a fixed offset. Is that expected (reclaimer or slice coordinates of the simulator), and
   should F2 use the integrated derivation of the vault? Profiles are stored, so F2 can be recomputed for any definition without
   simulating.
2. **Where the store lives.** Runs are logged into `~agent/offload/BlendingEvaluation/workdir/bmh-ml-store` on micha-pc. On
   2026-10-03 at 22:30 it was copied completely to the synced `workdir/bmh-ml-store` (161 runs, 3.1 GB, paths fixed), so both are
   identical. Keep this pattern (micha-pc logs, the synced copy is refreshed at the end of a working day), or move the store out of the
   synced folder?
3. **Linearity at higher detail.** The mixing model assumes that quality is a passive label of the particles; checked at `ppm3` = 1
   only. Checked again at L4 in phase 0.
4. **GPU** (Navi 10, no ROCm): only worth an unofficial ROCm setup for larger models or ensembles (B5); a system change, Micha's call.

### Tasks for Micha

1. **Rebuild the repository environment on the laptop.** On 2026-10-03 an agent's `uv run` replaced the laptop's `.venv` (it pointed to a
   Python that did not exist there) with an empty one; `.venv` is not synced, the other machine is unaffected:
   `uv sync --locked --all-packages --dev` in the repository.
2. **Convert the 18 exported `.FUN` files** with the old header line (`E1` to `E6 ... Evaluation ... (non-dominated).FUN` and one in the
   top folder of `Experiments`); jMetal fails on them (`could not convert string to float: '"F1/Ash'`). The script removes the header,
   writes the objective names to an `.OBJ` file next to it, keeps the numbers and first copies the originals to
   `Experiments/fun-header-backup` (tested on a copy of exactly these files):
   ```shell
   REPO=/home/mci/Sync/micha-data/University/Promotion/Code/BlendingEvaluation
   EXPERIMENTS=/home/mci/Sync/micha-data/University/Promotion/Experiments
   python3 "$REPO/workdir/convert-headered-fun-files.py" "$EXPERIMENTS"            # lists the 18 files
   python3 "$REPO/workdir/convert-headered-fun-files.py" "$EXPERIMENTS" --write    # converts them
   python3 "$REPO/workdir/convert-headered-fun-files.py" "$EXPERIMENTS"            # must report: 0 file(s) would be converted
   ```
   Delete `Experiments/fun-header-backup` when satisfied; `export_fun` writes the new format itself since 2026-09-20.
3. **Clean up** when convenient: `workdir/eaf-sample/` (30 MB, plot demo). In `apps/bmh_ml/src/bmh_ml/data` (git-ignored) the files of
   the first experiments: `training_data.csv` is truncated (1,785 usable rows of 250,000), `lstm_model_f{1,2}.keras` belong to it, and
   `scaler.pkl` belongs to neither model set exactly. Keep `lstm_model_f{1,2}_random_training_data.keras` until M4 replaces the model
   sets of the old scripts. Regenerating the data and retraining the old LSTM model sets is no longer needed: the pipeline replaces it.

## 4. Roadmap and backlog

Durations are human-equivalent working days; real elapsed time is set by compute and by reviews, and is reported per milestone.

* **M0 facts, plan, tooling check**: done 2026-09-20.
* **M1 training and evaluation pipeline** (dataset builder, model interface and baselines, evaluation and MLflow logging, `report`, `ui`):
  done 2026-09-20. Acceptance: three documented commands train the baselines and show them with noise ceilings; a second run reproduces
  the metrics exactly; CI green.
* **M2 operating-region data and the deployment check** (refinement loop, transfer test as a pipeline stage, Optuna sweeps as nested
  runs): done 2026-09-24. Finding: the training distribution changes the usefulness for optimization much more than the accuracy on T2,
  and the operating region is specific to each model.
* **M3 modeling iterations** (open ended, in progress): work through the backlog by expected value; every experiment gets a hypothesis,
  its runs in MLflow and a dated row in section 9. Stop rule for an idea: it beats the current champion in the transfer test (seed spread
  considered), or it is documented as tried. Check in with Micha after each item that changes the picture.
* **M4 champions into the repository** (about 1 day): registry aliases `champion-S1`, `champion-S2` that the scripts load; fast batched
  inference for the optimizer; end-to-end check with the real optimization scripts.
* **M5 report and cleanup** (about 1 day): final comparison table (transfer, accuracy, robustness, data efficiency) and the recommendation;
  remove the obsolete trainer paths, update the documentation.

| # | Idea | Hypothesis | Status |
|---|---|---|---|
| B1 | Operating-region training data (refinement loop) | the largest gain where the optimizer goes | done for S1 (essential for the MLP, little effect on the hybrid); S2: hybrid not helped, mixing model running |
| B2 | Positive outputs, log scale, loss weighted to the good region | relative errors matter, impossible values disappear | positivity solved by computing F2 from the profile; the rest open |
| B3 | Structured deposition data and pile features | cheaper coverage of good regions | open |
| B4 | Structure of F1: linear mixing of the material | far better generalization across materials | linearity confirmed; `mixing_mlp` built and measured, refinement running |
| B5 | Architectures (wider, residual, convolution over material or deposition, attention), deep ensembles | sequence models, uncertainty against exploitation | open |
| B6 | LightGBM and Gaussian processes for S1 | strong on small, well-placed data | LightGBM done (refined 0.74, poor after sweeps); GP open |
| B7 | Predict the reclaimed profile, compute F1 and F2 from it | richer supervision | done: the hybrid is the S1 champion |
| B8 | Learning curves over the data size, repeated labels, heteroscedastic loss | what more data buys | S1 measured; S2 running |
| B9 | Inference speed (vectorized models, export) | matters when the surrogate runs in the optimizer | Keras call fix done (79 ms to 6 ms per batch of 100); rest with M4 |
| B10 | Gradient-based optimization through a differentiable model | methods the noisy simulator cannot offer | open |
| B11 | Uncertainty guards (penalize ensemble disagreement) | removes exploitation of model errors | open |
| B12 | Multi-fidelity (cheap noisy simulations as extra data) | more data per second of compute; directly relevant for the detailed simulators | open |

## 5. Facts

**The simulator is noisy, which caps every model** (the same input simulated 30 times, 2026-09-20):

| Region | F1 noise sd | F1 spread sd | best possible R2 (F1) | F2 noise sd | F2 spread sd | best possible R2 (F2) |
|---|---|---|---|---|---|---|
| fixed material, random depositions | 0.0080 | 0.152 | 0.997 | 0.137 | 3.81 | 0.999 |
| fixed material, **optimized** depositions | 0.0093 | 0.035 | **0.929** | 0.249 | 3.51 | 0.995 |
| random material and deposition | 0.0071 | 0.208 | 0.999 | 0.145 | 4.46 | 0.999 |

In the region that matters, F1 differences below about 0.01 are noise; test labels are means of 16 simulations and errors are read in
units of the noise (`nrmse`). `build_bundle` prints this table for every bundle.

**Random data does not cover the optimizer's region.** Fixed material: random depositions have F1 0.30 / 0.48 / 0.76 and F2 16.9 / 24.1 /
31.9 (5 / 50 / 95 %), the thesis fronts F1 0.05 to 0.47 and F2 2.9 to 24.3; 100 % of the optimized solutions lie below the 1st percentile
of random data in at least one objective. The first surrogate optimizations predicted negative F2 for 77 % of their front solutions.

**Linear mixing of the quality** (2026-10-03, 4 depositions, 64 repeats each): stretching a material around its mean by 2 multiplies F1 by
1.999 to 2.008, shifting it changes nothing, and the quality profile of m1 + m2 equals the sum of the two profiles except in nearly empty
slices (volume below 4, within 1 to 2 standard errors). Empty slices store the quality 0. The material spread (sd of the 50 values) of
S2 rows ranges from 0.17 to 2.1.

**Profiles.** The simulator returns 60 reclaimed slices (volume and quality); F1 and F2 are exact functions of them. The F of a mean
profile is slightly below the mean F of noisy simulations: about +0.001 in F1^2 and +1.1 in F2^2 (S1-v2), measured from repeated data
alone (`get_noise_correction`).

**Detail levels of the fast simulator** (2026-10-04; one material, 4 random depositions and Chevron, 4 to 8 repeats). `ppm3` is the
number of particles per cubic meter (1.0 so far, about 2,500 particles for the pile). By default the reclaim step is `1/sqrt(ppm3)`, so
at higher `ppm3` the simulator reclaims more, thinner slices (472 at `ppm3` = 64) and F2, a spread of volume per slice, shrinks with
the slice width (about half per fourfold `ppm3`). With the reclaim step fixed at 1 m (60 slices, as the profile models expect):

| `ppm3` | ms per simulation | F2 of the 5 inputs (Chevron last) | F1 of the 5 inputs | noise sd F1 / F2 |
|---|---|---|---|---|
| 1 | 2.5 | 30.6 25.1 24.2 25.4 12.6 | 0.252 0.380 0.331 0.267 0.373 | 0.0055 / 0.13 |
| 4 | 4.7 | 30.8 24.5 23.6 26.2 11.6 | 0.231 0.373 0.322 0.260 0.363 | 0.0027 / 0.08 |
| 16 | 17 | 31.3 24.5 23.9 26.4 11.5 | 0.226 0.371 0.330 0.267 0.371 | 0.0014 / 0.045 |
| 64 | 91 | 32.0 24.9 24.1 27.0 11.4 | 0.231 0.365 0.331 0.264 0.372 | 0.0005 / 0.018 |

So the real effect of detail on the objectives is small here (F2 up to 10 %, Chevron's F2 improves from 12.6 to 11.4; F1 up to 8 %),
the noise falls tenfold, and the cost grows about linearly with the particles above `ppm3` = 4. Whether optimized solutions keep their
order across levels is phase 0 (b). `detailed=True` costs about 64 s per run and changes F1 by 37 %, but in the current configuration it
is a concept, not a realistic simulation, and is not used.

**Ideal stockpile of F2** (2026-10-01): F2 uses `bmh.helpers.stockpile_math.get_ideal_stockpile_volumes` (the vault note "Ideal
Stockpile", the cut area at one point per slice). The newer derivation in the vault (`Concepts/Stockpile Math/Ideal Stockpile
Derivation`, integrated over each slice) ramps about one slice later and sums exactly to the total volume (the point rule gives 2503.5 for
2500). Even Chevron (64 runs) reclaims nothing before slice 7, about 4 slices later than either ideal, and overshoots to about 62 in
slices 12 to 14. Cause, measured on the pile: the ideal has the full ridge height 7.3 from x = 10 to 49 with half cones out to 2.7 and
56.3; the simulated pile starts at x = 5, reaches full height only at about x = 14, falls off from about x = 43 and ends at about 53. The
simulator reclaims a particle at column x and height h in slice x - h, and the binding labels the slice between positions k-1 and k as
x = k (one slice of offset).

**Cost.** One NSGA-III run with 100,000 evaluations took 106 s on the first LSTM surrogate and 49 s on the simulator (16 cores); a
surrogate earns its place by data efficiency on expensive simulators, batched inference, smoothness or differentiability, not by speed
against this simulator. The first "LSTM" is a dense network: it sees sequences of length one.

**Hardware.** micha-pc: i9-9900K (8 cores / 16 threads), 62 GB RAM, AMD Navi 10 GPU without ROCm (unused); the cooling was fixed on
2026-10-03, all cores may be used. Laptop: Ryzen 7 5825U (8 cores / 16 threads), 37 GB RAM, used by Micha, so jobs there run at the
lowest priority. A single training of these networks uses only about 3 cores well; several jobs side by side use a machine better than
more threads per job. LightGBM is fastest with 8 to 12 threads.

## 6. Decisions

| Decision | Choice | Why |
|---|---|---|
| Experiment tracking | MLflow, SQLite backend, file artifacts, one experiment per scope, one run per training, sweeps and refinement rounds as nested runs | runs, parameters, artifacts and a comparison UI without an account; works with any framework |
| Model frameworks | Keras 3, scikit-learn, LightGBM; PyTorch only if needed | models sit behind one interface (`fit`, `predict`, `save`, `load`) |
| Configuration | `--param key=value` on the defaults of the model classes, no configuration framework | defaults live next to the code, a run logs its complete parameters |
| Hyperparameter search | Optuna, every trial a nested run | results in the same UI |
| Labels | single simulations for training, 8 repeats for validation, 16 for test sets | noisy test labels would hide differences between good models |
| Test sets | frozen bundles, model selection on validation data only | many attempts would otherwise overfit the test sets |
| Dependencies | heavy packages (TensorFlow, LightGBM, MLflow, Optuna) only in `bmh_ml`, their tests behind `importorskip` | CI installs only the dev group and must stay green |
| Scope priority | track S1 and S2; S2 is where a surrogate pays | (2026-10-02, after the budget comparison) |
| Reference for relative objectives | Chevron with 19 passes, alternating ends, cached per material (`references/`); F1 and F2 both divided by Chevron's | (2026-09-24, with Micha) |
| Order of the work | quality of the results; the SSCI 2027 abstract deadline (1 November 2026) does not set it | (2026-09-24, with Micha) |
| Compute | CPU only; runs on the offload hosts, micha-pc main, laptop helping at low priority | (2026-10-03) |

## 7. How to run things

### 7.1 Environments

* **Never touch the repository's `.venv`**, and never run `uv run` in the repository without `UV_PROJECT_ENVIRONMENT` set to an
  environment outside it (on 2026-10-03 a plain `uv run` replaced the laptop's `.venv`). On the offload hosts the environment is
  `~agent/offload/env` (`UV_PROJECT_ENVIRONMENT=$HOME/offload/env uv sync --locked --all-packages --dev`, under a minute).
* `uv sync --package bmh_ml` would remove the other workspace packages; use `--all-packages`.

### 7.2 Where the code, the data and the runs are

* Checkout on both offload hosts: `~agent/offload/BlendingEvaluation`, synced from the laptop with
  `rsync -a --delete --exclude-from=.gitignore --exclude=.git ./ <host>:~/offload/BlendingEvaluation/` (git-ignored data separately).
* **Store** (datasets, bundles, MLflow database and artifacts): selected with `BMH_ML_STORE`; without it a new empty store starts at
  `~/bmh-ml-store`. The store that receives new runs is `~agent/offload/BlendingEvaluation/workdir/bmh-ml-store` on micha-pc. The
  laptop's agent has a copy of its datasets, bundles and artifacts (no database). The synced `workdir/bmh-ml-store` was refreshed from
  micha-pc on 2026-10-03 22:30 and is identical to it then (open question 2). To refresh it: snapshot the database on micha-pc with
  SQLite's backup (`python3 -c "import sqlite3; sqlite3.connect('mlflow.db').backup(sqlite3.connect('/home/agent/offload/mlflow-snapshot.db'))"`
  in the store, safe while the server runs), rsync the store without `mlflow.db` and the snapshot as `mlflow.db` into the synced
  `workdir/bmh-ml-store`, then run `workdir/fix-mlflow-store-paths.py` on it. 23 runs have no artifact directory on purpose (refinement
  parents, runs killed before they saved anything).
* **MLflow server** on micha-pc over that store: `python -m bmh_ml.ui --port 5055` in the scope `mlflow`, localhost only. Jobs log to it
  with `BMH_ML_TRACKING_URI=http://127.0.0.1:<port>`, which `tracking/store.py` prefers over the SQLite file (not
  `MLFLOW_TRACKING_URI`: MLflow sets that one itself). The laptop's agent reaches it
  through its own tunnel (its key may forward only to port 5055 on micha-pc). Both agent accounts have the same home path, so runs on the
  laptop write their artifacts to the same path there; the queue copies them to micha-pc after each job.
* **Moving a store:** MLflow keeps absolute artifact paths in its database. After copying a store to another path, run
  `python3 workdir/fix-mlflow-store-paths.py <store>` once (idempotent; it rewrites only paths whose `artifacts` root does not match).

### 7.3 Running jobs

Long jobs go through the job queue (`~/offload/queue/` on both hosts, documented in `~/.claude/CLAUDE.md`; a copy of its scripts is in
`workdir/offload-queue/`, the job logs of 2026-10-03 in `workdir/logs/queue-2026-10-03/`): write a job script into
micha-pc's `~/offload/queue/pending/`, the runners start it when cores are free, the laptop takes jobs that do not write datasets. A job
for this project starts with `source ~/offload/queue/env.sh` (checkout, store, MLflow port, `R="$HOME/offload/env/bin/python -m"`) and
declares `# cores: N`. The laptop copies new datasets, bundles and run artifacts from micha-pc before a job and back after it (datasets
are content-addressed and bundle names unique, so this is safe); bundle and refinement names must therefore be new. Jobs run the code of the
host's own checkout: after changing code, sync it to **both** hosts before queueing jobs that need it. Jobs that need a bundle
another job builds declare `# needs: ~/offload/BlendingEvaluation/workdir/bmh-ml-store/bundles/<name>.json`. On micha-pc every process of
a closed SSH login is killed, so anything started by hand must use `systemd-run --user --scope --unit=<name> tmux new -d -s <name> "..."`.

**Stopping and restarting.** `bash ~/offload/queue/stop-jobs.sh [grace seconds]` on each host stops its queue gracefully (no new jobs,
Ctrl+C to the running ones, terminated after the grace period; the laptop still copies its results back), then
`bash ~/offload/queue/stop-server.sh` on micha-pc waits until no job runs anywhere, closes runs left open as KILLED and stops the MLflow
server. Nothing of this survives a reboot. To start again: on micha-pc the MLflow server
(`systemd-run --user --scope --unit=mlflow tmux new -d -s mlflow "cd ~/offload/BlendingEvaluation && BMH_ML_STORE=$PWD/workdir/bmh-ml-store ~/offload/env/bin/python -m bmh_ml.ui --port 5055 > ~/offload/logs/mlflow-server.log 2>&1"`),
then remove `~/offload/queue/stop` on both hosts and start the runners (micha-pc: `systemd-run --user --scope --unit=queue tmux new -d -s queue
"bash ~/offload/queue/runner.sh"`; laptop: the same with `QUEUE_HOST=laptop QUEUE_REMOTE=micha-pc QUEUE_MLFLOW_PORT=5057 nice -n 19 ionice -c 3`
before `bash`). An interrupted refinement loop is queued again with its original command plus `--resume`.

### 7.4 Commands

`python -m bmh_ml.<command>`, details in [README.md](README.md): `build_bundle` (`--profiles`, `--new-val`, `--tests-from`,
`--material-fronts K`), `train` (`--seed 1 2 3 --workers 3 --transfer --transfer-reference T6`), `refine` (`--materials-per-round`,
`--start-run`, `--control`), `transfer` (`--simulator-baseline 5000 10000`, `--recompute`), `sweep`, `report`, `ui`,
`tracking.annotate --all`, `tracking.views`. Models: `mean`, `ridge`, `lightgbm`, `mlp`, `legacy_lstm`, `profile_mlp` (the hybrid;
`material_scaling` for S2), `mixing_mlp` (S2).

**Datasets.** A bundle names the training data, the validation data and the frozen test sets; it cannot be overwritten, `--tests-from`
reuses the test sets of another bundle. Every dataset is stored under its content hash with its manifest.

| Set | Content | Purpose |
|---|---|---|
| `val` | random inputs, 8 repeats | early stopping, model selection |
| `valop` | solutions of separate optimizations of every refinement round's model, 8 repeats | selection for the operating region (refined bundles) |
| T1 | random inputs | accuracy as usually reported |
| T2 | solutions of the thesis optimizations on the simulator (fixed material) | operating region; reference front of the S1 transfer test |
| T2s | solutions of the old surrogate optimizations | exploited regions |
| T3 | 20 unseen random materials x 50 depositions (S2) | generalization to new materials |
| T5 | extreme depositions (edges, constants, jumps) | robustness |
| T6 | NSGA-III fronts on the simulator for 8 unseen materials (S2) | operating region of new materials; reference of the S2 transfer test |

### 7.5 Watching runs

The MLflow UI (on the laptop through a tunnel to port 5055, `ssh -N -L 127.0.0.1:5055:127.0.0.1:5055 agent-micha-pc`, then
http://127.0.0.1:5055): experiments `S1-fixed-material` and `S2-general-material`, switch to "Model training", saved views ("1 Leaderboard",
"3 Refinement loops", ...). Every run has a generated description with links. Keras trainings log `epoch/loss` and `epoch/val_loss` per
epoch (live curves) and print a line per epoch to their log.

### 7.6 Before declaring something done

The CI-equivalent check in a fresh environment outside the repository:
`uv sync --locked --all-extras --dev && uv run ruff check . && uv run ruff format --check . && uv run pytest`, plus the `bmh_ml` tests with
all packages (`uv sync --locked --all-packages --dev`, `pytest apps/bmh_ml`). Last run 2026-10-03: 374 passed / 29 skipped with the dev
group only, 226 `bmh_ml` tests pass with all packages (jobs `50-ci-check` and `51-full-tests` of the queue do both).

## 8. Conventions and pitfalls

**Conventions (from Micha)**

* Commit messages: one imperative line in the style of the history, no body, no Co-Authored-By trailer. Micha pushes, agents never do.
* Every milestone and every experiment that changes the picture gets a dated row with measured numbers in section 9; results that depend
  on the simulator being cheap say so.
* MLflow runs carry a human-readable description (`tracking/annotate.py`); `train --description` says why a run was made.
* Results are measured, not assumed; the CPU is used fully for long computations (the laptop at low priority).
* Every result exists in the tracking store or in the repository, never only in a chat. Small decisions are taken and recorded here;
  anything that changes the meaning of "best" or costs Micha time or money is asked.

**Pitfalls that cost time**

* `pkill -f "<pattern>"` also kills the calling shell when the pattern is in its command line. Select by PID, or match the environment
  path only.
* Simulation worker pools start with **spawn** (forking a process with TensorFlow threads deadlocked once). One-off scripts that simulate
  need an `if __name__ == "__main__":` guard.
* Parallel runs registering the same dataset in MLflow collided once on a UNIQUE constraint; logging retries now.
* Results added to a finished run must be logged by run id (`log_metrics_to_run`), not by reopening it, or its end time moves.
* Results trained on one machine have their artifacts there until the queue copies them; a refinement started from such a run fails
  with "Failed to download artifacts".
* Bundles are frozen: a resumed refinement loop needs a new name (or a one-off completion script, used once for `S1-v2-rhybrid`).
* Small tests of learned behavior need enough data to be stable: `mixing_mlp` on 300 simulations gave R2 -0.26 to 0.31 across seeds,
  on 2,000 0.37 to 0.43.

**Risks:** overfitting the test sets through many attempts (frozen sets, selection on validation data); reading noise as signal (noise
ceilings next to every metric, seed spreads before stating an effect); the store growing without bound (old runs can be archived).

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
| 2026-10-02 | **Refinement over many materials** (`refine --materials-per-round 20`, 2 NSGA-III runs per material and round, `refine-S2-v1-rhybrid`, from the S2 hybrid seed 1; about 11,000 rows added per round). Rounds 1 to 3: T6 F2 R2 0.09 to 0.63 / 0.69 / 0.65, but T6 F1 R2 stays -7 to -9.5 and the transfer on the 8 unseen materials does not improve (hypervolume ratio per run 0.074 / 0.042 / 0.030, beats Chevron 56 / 46 / 56 %, worst material 0-1 %, F1 bias -0.09 to -0.13). **F1 for new materials is the blocker.** Round 4 and a second loop (seed 2, round 1) were stopped by the temperature watchdog at 100 C (two loops in parallel, load about 30); runs closed as KILLED, bundles kept. Optimizations of all materials of a round or a test now run in one parallel batch. Next: F1 relative to the material's own spread, a sequence model over the material curve. Handoff for the next agent: `workdir/handoff-bmh-ml-2026-10-02.md`. |
| 2026-10-03 | **Linear mixing, two material-aware models, offloaded runs.** Measured that the reclaimed quality mixes the material linearly (F1(a m + b) = \|a\| F1(m) within 0.4 %, profiles add up), so E[F1^2] is a quadratic form in the centered material with a matrix that depends on the deposition only. New: `profile_mlp --param material_scaling=true` (each material standardized by its own mean and spread) and the model `mixing_mlp` (F1 = \|L' m_c\| with L predicted from the deposition, F2 from the predicted volume profile, optional quality profile through predicted mixing weights), both exact in scale and offset of the material; per-epoch progress in MLflow; `BMH_ML_TRACKING_URI` selects a tracking server. S2-v1, 3 seeds each, transfer on T6 (ratio per run / beats Chevron / F1 bias): hybrid 0.031-0.045 / 43-53 % / -0.06 to -0.10, material-scaled 0.013-0.049 / 48-65 % / -0.03 to -0.04, mixing without quality output 0.036-0.048 / 40-53 % / -0.10 to -0.13, mixing with quality output 0.015-0.026 / 51-55 % / -0.12 to -0.13 (2 hours per training, the 3,000 mixing weights); T6 F1 R2 -8 to -22 for all. **Neither fixes the transfer; as the mixing model handles new materials exactly, the failure lies in the optimized depositions, not in the materials.** Runs now execute on the agent accounts of micha-pc and the laptop through a job queue with one MLflow server (section 7). The simulator was clarified as a deliberately cheap stand-in for the detailed simulations (section 1). Running: refinement over many materials for the mixing model and the material-scaled hybrid (3 seeds, one with a control). The workdir plan, handoff and note files were merged into this document. **Learning curve S2** (section 2; 10k / 30k / 100k / 500k random
simulations, 3 seeds each): material scaling is the most data-efficient (T3 F1 R2 0.77 / 0.83 / 0.90 / 0.95 against 0.67 / 0.75 / 0.85 /
0.93 for the hybrid) and halves the F1 optimism; the mixing model is not more data-efficient; nothing transfers to new materials from random
data alone (per run at most 0.04). |
| 2026-10-03 | **Refinement over many materials, second attempt** (20 new random materials per round, 2 NSGA-III runs each, transfer test on T6 per round; the loops `rrel-s2` and `rrel-s3` were stopped for the night after round 3, `--resume` continues them). Transfer ratio per run, round 0 -> best round (beats Chevron): material-scaled hybrid, 3 seeds: 0.013 -> 0.060 (48 -> 71 %), 0.043 -> 0.078 (65 -> 63 %), 0.049 -> 0.066 (51 -> 60 %); mixing model without quality output: 0.048 -> 0.039 (53 -> 50 %, no gain); for comparison the plain hybrid's loop of 2026-10-02 0.045 -> 0.074 (43 -> 56 %). The worst material stays at 0 in every round of every loop, T6 F1 R2 at best -5, F1 bias on the solutions found -0.04 to -0.08 (material-scaled) and -0.10 to -0.13 (mixing). NSGA-III on the simulator per material: 0.16 with 5,000 and 0.47 with 10,000 simulations. **Refinement over many materials buys at most a small, noisy gain; a model trained once does not reach the optimizer's region of a new material. Material scaling is the only change that helps consistently.** |
| 2026-10-04 | **Goals and evaluation revised with Micha** (section 1, section 3): fast models that represent a slower simulation inside the optimizer, material-independent (S2 only; S1 is background), shown with the fast simulator at several detail levels; six phases, each re-planned after the results of the one before. Measured the detail levels (section 5): with the default reclaim step F2 shrinks with `ppm3` only because the slices get thinner; with a fixed 1 m step F2 and F1 change by at most 10 % between `ppm3` = 1 and 64, the noise falls tenfold, `ppm3` = 64 costs 91 ms per simulation (36 times `ppm3` = 1). |
