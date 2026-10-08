# Simulator comparison

Compares the stockpile simulation methods (`METHODS` in `scenarios.py`: mathematical, smooth, fast, HCP lattice, detailed) on the same
60 m x 20 m bed, using a cone and chevrons of 1 to 500 layers (`PATTERNS`) and four particle sizes (`RESOLUTIONS`). `viewer/` holds a
three.js page with a 3D view of each run over the stacking time, comparison charts and a metrics table.

## Generating the results

From the repository root:

```shell
uv run python -m bmh_apps.simulator_comparison.run_comparison all --out workdir/simulator-comparison --jobs $(nproc)
```

`all` runs every combination, each in its own process. It writes:

* `<out>/runs/<pattern>/input.json`: the deposition (time, stacker position, volume and quality)
* `<out>/runs/<pattern>/<resolution>/<method>.json`: reclaimed curves, metrics and snapshots for the time slider
* `<out>/runs/<pattern>/<resolution>/<method>.particles.txt`: particles of `fast` and `hcp`, as base64 text
* `<out>/index.json`: the list of runs and the state of each one (done, failed, timeout)

The last full run (102 runs, 76 MB) took about 12 minutes of total run time. Almost all of it went to the two detailed (Bullet) runs:
4 minutes for `chevron-1/mid` and 25 s for `cone/mid`. The detailed method only runs for the combinations in `DETAILED_RUNS`, because it
can take minutes to hours per run. Runs that exceed `--timeout` (default 3 h) are killed.

Useful variations:

```shell
# Without the detailed simulation
uv run python -m bmh_apps.simulator_comparison.run_comparison all --out workdir/simulator-comparison --methods mathematical smooth fast hcp
# A single run, e.g. to debug a method
uv run python -m bmh_apps.simulator_comparison.run_comparison one --out workdir/simulator-comparison --pattern cone --resolution mid --method hcp
# Rebuild index.json from the run files that exist, e.g. after single runs or an interrupted "all"
uv run python -m bmh_apps.simulator_comparison.run_comparison index --out workdir/simulator-comparison
```

`all` keeps the states of earlier runs in `index.json`, so you can run it once per subset of methods into the same directory.

## Viewing the results

The viewer loads `index.json`, `notes.json` and `runs/...` from paths relative to `index.html`. So copy the viewer into the output
directory and serve it over HTTP. Opened as a `file://` URL, the browser blocks the requests.

```shell
cp apps/bmh_apps/src/bmh_apps/simulator_comparison/viewer/{index.html,notes.json} workdir/simulator-comparison/
python -m http.server 8000 --directory workdir/simulator-comparison
```

Then open http://localhost:8000/. Copy the viewer again after changing it here.

`notes.json` holds the findings shown below the charts. It is written by hand from the results, not generated, so check it against new
results after changing a simulator or a scenario.

The particles are stored as base64 text, not as binary files, so that the directory can also be published
on web hosts that serve no binary files. For that, publish `index.html` as the page and everything else as supporting files at the same
relative paths.
