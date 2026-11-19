# online-algorithm-selection

Experiments for the master thesis *Online Algorithm Selection for Instance
Streams under Data Drift*.

The methods (selectors, drift detectors, streams, the prequential evaluator)
live in [ASF](https://github.com/BrianSchiller/asf), package `asf.online`.
This repository holds everything specific to the thesis: experiment configs,
the runner, results, and later dataset preparation, analysis and plots. The
seam between the two is a saved `OnlineResult`: ASF writes and reads result
files, this repository never does so itself.

## Layout

```
configs/                    experiment configs (YAML)
src/oas_experiments/
  config.py                 YAML -> grid of single runs, run ids
  build.py                  config blocks -> selector objects
  streams.py                stream kinds -> asf InstanceStream
  provenance.py             commit + dirty flag of this repo and ASF, --final checks
  io.py                     result paths; the only caller of OnlineResult.save/load
  run.py                    python -m oas_experiments.run    (one run)
  list.py                   python -m oas_experiments.list   (run ids of a config)
  launch.py                 python -m oas_experiments.launch (all missing runs, in parallel)
  results.py                load an experiment into DataFrames
  datasets/                 placeholder: dataset preparation (bin packing, SATCOMP)
  analysis/                 metrics.py, stats.py (see Metrics); plots.py, tables.py, style.py
  figures.py                python -m oas_experiments.figures (all figures and tables)
scripts/                    SLURM job array (submit.sh, slurm_array.sh)
tests/                      tests, schema-v1 fixture results
results/                    <experiment>/<run id>/{meta.json, steps.parquet}  (not versioned)
```

## Install

Python 3.10 or newer. For development, with the ASF clone next to this
repository (`../asf`):

```
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"           # installs ASF at the pinned commit
pip install -e "../asf[online]"   # then replaces it with your local clone
```

The order matters: `pip install -e .` installs the pinned ASF and can replace
your local clone, so repeat the last line after it. Every result records which ASF commit was
imported and whether the checkout was dirty.

For final runs, skip the last line; see [Final runs](#final-runs).

## Workflow

1. **Describe the experiment** in one YAML file under `configs/`
   (format below).
2. **List its runs**:
   ```
   python -m oas_experiments.list configs/example_synthetic_sudden.yaml
   ```
3. **Run them**, one process per run. A single run:
   ```
   python -m oas_experiments.run configs/example_synthetic_sudden.yaml --run-id sudden-s0__cold__ds-ts__seed0
   ```
   or every run without a result, several at a time (on a laptop or
   workstation; see [On a cluster](#on-a-cluster) for SLURM):
   ```
   python -m oas_experiments.launch configs/example_synthetic_sudden.yaml --jobs 8
   python -m oas_experiments.launch configs/example_synthetic_sudden.yaml --jobs 8 --match "*__cold__*"
   ```
   The result goes to `results/<experiment>/<run id>/`. Use `--results-dir`
   or the environment variable `OAS_RESULTS_DIR` for another location.
   `launch` writes the output of each run to
   `results/_logs/<experiment>/<run id>.log`, limits numerical libraries to
   one thread per run, lists failed runs at the end and only starts runs
   without a result, so after a failure you simply launch again.
   `--dry-run` prints what would start.
4. **Load the results**:
   ```python
   from oas_experiments.results import load_experiment

   results = load_experiment("results/example_synthetic_sudden")
   results.runs  # one row per run
   results.steps  # one row per run and step
   ```

### Re-running

- A run whose result exists is skipped (exit code 0).
- If the existing result was produced with a different config for the same
  run id (you changed a parameter but kept the method name), the run fails
  with exit code 3 instead of silently keeping the old result. Rename the
  method, or pass `--overwrite`.
- A result directory appears only when it is complete: ASF saves into a
  temporary sibling directory (`<run id>.tmp-<pid>`), which is then renamed.
  Leftover `.tmp-` directories of killed jobs are ignored and can be deleted.

### On a cluster

Runs are independent, so an experiment is a SLURM job array: task `N` runs
the N-th line (0-based) of `oas_experiments.list` via `run --index N`.

```
bash scripts/submit.sh configs/my_experiment.yaml            # pilot
bash scripts/submit.sh configs/my_experiment.yaml --final    # thesis results
```

`submit.sh` activates `$OAS_VENV` (default `.venv`), counts the runs and
submits `scripts/slurm_array.sh` as an array with at most `MAX_PARALLEL`
(default 100) tasks at once; arguments after the config go to `run`. Adjust
the `#SBATCH` lines (time, memory, account) in `slurm_array.sh` to the
cluster. Task output goes to `logs/`.

To redo failed tasks, submit again: tasks whose result exists finish
immediately. `python -m oas_experiments.list <config> --missing` shows what
is still missing. The index always counts over all runs of the config, so it
stays valid while results arrive.

## Config format

A config is a grid: streams × stream seeds × protocols × methods × selector
seeds. See `configs/example_synthetic_sudden.yaml` for a complete file.

```yaml
name: my_experiment            # results/<name>/

evaluation:                    # optional; these are the defaults
  par_factor: 10.0
  result_mode: best_of_scheduled   # or: primary

seeds: [0, 1, 2]               # selector seeds

protocols:
  - {name: warm200, type: warm, n_warmstart: 200}
  - {name: cold, type: cold}

streams:
  - name: sudden
    kind: synthetic_scenario   # see "Stream kinds"
    seeds: [0]                 # stream seeds (default [0])
    params: {...}              # passed to the stream kind

methods:
  - name: ds-ts
    protocols: [warm200]       # optional: restrict to some protocols
    selector:
      class: asf.online.selectors.DiscountedTSSelector
      params: {gamma: 0.99}
```

**Run ids** have the form
`<stream>-s<stream seed>__<protocol>__<method>__seed<selector seed>`.
They are built from names only, so adding methods, streams or seeds never
changes the ids of existing runs. Names consist of letters and digits with
single `.`, `_` or `-` in between.

**Protocols.** `warm` fits the selector on the first `n_warmstart` instances
of the stream (all algorithms observed) and evaluates on the rest; `cold`
evaluates on the whole stream. Step numbers and drift points in a result
count from the first evaluated instance, so the same stream has its drift at
a different step under each protocol. Offline baselines only make sense warm:
give them `protocols: [<warm protocol>]`.

**Selector blocks.** Inside `params`, three forms are built into objects:

| YAML | becomes |
| --- | --- |
| `{class: pkg.mod.Name, params: {...}}` | an instance, `Name(**params)`, built recursively |
| `{type: pkg.mod.Name}` | the class itself (for arguments like `model_class`) |
| `{type: pkg.mod.Name, params: {...}}` | the class with preset arguments, `functools.partial(Name, **params)`, e.g. a `model_class` with `n_estimators: 50` |
| `"${seed}"`, `"${stream.cutoff}"` | the run's selector seed, the stream's cutoff |

The top-level selector always receives `random_state=<selector seed>`; do not
set it yourself. Nested components that need a seed take `"${seed}"`.

**Hyperparameters** are plain config values. To compare settings, either list
them as separately named methods (`ds-ts-g95`, `ds-ts-g99`; YAML anchors and
`<<:` merge keys avoid repetition) or give the method a `sweep`:

```yaml
  - name: frugal-unc
    sweep:
      gate.params.rate: [0.1, 0.25, 0.5]   # dotted path inside selector.params
    selector:
      class: asf.online.selectors.FrugalPairwiseSelector
      params:
        gate:
          class: asf.online.gate.UncertaintyGate
          params: {rate: 0.25, random_state: "${seed}"}   # the value is replaced
```

This expands into the methods `frugal-unc-rate0.1`, `frugal-unc-rate0.25` and
`frugal-unc-rate0.5` (name, last part of the path, value). Several paths give
every combination (`-rate0.1-n2`, ...). The enclosing block of a path must be
in the selector; the last key may be new. Swept runs store `family` and
`sweep` in `meta.extra.run.method`, and `load_experiment` reports the family
as `method_family` (the method name for runs without a sweep), so the points
of a cost-versus-quality curve can be grouped.

### Stream kinds

| kind | builds | seed goes to |
| --- | --- | --- |
| `synthetic_scenario` | `asf.online.streams.make_synthetic_scenario(**params)` | `random_state` |
| `synthetic` | `asf.online.streams.make_synthetic_stream` from `regimes`, `phases`, `performance_model`, `cutoff` and further keyword arguments | `random_state` |
| `aslib` | one or more ASlib scenarios (`root`, `scenarios`), ordered by the `drift` block; without one, in random order | sampling and shuffling |

The synthetic kinds reject a `drift` block. For `aslib` it chooses how the
instances are ordered (see `configs/example_sat12_switching.yaml` and ASF's
`docs/features/stream_construction.md`):

```yaml
- name: sat12-halves
  kind: aslib
  params:
    root: ../../aslib_data          # relative to the working directory
    scenarios: [SAT12-HAND, SAT12-INDU, SAT12-RAND]   # stacked; must share features, solvers, cutoff
  drift:
    builder: grouped                # play groups one after another
    params:
      labels: scenario              # or best_algorithm | {clusters: n} | {feature: <expr>, quantiles: n}
      schedule: [SAT12-HAND, SAT12-INDU, SAT12-RAND, SAT12-HAND, SAT12-INDU, SAT12-RAND]
      # segment_length: 300         # default: all instances, each group split evenly over its entries
      # transition_length: 100      # default 0 = sudden; a list entry [A, B] mixes groups
```

`builder: sorted` orders by `key: {feature: <column or expression>}` (e.g.
`nclauses / nvars`) or `key: {runtime: [algorithms] | all, agg: max}`, with
optional `ascending` and `jitter`. Clusters are always computed with seed 0,
so a cluster id means the same cluster for every stream seed.

## What a result contains

ASF defines the files (OnlineResult schema v1: `meta.json` and
`steps.parquet`). The runner adds the following under `meta.extra`:

| key | content |
| --- | --- |
| `experiment` | experiment name, run id, config file name, hash of the parsed config |
| `run` | the resolved config of this run: method with selector block, stream, protocol, seed, evaluation settings. `${seed}` is substituted; `${stream.cutoff}` stays symbolic (its value is `meta.stream.cutoff`) |
| `repo` | commit, dirty flag and source of this repository |
| `asf_pinned_commit` | the ASF commit pinned in `pyproject.toml` |
| `final` | whether the run was started with `--final` |
| `environment` | Python version, platform, versions of numpy, pandas, scikit-learn, scipy, pyarrow |

The state of ASF itself is recorded by ASF as `meta.asf`.

### Loading

`load_experiment(dir)` reads every run with `OnlineResult.load`, rejects any
`schema_version` it does not know, and returns:

- `runs`: one row per run. Config columns (`method`, `selector_class`,
  `stream`, `stream_seed`, `protocol`, `protocol_type`, `n_warmstart`, `seed`,
  ...), everything `OnlineResult.summary()` reports (`closed_gap`,
  `mean_user_cost`, `mean_compute_cost`, ...), totals over the stream
  (`total_compute_cost`, `total_user_cost`, `total_selector_seconds`,
  `total_detector_seconds`), provenance (`final`, `repo_commit`, `asf_commit`,
  ...), and the selector parameters as `param.*` columns
  (`param.gamma`, `param.detector.chunk_size`).
- `steps`: the per-step frames of all runs stacked, with `run_id` as the
  first column. Join with `runs` on `run_id`.

`collect_runs(dir)` and `collect_steps(dir, columns=[...])` return one of the
two. Incomplete run directories are skipped with a warning.

Metrics and statistics: see [Metrics](#metrics).

## Metrics

`oas_experiments.analysis.metrics` and `.stats` compute every number the
thesis reports from the frames of `load_experiment`. Defaults: `window=50`
steps, `tolerance=0.1`, `detection_tolerance=100`, `confidence=0.95`. All are
function parameters. Steps and drift points count from the first evaluated
instance, so never compare curves across protocols or streams.

```python
from oas_experiments.analysis import metrics, stats

table = metrics.run_table(results.runs, results.steps)  # one row per run
summary = metrics.aggregate(table)  # over seeds
steps = metrics.add_step_metrics(results.steps, results.runs)
curves = metrics.aggregate_curves(metrics.rolling_curves(steps, "regret"), results.runs)
```

`load_experiment` also gives per run `maximize` and, as lists in step
indexing, `drift_points`, `transition_intervals` (half-open `[start, end)`),
`drift_steps` and `retrain_steps`.

**Per step** (`add_step_metrics`, within each run):

| metric | definition |
| --- | --- |
| `regret` | `user_cost - vbs_cost`; `vbs_cost - user_cost` when maximising |
| `cum_regret` | cumulative sum of `regret` |
| `extra_run` | 1.0 if more than one algorithm ran (`n_runs > 1`), else 0.0; its mean is `extra_run_rate` |
| `extra_cpu` | `compute_cost - primary_runtime` (CPU of extra runs; NaN when maximising) |
| `overhead_seconds` | `selector_seconds + detector_seconds` |
| `rolling(steps, column, window)` | trailing mean over `window` steps; NaN for the first `window - 1` steps |

**Per drift point** (`drift_metrics`). For a drift point `p`: `end` is the
next drift point or the end of the stream; `start` is `p`, or for gradual
drift the end of the transition interval that begins at `p`.

| metric | definition |
| --- | --- |
| `pre_level` | mean regret over `[p - window, p)` (shorter at the stream start; NaN at step 0) |
| `peak_regret` | largest mean regret of a full `window` inside `[p, end)` |
| `recovery` | `s - start` for the first `s >= start` such that the window `[s, s + window)` lies before `end` and its mean regret is at most `(1 + tolerance) * pre_level` (the trailing rolling regret at `s + window - 1`); 0 = no visible degradation, NaN = never recovered or no `pre_level` |
| `extra_run_rate_before` / `_after` | share of steps with `n_runs > 1` in `[p - window, p)` / `[p, min(p + window, end))` |

Transition intervals that start inside the warm-start prefix have no drift
point in the evaluated stream and are ignored.

**Per run** (`run_table`): the config columns, `uses_detector` (the
selector's `detector` parameter is set), then `closed_gap`,
`mean_user_cost` (from `OnlineResult.summary()`), `solver_cpu` (total
`compute_cost`), `extra_cpu`, `overhead_seconds` (totals), `extra_run_rate`
(share of steps with more than one run), `n_drift_signals`; `mean_delay`,
`n_missed`, `n_false_alarms` (ASF's `detection_metrics` with
`detection_tolerance`: a signal in `[p, p + tolerance]` detects `p`; NaN
without a detector, so a detector that never fires shows all drifts missed);
`mean_recovery` (over the drift points that recovered), `n_not_recovered`,
`mean_peak_regret` (over all drift points); then the `param.*` columns. Drift
metrics are NaN for streams without drift points; CPU metrics are NaN when
maximising.

**Over seeds** (`aggregate`): grouped by `(stream, protocol, method)` (plus
`method_family` and `uses_detector`), pooling stream and selector seeds; per metric `<m>_mean`,
`<m>_std` (ddof 1) and `<m>_ci95` (half-width of the t-interval, NaN for
fewer than two values), and `n` runs. `aggregate_curves` gives
`[stream, protocol, method, method_family, uses_detector, step, mean, lo, hi]`
with
`lo`/`hi` = mean ∓ CI half-width per step; it refuses groups whose runs have
different drift points.

**Method comparison** (`stats`): a method's value in a (stream, protocol)
cell is its mean over seeds; only cells where every method has a value are
used, so filter the run table to the methods to compare first (offline
baselines exist only warm). `mean_ranks` (1 = best, ties averaged),
`friedman` (needs three or more methods), `wilcoxon_vs_reference`
(two-sided signed-rank test per method against a reference, Holm-corrected
`p_holm`; `median_diff` = method − reference). With 6 cells the smallest
exact Wilcoxon p-value is 0.031.

## Figures and tables

Needs matplotlib: `pip install -e ".[analysis]"` (included in `dev`).

```
python -m oas_experiments.figures results/pilot_sat12 --out figures/pilot_sat12
python -m oas_experiments.figures results/pilot_sat12 --out figures/pilot_sat12 --only front
```

Writes, for every stream and protocol in the results:

| file | content |
| --- | --- |
| `regret_<stream>_<protocol>.pdf` | rolling regret per method (mean, CI band), drift points dashed, transition intervals shaded |
| `extra_runs_<stream>_<protocol>.pdf` | rolling extra-run rate, same layout |
| `raster_<stream>_<protocol>.pdf` | detector signals per method and seed against the true drift (only if a method has a detector) |
| `front.pdf` | closed gap against solver CPU, one panel per stream and protocol, one line per method family (its sweep), ci95 error bars |
| `front_overhead.pdf` | the same with selector and detector time added to x |
| `table_main_<protocol>.tex` | closed gap, solver CPU, extra-run rate, recovery, drift points not recovered from: mean ± ci95 per method and stream, best in bold |
| `table_detection_<protocol>.tex` | delay, missed drifts, false alarms of methods with a drift detector |

`--only` takes the part before the stream name (`regret`, `extra_runs`,
`raster`, `front`, `front_overhead`, `table_main`, `table_detection`).
Tables are `tabular` environments (booktabs); `\input` them inside a `table`
with your own caption. Figures are 6.3 in wide (the thesis text width);
change it with `--width 4.5`, or `width=` when calling the functions in
`oas_experiments.analysis.plots` directly (call `style.apply()` first for the
thesis fonts). Colours are fixed per method family, assigned by sorted family
name, so they stay the same across the figures of one experiment.

## Final runs

Results that go into the thesis must be traceable to committed code:

```
pip install -e .        # this repo from a clean clone, ASF at the pinned commit
python -m oas_experiments.run <config> --run-id <id> --final
```

`--final` refuses to start (exit code 4) unless

- this repository is a git checkout without uncommitted or untracked files,
- the imported ASF has a known commit and no uncommitted changes, and
- that commit is the one pinned in `pyproject.toml`.

To move the pin, push the ASF commit, change the hash in the
`asf[online] @ git+...@<hash>` line of `pyproject.toml` and commit. The pin
points at `online-selection` with all Phase 1 packages merged (`ae9fcb5`).

## Tests

```
pytest
ruff check . && ruff format --check .
```

- `tests/fixtures/results/` holds small result directories that follow schema
  v1. Their numbers are synthetic (so tests can rely on fixed values) and are
  written with ASF's `OnlineResult.save` by
  `python tests/fixtures/make_fixtures.py`.
- `tests/test_e2e.py` runs all 15 runs of the example config with the
  installed ASF and loads them back.
