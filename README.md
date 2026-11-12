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
  run.py                    python -m oas_experiments.run   (one run)
  list.py                   python -m oas_experiments.list  (run ids of a config)
  results.py                load an experiment into DataFrames
  datasets/                 placeholder: dataset preparation (bin packing, SATCOMP)
  analysis/                 placeholder: metrics, statistics, plots
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

The order matters: `pip install -e .` always (re)installs the pinned ASF, so
run the last line again after it. Every result records which ASF commit was
imported and whether the checkout was dirty.

For final runs, skip the last line; see [Final runs](#final-runs).

## Workflow

1. **Describe the experiment** in one YAML file under `configs/`
   (format below).
2. **List its runs**:
   ```
   python -m oas_experiments.list configs/example_synthetic_sudden.yaml
   ```
3. **Run them**, one process per run:
   ```
   python -m oas_experiments.run configs/example_synthetic_sudden.yaml --run-id sudden-s0__cold__ds-ts__seed0
   ```
   The result goes to `results/<experiment>/<run id>/`. Use `--results-dir`
   or the environment variable `OAS_RESULTS_DIR` for another location.
4. **Load the results**:
   ```python
   from oas_experiments.results import load_experiment

   results = load_experiment("results/example_synthetic_sudden")
   results.runs    # one row per run
   results.steps   # one row per run and step
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

Runs are independent, so an experiment is a job array. `--index N` runs the
N-th line (0-based) of `oas_experiments.list`:

```
N=$(python -m oas_experiments.list configs/my_experiment.yaml --count)
sbatch --array=0-$((N-1)) job.sh

# job.sh
python -m oas_experiments.run configs/my_experiment.yaml --index "$SLURM_ARRAY_TASK_ID" --final
```

`python -m oas_experiments.list <config> --missing` prints the run ids that
have no result yet, to resubmit failed jobs with `--run-id`. The index always
counts over all runs of the config, so it stays valid while results arrive.

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
| `"${seed}"`, `"${stream.cutoff}"` | the run's selector seed, the stream's cutoff |

The top-level selector always receives `random_state=<selector seed>`; do not
set it yourself. Nested components that need a seed take `"${seed}"`.

**Hyperparameters** are plain config values. To compare settings, list them as
separately named methods (`ds-ts-g95`, `ds-ts-g99`); YAML anchors avoid
repetition. A per-method sweep syntax is deliberately not part of this
skeleton.

### Stream kinds

| kind | builds | seed goes to |
| --- | --- | --- |
| `synthetic_scenario` | `asf.online.streams.make_synthetic_scenario(**params)` | `random_state` |
| `synthetic` | `asf.online.streams.make_synthetic_stream` from `regimes`, `phases`, `performance_model`, `cutoff` and further keyword arguments | `random_state` |

Dataset-backed kinds (an ASlib scenario plus a drift construction) are added
in `streams.py` with `@stream_kind("name")`. The `drift: {builder, params}`
block of a stream entry is reserved for them and rejected by the synthetic
kinds.

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

Plots and statistics belong in `src/oas_experiments/analysis/`.

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

To move the pin, change the hash in the `asf[online] @ git+...@<hash>` line of
`pyproject.toml` and commit. The pin currently points at the head of
`online-selection` from before the F1 interfaces; **move it to the commit that
contains them before the first final run.**

## Tests

```
pytest
ruff check . && ruff format --check .
```

- `tests/fixtures/results/` holds small result directories that follow schema
  v1 exactly. They are synthetic and generated by
  `python tests/fixtures/make_fixtures.py`.
- `tests/support/schema_v1.py` contains the schema as agreed with the F1 work
  package and a stand-in for `OnlineResult` and `run_prequential`. Tests use
  it so that they pass with an ASF that cannot save results yet.
- `tests/test_e2e.py` runs all 15 runs of the example config and loads them
  back, once with the stand-in and once with the real ASF. The real variant
  and a check that ASF loads the fixtures are skipped while ASF lacks
  `OnlineResult.save`. Once the F1 interfaces are merged and pinned, the
  stand-in and its tests can be removed.
