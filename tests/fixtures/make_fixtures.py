"""Generate the fixture results under ``tests/fixtures/results``.

    python tests/fixtures/make_fixtures.py

These directories follow OnlineResult schema v1 exactly and stand in for real
results until ASF can write them (work package F1). They are synthetic: the
numbers are drawn from a seeded generator, not produced by a selector, but
they are internally consistent (costs follow from the runs, the VBS and SBS
columns from the hidden runtimes). The script is deterministic.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1] / "src")]

from oas_experiments.config import RunSpec, load_config  # noqa: E402
from support.schema_v1 import StandInOnlineResult, make_meta  # noqa: E402

ALGORITHMS = ["a0", "a1", "a2"]
CUTOFF, PAR_FACTOR, N_INSTANCES, DRIFT_POINT = 100.0, 10.0, 40, 25
REPO = {"commit": "1" * 40, "dirty": False, "commit_source": "git"}


def _par(runtime: np.ndarray) -> np.ndarray:
    return np.where(runtime <= CUTOFF, runtime, PAR_FACTOR * CUTOFF)


def _truth() -> pd.DataFrame:
    """Hidden runtimes: a0 is best before the drift, a2 after it."""
    rng = np.random.default_rng(0)
    scale = np.where(
        np.arange(N_INSTANCES)[:, None] < DRIFT_POINT, [5, 20, 60], [80, 20, 5]
    )
    runtimes = scale * rng.lognormal(0.0, 0.8, size=(N_INSTANCES, 3))
    index = [f"inst_{i}" for i in range(N_INSTANCES)]
    return pd.DataFrame(runtimes, index=index, columns=ALGORITHMS)


def make_result(
    run: RunSpec, truth: pd.DataFrame, config_hash: str
) -> StandInOnlineResult:
    seed = sum(map(ord, run.run_id))  # stable across processes, unlike hash()
    rng = np.random.default_rng(seed)
    warm, tail = truth.iloc[: run.n_warmstart], truth.iloc[run.n_warmstart :]
    par = pd.DataFrame(_par(tail.to_numpy()), index=tail.index, columns=ALGORITHMS)
    sbs_stream = par.mean().idxmin()
    sbs_warm = (
        pd.DataFrame(_par(warm.to_numpy()), columns=ALGORITHMS).mean().idxmin()
        if len(warm)
        else None
    )
    two_runs = run.method == "rook"
    drift_step = DRIFT_POINT - run.n_warmstart + 4 if two_runs else None

    rows = []
    for step, (instance, runtimes) in enumerate(tail.iterrows()):
        primary = str(rng.choice(ALGORITHMS))
        extras = (
            [str(rng.choice([a for a in ALGORITHMS if a != primary]))]
            if two_runs
            else []
        )
        runs = {
            a: {
                "runtime": float(min(runtimes[a], CUTOFF)),
                "censored": bool(runtimes[a] > CUTOFF),
            }
            for a in [primary, *extras]
        }
        cost = {
            a: PAR_FACTOR * CUTOFF if r["censored"] else r["runtime"]
            for a, r in runs.items()
        }
        best = min(cost.values())
        predicted = (
            float(runtimes[primary] * rng.lognormal(0.0, 0.2))
            if run.method == "ds-ts"
            else np.nan
        )
        rows.append(
            {
                "step": step,
                "instance_id": instance,
                "primary": primary,
                "extras": json.dumps(extras),
                "runs": json.dumps(runs),
                "n_runs": len(runs),
                "pred_runtime": predicted,
                "pred_score": float(rng.normal()),
                "pred_uncertainty": float(rng.uniform()),
                "instance_score": np.nan,
                "primary_runtime": runs[primary]["runtime"],
                "primary_censored": runs[primary]["censored"],
                "cost_primary": cost[primary],
                "cost_best_of_scheduled": best,
                "user_cost": best
                if run.result_mode == "best_of_scheduled"
                else cost[primary],
                "compute_cost": sum(r["runtime"] for r in runs.values()),
                "vbs_cost": float(par.loc[instance].min()),
                "vbs_algorithm": str(par.loc[instance].idxmin()),
                "sbs_cost": float(par.loc[instance, sbs_stream]),
                "sbs_cost_warmstart": float(par.loc[instance, sbs_warm])
                if sbs_warm
                else np.nan,
                "retrained": step == drift_step,
                "drift_signal": step == drift_step,
                "drift_warning": drift_step is not None
                and drift_step - 2 <= step < drift_step,
                "detector_score": float(rng.uniform()) if two_runs else np.nan,
                "drift_statistic": np.nan,
                "response": "retrain" if step == drift_step else "",
                "selector_seconds": float(rng.uniform(1e-4, 1e-3)),
                "detector_seconds": float(rng.uniform(1e-5, 1e-4)) if two_runs else 0.0,
            }
        )
    meta = make_meta(
        selector_class=run.selector["class"],
        selector_params={"random_state": run.seed},
        stream={
            "name": run.stream,
            "seed": run.stream_seed,
            "n_instances": N_INSTANCES,
            "n_evaluated": len(tail),
            "cutoff": CUTOFF,
            "algorithms": ALGORITHMS,
            "n_features": 3,
            "drift_points": [DRIFT_POINT - run.n_warmstart],
            "transition_intervals": [],
        },
        n_warmstart=run.n_warmstart,
        seed=run.seed,
        result_mode=run.result_mode,
        par_factor=run.par_factor,
        sbs_algorithm_stream=str(sbs_stream),
        sbs_algorithm_warmstart=sbs_warm,
        retrain_steps=[drift_step] if two_runs else [],
        drift_steps=[drift_step] if two_runs else [],
        retrain_seconds=0.25 if two_runs else 0.0,
        extra={
            "experiment": {
                "name": run.experiment,
                "run_id": run.run_id,
                "config_file": "fixture_exp.yaml",
                "config_hash": config_hash,
            },
            "run": run.to_dict(),
            "repo": REPO,
            "asf_pinned_commit": "0" * 40,
            "final": False,
            "environment": {"python": "3.10.12", "platform": "fixture", "packages": {}},
        },
    )
    return StandInOnlineResult(pd.DataFrame(rows), meta)


def main() -> None:
    config = load_config(HERE / "fixture_exp.yaml")
    out = HERE / "results" / config.name
    if out.exists():
        shutil.rmtree(out)
    truth = _truth()
    for run in config.runs:
        make_result(run, truth, config.config_hash).save(out / run.run_id)
    print(f"wrote {len(config.runs)} fixture results to {out}")


if __name__ == "__main__":
    main()
