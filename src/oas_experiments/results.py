"""Collect the results of an experiment into DataFrames.

    results = load_experiment("results/example_synthetic_sudden")
    results.runs    # one row per run: config columns, summary metrics, totals
    results.steps   # one row per run and step: the per-step frame plus run_id

Every result is read with ``OnlineResult.load`` and its ``schema_version`` is
checked. Config columns come from the resolved config that the runner stored
in the result, so no config file is needed to load an experiment.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from oas_experiments import io

SUPPORTED_SCHEMA_VERSIONS = (1,)

_TOTALS = {
    "total_user_cost": "user_cost",
    "total_compute_cost": "compute_cost",
    "total_selector_seconds": "selector_seconds",
    "total_detector_seconds": "detector_seconds",
}


class ResultError(RuntimeError):
    """A result directory cannot be used."""


class SchemaVersionError(ResultError):
    """A result was written with a schema version this code does not know."""


@dataclass(frozen=True)
class ExperimentResults:
    """All results of one experiment."""

    runs: pd.DataFrame
    steps: pd.DataFrame


def run_directories(experiment_dir: str | Path) -> list[Path]:
    """Complete run directories of an experiment, sorted by run id.

    Temporary directories of running or killed jobs are ignored; directories
    that look like a run but are incomplete are reported with a warning.
    """
    experiment_dir = Path(experiment_dir)
    if not experiment_dir.is_dir():
        raise ResultError(f"{experiment_dir} is not a directory.")
    complete, incomplete = [], []
    for path in sorted(p for p in experiment_dir.iterdir() if p.is_dir()):
        if io.is_temporary(path):
            continue
        (complete if io.is_complete(path) else incomplete).append(path)
    if incomplete:
        warnings.warn(
            f"{experiment_dir}: skipping {len(incomplete)} incomplete run "
            f"directories: {[p.name for p in incomplete]}",
            stacklevel=2,
        )
    if not complete:
        raise ResultError(f"{experiment_dir} contains no results.")
    return complete


def flatten_params(params: dict[str, Any], prefix: str = "param") -> dict[str, Any]:
    """Selector params as flat columns: ``param.gamma``, ``param.detector.class``.

    ``{class, params}`` blocks contribute their class and their own params,
    ``{type: X}`` becomes ``X``; lists are stored as JSON text.
    """
    flat: dict[str, Any] = {}
    for key, value in params.items():
        name = f"{prefix}.{key}"
        if isinstance(value, dict) and set(value) == {"type"}:
            flat[name] = value["type"]
        elif isinstance(value, dict) and "class" in value:
            flat[f"{name}.class"] = value["class"]
            flat.update(flatten_params(value.get("params") or {}, name))
        elif isinstance(value, dict):
            flat.update(flatten_params(value, name))
        elif isinstance(value, list):
            flat[name] = json.dumps(value, sort_keys=True)
        else:
            flat[name] = value
    return flat


def _checked_meta(result: Any, path: Path) -> dict[str, Any]:
    meta = io.result_meta(result)
    version = meta.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise SchemaVersionError(
            f"{path}: schema_version {version!r} is not supported "
            f"(supported: {list(SUPPORTED_SCHEMA_VERSIONS)})."
        )
    run = (meta.get("extra") or {}).get("run")
    if not isinstance(run, dict):
        raise ResultError(
            f"{path}: no resolved config in meta.extra.run; the result was not "
            "produced by oas_experiments.run."
        )
    stored = (run["run_id"], run["seed"], run["protocol"]["n_warmstart"])
    actual = (path.name, meta["seed"], meta["n_warmstart"])
    if stored != actual:
        raise ResultError(
            f"{path}: stored config (run id, seed, n_warmstart) = {stored} does "
            f"not match the result itself {actual}."
        )
    return meta


def _run_row(result: Any, meta: dict[str, Any]) -> dict[str, Any]:
    extra = meta["extra"]
    run = extra["run"]
    repo = extra.get("repo") or {}
    cutoff = meta["stream"]["cutoff"]
    row: dict[str, Any] = {
        "run_id": run["run_id"],
        "experiment": run["experiment"],
        "method": run["method"]["name"],
        "method_family": run["method"].get("family", run["method"]["name"]),
        "selector_class": run["method"]["selector"]["class"],
        "stream": run["stream"]["name"],
        "stream_kind": run["stream"]["kind"],
        "stream_seed": run["stream"]["seed"],
        "protocol": run["protocol"]["name"],
        "protocol_type": run["protocol"]["type"],
        "n_warmstart": run["protocol"]["n_warmstart"],
        "seed": run["seed"],
        "result_mode": meta["result_mode"],
        "par_factor": meta["par_factor"],
        "cutoff": math.inf if cutoff is None else cutoff,
        "n_evaluated": meta["stream"]["n_evaluated"],
        "maximize": bool(meta.get("maximize", False)),
        # Lists in step indexing (they differ between warm and cold protocols).
        "drift_points": list(meta["stream"]["drift_points"]),
        "transition_intervals": [
            list(i) for i in meta["stream"]["transition_intervals"]
        ],
        "drift_steps": list(meta["drift_steps"]),
        "retrain_steps": list(meta["retrain_steps"]),
    }
    summary = dict(result.summary())
    clash = sorted(set(summary) & set(row))
    if clash:
        raise ResultError(f"summary() keys clash with config columns: {clash}")
    row.update(summary)
    for column, source in _TOTALS.items():
        row[column] = float(result.steps[source].sum(skipna=False))
    row.update(
        {
            "final": extra.get("final"),
            "repo_commit": repo.get("commit"),
            "repo_dirty": repo.get("dirty"),
            "asf_commit": meta["asf"]["commit"],
            "asf_dirty": meta["asf"]["dirty"],
            "created_utc": meta["created_utc"],
        }
    )
    row.update(flatten_params(run["method"]["selector"].get("params") or {}))
    return row


def load_experiment(
    experiment_dir: str | Path,
    *,
    steps: bool = True,
    step_columns: Iterable[str] | None = None,
) -> ExperimentResults:
    """Load every result under ``<results dir>/<experiment>``.

    Parameters
    ----------
    experiment_dir
        Directory with one sub-directory per run.
    steps
        Set to ``False`` to skip building the long per-step frame.
    step_columns
        Keep only these per-step columns (``run_id`` is always included).
    """
    keep = None if step_columns is None else list(step_columns)
    rows: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for path in run_directories(experiment_dir):
        result = io.load_result(path)
        meta = _checked_meta(result, path)
        rows.append(_run_row(result, meta))
        if steps:
            frame = result.steps if keep is None else result.steps[keep]
            frame = frame.copy()
            frame.insert(0, "run_id", path.name)
            frames.append(frame)
    return ExperimentResults(
        runs=pd.DataFrame(rows),
        steps=pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
    )


def collect_runs(experiment_dir: str | Path) -> pd.DataFrame:
    """One row per run: config columns, summary metrics, totals, ``param.*``."""
    return load_experiment(experiment_dir, steps=False).runs


def collect_steps(
    experiment_dir: str | Path, columns: Iterable[str] | None = None
) -> pd.DataFrame:
    """One row per run and step; join with :func:`collect_runs` on ``run_id``."""
    return load_experiment(experiment_dir, step_columns=columns).steps
