"""OnlineResult schema v1 as agreed with work package F1, plus a stand-in.

The real ``OnlineResult.save`` / ``load`` live in ASF (branch
``online/f1-interfaces``, not merged when this was written). Until then the
tests use :class:`StandInOnlineResult`, which reads and writes exactly the
agreed files, and :func:`stand_in_run_prequential`, which wraps today's
``run_prequential`` and pads its output to the v1 columns.

Nothing in ``src/`` imports this module. Once F1 is merged, the fixtures under
``tests/fixtures/results`` double as a compatibility check for the real
``OnlineResult.load`` (see ``tests/test_e2e.py``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1

# column -> kind, in file order
STEP_COLUMNS: dict[str, str] = {
    "step": "int64",
    "instance_id": "string",
    "primary": "string",
    "extras": "string",
    "runs": "string",
    "n_runs": "int64",
    "pred_runtime": "float64",
    "pred_score": "float64",
    "pred_uncertainty": "float64",
    "instance_score": "float64",
    "primary_runtime": "float64",
    "primary_censored": "bool",
    "cost_primary": "float64",
    "cost_best_of_scheduled": "float64",
    "user_cost": "float64",
    "compute_cost": "float64",
    "vbs_cost": "float64",
    "vbs_algorithm": "string",
    "sbs_cost": "float64",
    "sbs_cost_warmstart": "float64",
    "retrained": "bool",
    "drift_signal": "bool",
    "drift_warning": "bool",
    "detector_score": "float64",
    "drift_statistic": "float64",
    "response": "string",
    "selector_seconds": "float64",
    "detector_seconds": "float64",
}

META_KEYS = {
    "schema_version",
    "asf",
    "selector",
    "stream",
    "protocol",
    "n_warmstart",
    "seed",
    "result_mode",
    "par_factor",
    "maximize",
    "sbs_algorithm_stream",
    "sbs_algorithm_warmstart",
    "retrain_steps",
    "drift_steps",
    "retrain_seconds",
    "created_utc",
    "extra",
}


def null_non_finite(value: Any) -> Any:
    """JSON in schema v1 never holds NaN or Infinity: they become null."""
    if isinstance(value, dict):
        return {k: null_non_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [null_non_finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def conform_steps(steps: pd.DataFrame) -> pd.DataFrame:
    """Order and cast a frame to the v1 columns; fail on missing or extra ones."""
    if set(steps.columns) != set(STEP_COLUMNS):
        raise ValueError(
            f"steps columns differ from schema v1: "
            f"{sorted(set(steps.columns) ^ set(STEP_COLUMNS))}"
        )
    out = pd.DataFrame(index=range(len(steps)))
    for column, kind in STEP_COLUMNS.items():
        values = steps[column].to_numpy()
        if kind == "string":
            out[column] = pd.array([str(v) for v in values], dtype=object)
        else:
            out[column] = values.astype(kind)
    return out


def _closed_gap(cost: float, sbs: float, vbs: float) -> float:
    gap = sbs - vbs
    return float((sbs - cost) / gap) if gap > 0 else float("nan")


class StandInOnlineResult:
    """Schema-faithful stand-in for the F1 ``OnlineResult`` (tests only)."""

    def __init__(self, steps: pd.DataFrame, meta: dict[str, Any]) -> None:
        self.steps = steps
        self.meta = meta

    def save(self, path: str | Path, overwrite: bool = False) -> Path:
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = null_non_finite(self.meta)
        if set(meta) != META_KEYS:
            raise ValueError(f"meta keys differ: {sorted(set(meta) ^ META_KEYS)}")
        (path / "meta.json").write_text(
            json.dumps(meta, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        conform_steps(self.steps).to_parquet(
            path / "steps.parquet", engine="pyarrow", index=False
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> StandInOnlineResult:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        if meta["schema_version"] > SCHEMA_VERSION:
            raise ValueError(
                f"{path}: schema_version {meta['schema_version']} is newer than "
                f"{SCHEMA_VERSION}."
            )
        return cls(pd.read_parquet(path / "steps.parquet"), meta)

    def summary(self, detection_tolerance: int = 100) -> dict[str, float]:
        s = self.steps
        user = float(s["user_cost"].mean())
        sbs, vbs = float(s["sbs_cost"].mean()), float(s["vbs_cost"].mean())
        sbs_warm = float(s["sbs_cost_warmstart"].mean())
        return {
            "mean_user_cost": user,
            "mean_compute_cost": float(s["compute_cost"].mean()),
            "mean_runs_per_instance": float(s["n_runs"].mean()),
            "sbs_cost": sbs,
            "vbs_cost": vbs,
            "closed_gap": _closed_gap(user, sbs, vbs),
            "sbs_cost_warmstart": sbs_warm,
            "closed_gap_warmstart": _closed_gap(user, sbs_warm, vbs),
            "n_retrains": float(len(self.meta["retrain_steps"])),
            "retrain_seconds": float(self.meta["retrain_seconds"]),
            "n_drift_signals": float(len(self.meta["drift_steps"])),
        }


def make_meta(
    *,
    selector_class: str,
    selector_params: dict[str, Any],
    stream: dict[str, Any],
    n_warmstart: int,
    seed: int | None,
    result_mode: str,
    par_factor: float,
    sbs_algorithm_stream: str,
    sbs_algorithm_warmstart: str | None,
    retrain_steps: list[int],
    drift_steps: list[int],
    retrain_seconds: float,
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "asf": {
            "version": "0.1.8",
            "commit": "0" * 40,
            "dirty": False,
            "commit_source": "git",
        },
        "selector": {"class": selector_class, "params": selector_params},
        "stream": stream,
        "protocol": "warm" if n_warmstart else "cold",
        "n_warmstart": n_warmstart,
        "seed": seed,
        "result_mode": result_mode,
        "par_factor": par_factor,
        "maximize": False,
        "sbs_algorithm_stream": sbs_algorithm_stream,
        "sbs_algorithm_warmstart": sbs_algorithm_warmstart,
        "retrain_steps": retrain_steps,
        "drift_steps": drift_steps,
        "retrain_seconds": retrain_seconds,
        "created_utc": "2026-11-12T15:00:00Z",
        "extra": extra,
    }


def stand_in_run_prequential(
    selector: Any,
    stream: Any,
    n_warmstart: int = 0,
    par_factor: float = 10.0,
    result_mode: str | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    stream_seed: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> StandInOnlineResult:
    """Today's ``run_prequential`` behind the F1 signature, padded to schema v1.

    Columns the pre-F1 evaluator does not produce are filled with neutral
    values (NaN, False, empty); the costs and counts are the real ones.
    """
    from asf.online.evaluation import run_prequential

    result = run_prequential(
        selector,
        stream,
        n_warmstart=n_warmstart,
        par_factor=par_factor,
        result_mode=result_mode,
        fit_kwargs=fit_kwargs,
    )
    mode = result_mode or selector.RESULT_MODE
    old = result.steps.reset_index(drop=True)
    n = len(old)
    algorithms = [str(a) for a in stream.performance.columns]
    nan = np.full(n, np.nan)
    steps = pd.DataFrame(
        {
            "step": np.arange(n),
            "instance_id": old["instance_id"].astype(str),
            "primary": old["primary"].astype(str),
            "extras": "[]",
            "runs": "{}",
            "n_runs": old["n_runs"],
            "pred_runtime": nan,
            "pred_score": nan,
            "pred_uncertainty": nan,
            "instance_score": nan,
            "primary_runtime": nan,
            "primary_censored": False,
            "cost_primary": old["user_cost"] if mode == "primary" else nan,
            "cost_best_of_scheduled": old["user_cost"],
            "user_cost": old["user_cost"],
            "compute_cost": old["compute_cost"],
            "vbs_cost": old["vbs_cost"],
            "vbs_algorithm": algorithms[0],
            "sbs_cost": old["sbs_cost"],
            "sbs_cost_warmstart": old["sbs_cost"] if n_warmstart else nan,
            "retrained": old["retrained"],
            "drift_signal": [i in set(result.drift_steps) for i in range(n)],
            "drift_warning": False,
            "detector_score": nan,
            "drift_statistic": nan,
            "response": "",
            "selector_seconds": 0.0,
            "detector_seconds": 0.0,
        }
    )
    cutoff = float(stream.cutoff)
    meta = make_meta(
        selector_class=f"{type(selector).__module__}.{type(selector).__qualname__}",
        selector_params={},
        stream={
            "name": stream.name,
            "seed": stream_seed,
            "n_instances": len(stream),
            "n_evaluated": n,
            "cutoff": cutoff if math.isfinite(cutoff) else None,
            "algorithms": algorithms,
            "n_features": int(stream.features.shape[1]),
            "drift_points": [int(p) for p in result.drift_points],
            "transition_intervals": [list(i) for i in result.transition_intervals],
        },
        n_warmstart=n_warmstart,
        seed=selector.random_state,
        result_mode=mode,
        par_factor=par_factor,
        sbs_algorithm_stream=algorithms[0],
        sbs_algorithm_warmstart=algorithms[0] if n_warmstart else None,
        retrain_steps=[int(s) for s in result.retrain_steps],
        drift_steps=[int(s) for s in result.drift_steps],
        retrain_seconds=float(result.retrain_seconds),
        extra=metadata or {},
    )
    return StandInOnlineResult(steps, meta)
