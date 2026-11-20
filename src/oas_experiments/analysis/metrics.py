"""Metrics reported in the thesis, computed from ``load_experiment`` frames.

    results = load_experiment("results/pilot_sat12")
    table = run_table(results.runs, results.steps)       # one row per run
    summary = aggregate(table)                            # mean, std, CI over seeds
    steps = add_step_metrics(results.steps, results.runs)
    curves = aggregate_curves(rolling_curves(steps, "regret"), results.runs)

Definitions and defaults are listed in the README, section "Metrics". Step
and drift-point indices count from the first evaluated instance, so never
compare curves across protocols or streams.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

WINDOW = 50
TOLERANCE = 0.1
DETECTION_TOLERANCE = 100
CONFIDENCE = 0.95
GROUP = ("stream", "protocol", "method")

CONFIG_COLUMNS = [
    "run_id",
    "experiment",
    "method",
    "method_family",
    "selector_class",
    "stream",
    "stream_kind",
    "stream_seed",
    "protocol",
    "protocol_type",
    "n_warmstart",
    "seed",
    "result_mode",
    "par_factor",
    "cutoff",
    "n_evaluated",
    "maximize",
]
METRICS = [
    "closed_gap",
    "mean_user_cost",
    "solver_cpu",
    "extra_cpu",
    "overhead_seconds",
    "extra_run_rate",
    "n_drift_signals",
    "mean_delay",
    "n_missed",
    "n_false_alarms",
    "mean_recovery",
    "mean_peak_regret",
    "n_not_recovered",
]
DRIFT_COLUMNS = [
    "run_id",
    "drift_index",
    "drift_point",
    "start",
    "end",
    "pre_level",
    "peak_regret",
    "recovery",
    "extra_run_rate_before",
    "extra_run_rate_after",
]


def add_step_metrics(steps: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Add ``regret``, ``cum_regret``, ``extra_run``, ``extra_cpu`` and
    ``overhead_seconds``.

    Parameters
    ----------
    steps
        Per-step frame of ``load_experiment``, steps in order within each run.
    runs
        Run frame of ``load_experiment``; only ``run_id`` and ``maximize`` are
        used.

    Returns
    -------
    pd.DataFrame
        A copy of ``steps`` with the five columns added. ``regret`` is
        ``user_cost - vbs_cost`` (``vbs_cost - user_cost`` when maximising), so
        it is never negative. ``extra_run`` is 1.0 if more than one algorithm
        ran (``n_runs > 1``), else 0.0; its mean is ``extra_run_rate``.
        ``extra_cpu`` is ``compute_cost -
        primary_runtime`` (NaN when maximising), ``overhead_seconds`` is
        ``selector_seconds + detector_seconds``.
    """
    maximize = steps["run_id"].map(runs.set_index("run_id")["maximize"])
    if maximize.isna().any():
        raise ValueError("steps contain run ids that are not in runs.")
    out = steps.copy()
    sign = np.where(maximize.astype(bool), -1.0, 1.0)
    out["regret"] = sign * (out["user_cost"] - out["vbs_cost"])
    out["extra_run"] = (out["n_runs"] > 1).astype(float)
    out["extra_cpu"] = out["compute_cost"] - out["primary_runtime"]
    out["overhead_seconds"] = out["selector_seconds"] + out["detector_seconds"]
    out["cum_regret"] = out.groupby("run_id")["regret"].cumsum()
    return out


def rolling(steps: pd.DataFrame, column: str, window: int = WINDOW) -> pd.Series:
    """Trailing mean of ``column`` over ``window`` steps, within each run.

    The first ``window - 1`` steps of every run are NaN. The result is aligned
    with ``steps``.
    """
    means = steps.groupby("run_id", sort=False)[column].rolling(window).mean()
    return means.reset_index(level=0, drop=True)


def rolling_curves(
    steps: pd.DataFrame, column: str, window: int = WINDOW
) -> pd.DataFrame:
    """Per-run rolling curves as a long frame ``[run_id, step, value]``.

    For derived columns (``regret``, ...) pass the output of
    :func:`add_step_metrics`.
    """
    return pd.DataFrame(
        {
            "run_id": steps["run_id"],
            "step": steps["step"],
            "value": rolling(steps, column, window),
        }
    )


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else float("nan")


def _drift_row(
    regret: np.ndarray,
    extra: np.ndarray,
    point: int,
    start: int,
    end: int,
    window: int,
    tolerance: float,
) -> dict[str, float]:
    level = _mean(regret[max(point - window, 0) : point])
    segment = regret[point:end]
    # means[i] = mean regret over steps point+i .. point+i+window-1
    means = (
        sliding_window_view(segment, window).mean(axis=1)
        if len(segment) >= window
        else np.empty(0)
    )
    peak = float(means.max()) if len(means) else float("nan")
    # Recovered once at most `tolerance` of the jump above the pre-drift level
    # is left; without a jump (peak <= level) that is the level itself.
    threshold = level + tolerance * max(peak - level, 0.0)
    hits = np.flatnonzero(means[start - point :] <= threshold)
    return {
        "pre_level": level,
        "peak_regret": peak,
        "recovery": float(hits[0]) if hits.size else float("nan"),
        "extra_run_rate_before": _mean(extra[max(point - window, 0) : point]),
        "extra_run_rate_after": _mean(extra[point : min(point + window, end)]),
    }


def drift_metrics(
    steps: pd.DataFrame,
    runs: pd.DataFrame,
    window: int = WINDOW,
    tolerance: float = TOLERANCE,
) -> pd.DataFrame:
    """Recovery, peak regret and extra-run rates around every drift point.

    For a drift point ``p`` with segment end ``end`` (the next drift point or
    the end of the stream) and ``start`` (``p``, or the end of the transition
    interval that begins at ``p`` for gradual drift):

    - ``pre_level``: mean regret over the steps ``[p - window, p)`` (fewer at
      the start of the stream, NaN if there are none).
    - ``peak_regret``: largest mean regret of a full window inside
      ``[p, end)``.
    - ``recovery``: ``s - start`` for the first ``s >= start`` whose window
      ``[s, s + window)`` lies inside the segment and has mean regret at most
      ``pre_level + tolerance * (peak_regret - pre_level)``, i.e. all but
      ``tolerance`` of the jump is gone (the level itself if there is no
      jump). This is the trailing rolling regret at step ``s + window - 1``.
      0 means no visible degradation, NaN means never recovered, or not
      measurable (no ``pre_level``, or the segment is shorter than
      ``window``).
    - ``extra_run_rate_before`` / ``_after``: share of steps with more than
      one run in ``[p - window, p)`` and ``[p, min(p + window, end))``.

    Parameters
    ----------
    steps
        Per-step frame; ``regret`` is added if missing.
    runs
        Run frame with ``drift_points`` and ``transition_intervals``.
    window
        Window length in steps for all windows above.
    tolerance
        Share of the jump above the pre-drift level that may remain.

    Returns
    -------
    pd.DataFrame
        One row per run and drift point, columns :data:`DRIFT_COLUMNS`.
        Transition intervals that start before the evaluated stream (warm
        start) have no drift point and are ignored.
    """
    if "regret" not in steps:
        steps = add_step_metrics(steps, runs)
    by_run = dict(tuple(steps.groupby("run_id", sort=False)))
    rows = []
    for run_id, points, intervals in zip(
        runs["run_id"],
        runs["drift_points"],
        runs["transition_intervals"],
        strict=True,
    ):
        if run_id not in by_run or not points:
            continue
        run = by_run[run_id]
        regret = run["regret"].to_numpy(float)
        extra = (run["n_runs"] > 1).to_numpy(float)
        transition_end = {a: b for a, b in intervals}
        for i, (point, end) in enumerate(
            zip(points, [*points[1:], len(run)], strict=True)
        ):
            start = transition_end.get(point, point)
            rows.append(
                {
                    "run_id": run_id,
                    "drift_index": i,
                    "drift_point": point,
                    "start": start,
                    "end": end,
                    **_drift_row(regret, extra, point, start, end, window, tolerance),
                }
            )
    return pd.DataFrame(rows, columns=DRIFT_COLUMNS)


def _detection(runs: pd.DataFrame, tolerance: int) -> pd.DataFrame:
    from asf.online.evaluation.metrics import detection_metrics  # slow import

    empty = {"mean_delay": np.nan, "n_missed": np.nan, "n_false_alarms": np.nan}
    rows = [
        detection_metrics(points, detected, tolerance) if points else empty
        for points, detected in zip(
            runs["drift_points"], runs["drift_steps"], strict=True
        )
    ]
    return pd.DataFrame(rows, index=runs["run_id"])


def run_table(
    runs: pd.DataFrame,
    steps: pd.DataFrame,
    window: int = WINDOW,
    tolerance: float = TOLERANCE,
    detection_tolerance: int = DETECTION_TOLERANCE,
) -> pd.DataFrame:
    """One row per run: config columns, ``uses_detector``, :data:`METRICS`, ``param.*``.

    ``uses_detector`` is True when the selector has a drift detector (its
    ``detector`` parameter is set). Detection metrics (``mean_delay``,
    ``n_missed``, ``n_false_alarms``) are NaN for runs without one.

    ``solver_cpu`` is the total ``compute_cost``, ``extra_cpu`` and
    ``overhead_seconds`` are totals of the step metrics, ``mean_recovery``
    averages the drift points that recovered and ``n_not_recovered`` counts the
    others (drift points where recovery cannot be measured, e.g. a drift at
    the first evaluated step, count as neither); ``mean_peak_regret``
    averages all drift points. Detection metrics
    are recomputed with ``detection_tolerance`` (ASF's ``detection_metrics``).
    Drift metrics are NaN for runs without drift points.
    """
    steps = add_step_metrics(steps, runs)
    indexed = runs.set_index("run_id")
    drift = drift_metrics(steps, runs, window, tolerance).groupby("run_id")
    derived = pd.DataFrame(
        {
            "closed_gap": indexed["closed_gap"],
            "mean_user_cost": indexed["mean_user_cost"],
            "solver_cpu": indexed["total_compute_cost"],
            # min_count=1: all-NaN (maximisation) stays NaN
            "extra_cpu": steps.groupby("run_id")["extra_cpu"].sum(min_count=1),
            "overhead_seconds": indexed["total_selector_seconds"]
            + indexed["total_detector_seconds"],
            "extra_run_rate": indexed["extra_run_rate"],
            "n_drift_signals": indexed["n_drift_signals"],
        }
    )
    uses_detector = pd.Series(_uses_detector(runs).to_numpy(), index=runs["run_id"])
    derived = derived.join(
        _detection(runs, detection_tolerance).where(uses_detector, axis=0)
    )
    derived["mean_recovery"] = drift["recovery"].mean()
    derived["mean_peak_regret"] = drift["peak_regret"].mean()
    derived["n_not_recovered"] = drift[["pre_level", "peak_regret", "recovery"]].apply(
        _not_recovered
    )
    config = [c for c in CONFIG_COLUMNS if c in runs]
    params = [c for c in runs if c.startswith("param.")]
    return pd.concat(
        [
            runs[config],
            pd.DataFrame({"uses_detector": uses_detector.to_numpy()}, index=runs.index),
            derived[METRICS].reindex(runs["run_id"]).set_axis(runs.index),
            runs[params],
        ],
        axis=1,
    )


def _not_recovered(drift: pd.DataFrame) -> int:
    """Drift points with a measurable recovery that never recovered."""
    measurable = drift["pre_level"].notna() & drift["peak_regret"].notna()
    return int((measurable & drift["recovery"].isna()).sum())


def _uses_detector(runs: pd.DataFrame) -> pd.Series:
    """True for runs whose selector has a drift detector (``detector`` param)."""
    return runs.get("param.detector.class", pd.Series(index=runs.index)).notna()


def _keys(by: Sequence[str]) -> list[str]:
    """Group keys; ``method_family`` and ``uses_detector`` ride along with
    ``method`` (both are constant per method)."""
    keys = list(by)
    if "method" in keys:
        at = keys.index("method") + 1
        keys[at:at] = [k for k in ("method_family", "uses_detector") if k not in keys]
    return keys


Frame = pd.Series | pd.DataFrame


def _ci(std: Frame, count: Frame, level: float) -> Frame:
    """Half-width of the t-interval; NaN for fewer than two values."""
    return std / np.sqrt(count) * stats.t.ppf(0.5 + level / 2, count - 1)


def aggregate(
    table: pd.DataFrame,
    by: Sequence[str] = GROUP,
    metrics: Sequence[str] = METRICS,
    confidence: float = CONFIDENCE,
) -> pd.DataFrame:
    """Mean, std and t-interval half-width of every metric over seeds.

    Returns
    -------
    pd.DataFrame
        The group keys (plus ``method_family`` and ``uses_detector`` when
        grouping by ``method``), ``<m>_mean``, ``<m>_std``, ``<m>_ci95`` per
        metric, and ``n`` (runs in the group). std uses ddof=1; mean, std and
        CI skip NaN values, so a metric with fewer than two values has NaN std
        and CI. The column name
        stays ``_ci95`` for any ``confidence``.
    """
    if "uses_detector" not in table:
        table = table.assign(uses_detector=_uses_detector(table))
    groups = table.groupby(_keys(by), sort=False)
    values = groups[list(metrics)]
    mean, std, count = values.mean(), values.std(), values.count()
    ci = _ci(std, count, confidence)
    columns = {}
    for m in metrics:
        columns |= {f"{m}_mean": mean[m], f"{m}_std": std[m], f"{m}_ci95": ci[m]}
    columns["n"] = groups.size()
    return pd.DataFrame(columns).reset_index()


def aggregate_curves(
    curves: pd.DataFrame,
    runs: pd.DataFrame,
    by: Sequence[str] = GROUP,
    confidence: float = CONFIDENCE,
) -> pd.DataFrame:
    """Mean curve and t-interval per group and step.

    Parameters
    ----------
    curves
        Output of :func:`rolling_curves`.
    runs
        Run frame with the group keys and ``drift_points``.

    Returns
    -------
    pd.DataFrame
        Columns ``[*by, method_family, uses_detector, step, mean, lo, hi]``;
        for the default ``by``: ``[stream, protocol, method, method_family,
        uses_detector, step, mean, lo, hi]``. Raises if the runs of a group
        have different drift points, since then a step does not mean the same
        position in the stream.
    """
    runs = runs.assign(uses_detector=_uses_detector(runs))
    keys = _keys(by)
    n_drifts = runs.groupby(keys)["drift_points"].agg(lambda s: len(set(map(tuple, s))))
    if (n_drifts > 1).any():
        raise ValueError(
            "runs of a group have different drift points: "
            f"{list(n_drifts[n_drifts > 1].index)}"
        )
    merged = curves.merge(runs[["run_id", *keys]], on="run_id")
    values = merged.groupby([*keys, "step"], sort=False)["value"]
    mean = values.mean()
    ci = _ci(values.std(), values.count(), confidence)
    out = pd.DataFrame({"mean": mean, "lo": mean - ci, "hi": mean + ci})
    return out.reset_index()
