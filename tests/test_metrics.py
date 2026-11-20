from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from conftest import FIXTURE_RESULTS
from oas_experiments.analysis.metrics import (
    CONFIG_COLUMNS,
    DRIFT_COLUMNS,
    METRICS,
    add_step_metrics,
    aggregate,
    aggregate_curves,
    drift_metrics,
    rolling,
    rolling_curves,
    run_table,
)
from oas_experiments.results import load_experiment

ROOK = "sudden-s0__cold__rook__seed0"


def _steps(run_id, regret, n_runs=None):
    n = len(regret)
    return pd.DataFrame(
        {
            "run_id": run_id,
            "step": np.arange(n),
            "user_cost": np.asarray(regret, float),
            "vbs_cost": 0.0,
            "n_runs": n_runs if n_runs is not None else 1,
            "compute_cost": 1.0,
            "primary_runtime": 1.0,
            "selector_seconds": 0.0,
            "detector_seconds": 0.0,
        }
    )


def _runs(run_id, points, intervals=()):
    return pd.DataFrame(
        {
            "run_id": [run_id],
            "maximize": [False],
            "drift_points": [list(points)],
            "transition_intervals": [[list(i) for i in intervals]],
        }
    )


def _drift(regret, points, intervals=(), n_runs=None, window=2):
    out = drift_metrics(
        _steps("r", regret, n_runs), _runs("r", points, intervals), window=window
    )
    return out.set_index("drift_point")


@pytest.fixture(scope="module")
def fixture():
    return load_experiment(FIXTURE_RESULTS / "fixture_exp")


# --- step metrics ---------------------------------------------------------


def test_step_metrics_by_hand():
    steps = pd.DataFrame(
        {
            "run_id": "r",
            "user_cost": [3.0, 4, 5, 6, 7],
            "vbs_cost": [1.0, 1, 2, 6, 0],
            "compute_cost": [5.0, 4, 10, 6, 7],
            "primary_runtime": [3.0, 4, 5, 6, 7],
            "selector_seconds": [0.1, 0.2, 0.3, 0.4, 0.5],
            "detector_seconds": [0.01, 0, 0, 0, 0],
            "n_runs": [1, 2, 1, 1, 3],
        }
    )
    out = add_step_metrics(steps, _runs("r", []))
    assert out["regret"].tolist() == [2, 3, 3, 0, 7]
    assert out["cum_regret"].tolist() == [2, 5, 8, 8, 15]
    assert out["extra_cpu"].tolist() == [2, 0, 5, 0, 0]
    assert out["extra_run"].tolist() == [0, 1, 0, 0, 1]
    assert out["overhead_seconds"].tolist() == pytest.approx([0.11, 0.2, 0.3, 0.4, 0.5])
    assert "regret" not in steps  # input untouched


def test_step_metrics_maximisation():
    steps = pd.DataFrame(
        {
            "run_id": "m",
            "user_cost": [1.0, 2.0],
            "vbs_cost": [3.0, 2.0],
            "compute_cost": np.nan,
            "primary_runtime": [1.0, 2.0],
            "selector_seconds": 0.0,
            "detector_seconds": 0.0,
            "n_runs": 1,
        }
    )
    runs = _runs("m", []).assign(maximize=True)
    out = add_step_metrics(steps, runs)
    assert out["regret"].tolist() == [2, 0]
    assert out["extra_cpu"].isna().all()


def test_step_metrics_reject_unknown_runs():
    with pytest.raises(ValueError, match="not in runs"):
        add_step_metrics(_steps("x", [1.0]), _runs("r", []))


def test_rolling_stays_within_runs():
    steps = pd.concat([_steps("a", [1, 3]), _steps("b", [5, 7])], ignore_index=True)
    out = rolling(steps, "user_cost", window=2)
    assert out.tolist()[1::2] == [2, 6]
    assert out.isna().tolist()[::2] == [True, True]
    curves = rolling_curves(steps, "user_cost", window=2)
    assert list(curves.columns) == ["run_id", "step", "value"]


# --- drift metrics --------------------------------------------------------


def test_recovery_at_step_k():
    row = _drift([1, 1, 1, 1, 5, 5, 3, 1, 1, 1, 1], [4]).loc[4]
    # windows from step 4: 5, 4, 2, 1 -> back at 1 * 1.1 from step 7
    assert row["pre_level"] == 1
    assert row["recovery"] == 3
    assert row["peak_regret"] == 5
    assert (row["start"], row["end"]) == (4, 11)


def test_recovery_from_a_near_zero_level():
    # level 0, peak 2: recovered once the window mean is at most 0.2
    row = _drift([0, 0, 0, 0, 2, 2, 0.1, 0.1, 0.1], [4]).loc[4]
    assert row["pre_level"] == 0 and row["peak_regret"] == 2
    assert row["recovery"] == 2


def test_no_jump_recovers_at_once():
    row = _drift([2, 2, 2, 2, 1, 1, 1, 1], [4]).loc[4]
    assert row["recovery"] == 0


def test_no_recovery_is_nan():
    row = _drift([1, 1, 1, 1, 5, 5, 5, 5], [4]).loc[4]
    assert np.isnan(row["recovery"]) and row["peak_regret"] == 5


def test_recovery_searched_only_until_next_drift_point():
    out = _drift([1, 1, 1, 1, 5, 5, 5, 5, 1, 1, 1, 1], [4, 8])
    assert np.isnan(out.loc[4, "recovery"])  # windows crossing step 8 not used
    assert out.loc[4, "end"] == 8
    assert out.loc[8, "pre_level"] == 5
    assert out.loc[8, "recovery"] == 0 and out.loc[8, "peak_regret"] == 1
    assert out["drift_index"].tolist() == [0, 1]


def test_drift_at_first_step():
    row = _drift([5, 5, 1, 1], [0]).loc[0]
    assert np.isnan(row["pre_level"]) and np.isnan(row["recovery"])
    assert np.isnan(row["extra_run_rate_before"])
    assert row["peak_regret"] == 5


def test_gradual_drift_measured_from_transition_end():
    regret = [1, 1, 1, 1, 3, 5, 5, 3, 1, 1, 1, 1]
    gradual = _drift(regret, [4], intervals=[(4, 8)]).loc[4]
    assert gradual["start"] == 8
    assert gradual["recovery"] == 0
    assert gradual["peak_regret"] == 5  # peak counted from the drift point
    assert _drift(regret, [4]).loc[4, "recovery"] == 4


def test_extra_run_rate_around_drift():
    n_runs = [1, 1, 1, 2, 2, 2, 1, 1]
    row = _drift([1] * 8, [4], n_runs=n_runs).loc[4]
    assert row["extra_run_rate_before"] == 0.5
    assert row["extra_run_rate_after"] == 1.0


def test_runs_without_drift_points_give_no_rows():
    out = drift_metrics(_steps("r", [1, 2, 3]), _runs("r", []))
    assert out.empty and list(out.columns) == DRIFT_COLUMNS


# --- run table, aggregates, curves on the fixture -------------------------


def test_run_table_on_fixture(fixture):
    table = run_table(fixture.runs, fixture.steps, window=5)
    params = [c for c in fixture.runs if c.startswith("param.")]
    assert list(table.columns) == [*CONFIG_COLUMNS, "uses_detector", *METRICS, *params]
    assert table["run_id"].tolist() == fixture.runs["run_id"].tolist()
    row = table.set_index("run_id").loc[ROOK]
    run = fixture.runs.set_index("run_id").loc[ROOK]
    steps = fixture.steps[fixture.steps["run_id"] == ROOK]
    assert row["solver_cpu"] == pytest.approx(steps["compute_cost"].sum())
    assert row["extra_cpu"] == pytest.approx(
        steps["compute_cost"].sum() - steps["primary_runtime"].sum()
    )
    assert row["overhead_seconds"] == pytest.approx(
        steps["selector_seconds"].sum() + steps["detector_seconds"].sum()
    )
    for column in ["closed_gap", "mean_delay", "n_missed", "n_false_alarms"]:
        assert row[column] == run[column]
    assert row["mean_delay"] == 4  # signal at 29, drift at 25
    drift = drift_metrics(fixture.steps, fixture.runs, window=5)
    assert len(drift) == 10
    assert row["mean_peak_regret"] == drift.set_index("run_id").loc[ROOK, "peak_regret"]


def test_unmeasurable_recovery_is_not_counted_as_not_recovered(fixture):
    runs = fixture.runs.copy()
    runs["drift_points"] = [[0]] * len(runs)  # no pre-drift level anywhere
    table = run_table(runs, fixture.steps, window=5)
    assert (table["n_not_recovered"] == 0).all()
    assert table["mean_recovery"].isna().all()


def test_detection_metrics_only_for_detector_methods(fixture):
    runs = fixture.runs.copy()
    runs["drift_steps"] = [
        [] if r == ROOK else d
        for r, d in zip(runs["run_id"], runs["drift_steps"], strict=True)
    ]
    table = run_table(runs, fixture.steps, window=5).set_index("run_id")
    silent = table.loc[ROOK]  # detector that never fires: still reported
    assert silent["uses_detector"] and silent["n_missed"] == 1
    assert np.isnan(silent["mean_delay"]) and silent["n_false_alarms"] == 0
    bandit = table.loc["sudden-s0__cold__ds-ts__seed0"]
    assert not bandit["uses_detector"]
    assert bandit[["mean_delay", "n_missed", "n_false_alarms"]].isna().all()
    assert table["uses_detector"].sum() == 4  # the rook runs


def test_run_table_maximisation(fixture):
    runs, steps = fixture.runs.copy(), fixture.steps.copy()
    is_rook = steps["run_id"] == ROOK
    steps.loc[is_rook, "compute_cost"] = np.nan
    runs.loc[runs["run_id"] == ROOK, ["maximize", "total_compute_cost"]] = [
        True,
        np.nan,
    ]
    row = run_table(runs, steps, window=5).set_index("run_id").loc[ROOK]
    assert np.isnan(row["solver_cpu"]) and np.isnan(row["extra_cpu"])
    regret = add_step_metrics(steps, runs).loc[is_rook, "regret"]
    assert (regret <= 0).all()  # user >= vbs in this minimisation fixture: flipped


def test_aggregate_over_seeds(fixture):
    table = run_table(fixture.runs, fixture.steps, window=5)
    summary = aggregate(table)
    assert list(summary.columns[:5]) == [
        "stream",
        "protocol",
        "method",
        "method_family",
        "uses_detector",
    ]
    assert summary.columns[-1] == "n"
    assert [f"{m}_{s}" for m in METRICS for s in ("mean", "std", "ci95")] == list(
        summary.columns[5:-1]
    )
    assert len(summary) == 5 and (summary["n"] == 2).all()
    row = summary.set_index(["protocol", "method"]).loc[("cold", "rook")]
    values = table.query("protocol == 'cold' and method == 'rook'")["closed_gap"]
    assert row["closed_gap_mean"] == pytest.approx(values.mean())
    assert row["closed_gap_ci95"] == pytest.approx(
        values.std() / np.sqrt(2) * stats.t.ppf(0.975, 1)
    )
    single = aggregate(table, by=["stream", "protocol", "method", "seed"])
    assert (single["n"] == 1).all() and single["closed_gap_ci95"].isna().all()


def test_aggregate_curves(fixture):
    steps = add_step_metrics(fixture.steps, fixture.runs)
    curves = rolling_curves(steps, "regret", window=5)
    out = aggregate_curves(curves, fixture.runs)
    assert list(out.columns) == [
        "stream",
        "protocol",
        "method",
        "method_family",
        "uses_detector",
        "step",
        "mean",
        "lo",
        "hi",
    ]
    assert len(out) == 2 * 40 + 3 * 30
    row = out.set_index(["protocol", "method", "step"]).loc[("cold", "rook", 10)]
    ids = fixture.runs.query("protocol == 'cold' and method == 'rook'")["run_id"]
    expected = curves[curves["run_id"].isin(ids) & (curves["step"] == 10)]["value"]
    assert row["mean"] == pytest.approx(expected.mean())
    assert row["lo"] <= row["mean"] <= row["hi"]


def test_aggregate_curves_rejects_misaligned_runs(fixture):
    runs = fixture.runs.copy()
    runs["drift_points"] = [
        [3] if r == ROOK else p
        for r, p in zip(runs["run_id"], runs["drift_points"], strict=True)
    ]
    curves = rolling_curves(fixture.steps, "user_cost", window=5)
    with pytest.raises(ValueError, match="different drift points"):
        aggregate_curves(curves, runs)
