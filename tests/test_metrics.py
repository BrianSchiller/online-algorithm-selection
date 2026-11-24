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
    _not_recovered,
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


def _steps(run_id, regret, n_runs=None, gap=10.0, vbs_cost=0.0, maximize=False):
    """Steps with the given regret and gap ``sbs_cost - vbs_cost`` (both
    flipped when maximising)."""
    n = len(regret)
    sign = -1.0 if maximize else 1.0
    vbs = np.broadcast_to(np.asarray(vbs_cost, float), n)
    return pd.DataFrame(
        {
            "run_id": run_id,
            "step": np.arange(n),
            "user_cost": vbs + sign * np.asarray(regret, float),
            "vbs_cost": vbs,
            "sbs_cost": vbs + sign * np.broadcast_to(np.asarray(gap, float), n),
            "n_runs": n_runs if n_runs is not None else 1,
            "compute_cost": 1.0,
            "primary_runtime": 1.0,
            "selector_seconds": 0.0,
            "detector_seconds": 0.0,
        }
    )


def _runs(run_id, points, intervals=(), maximize=False):
    return pd.DataFrame(
        {
            "run_id": [run_id],
            "maximize": [maximize],
            "drift_points": [list(points)],
            "transition_intervals": [[list(i) for i in intervals]],
        }
    )


def _drift(
    regret,
    points,
    intervals=(),
    n_runs=None,
    window=2,
    hold=2,
    cap=0.1,
    gap=10.0,
    vbs_cost=0.0,
    maximize=False,
):
    out = drift_metrics(
        _steps("r", regret, n_runs, gap, vbs_cost, maximize),
        _runs("r", points, intervals, maximize),
        window=window,
        hold=hold,
        cap=cap,
    )
    return out.set_index("drift_point")


def _n_not_recovered(drift, window=2, hold=2):
    return _not_recovered(drift.reset_index(), window, hold)


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


# Window 2, tolerance 0.1, gap 10 per step unless set, so a window's closed
# gap is 1 - (sum of its two regrets) / 20.


def test_harder_segment_is_not_degradation():
    # regret and gap both x4 after the drift: same closed gap 0.9 throughout
    gap = [10] * 4 + [40] * 6
    row = _drift([1, 1, 1, 1, 4, 4, 4, 4, 4, 4], [4], gap=gap).loc[4]
    assert row["pre_gap"] == pytest.approx(0.9)
    assert row["low_gap"] == pytest.approx(0.9)
    assert row["peak_regret"] == 4
    assert row["recovery"] == 0  # the raw-regret definition gave NaN here


def test_degradation_then_recovery():
    # windows a = 4..10: 0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.9
    # threshold 0.9 - 0.1 * (0.9 - 0.5) = 0.86; first s with G(s), G(s+2) >= 0.86: 8
    regret = [1, 1, 1, 1, 5, 5, 3, 3, 1, 1, 1, 1]
    row = _drift(regret, [4]).loc[4]
    assert row["pre_gap"] == pytest.approx(0.9)
    assert row["low_gap"] == pytest.approx(0.5)
    assert row["peak_regret"] == 5
    assert row["recovery"] == 4


def test_cap_keeps_a_deep_drop_from_lowering_the_bar():
    # windows a = 4..12: -2, -0.625, 0.75, 0.75, 0.75, 0.825, 0.9, 0.9, 0.9
    # drop 2.9: uncapped threshold 0.9 - 0.29 = 0.61, capped 0.9 - 0.1 = 0.8
    regret = [1, 1, 1, 1, 30, 30, 2.5, 2.5, 2.5, 2.5, 1, 1, 1, 1]
    row = _drift(regret, [4]).loc[4]
    assert row["low_gap"] == pytest.approx(-2)
    assert row["recovery"] == 5  # s = 9: G(9) = 0.825, G(11) = 0.9
    assert _drift(regret, [4], cap=np.inf).loc[4, "recovery"] == 2


def test_one_lucky_window_does_not_hold():
    # windows a = 4..12: 0.5, 0.7, 0.9, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5
    # only G(6) >= 0.86, and G(8) = 0.5 breaks the hold
    regret = [1, 1, 1, 1, 5, 5, 1, 1, 5, 5, 5, 5, 5, 5]
    out = _drift(regret, [4])
    assert np.isnan(out.loc[4, "recovery"])
    assert _n_not_recovered(out) == 1
    single = _drift(regret, [4], hold=1)
    assert single.loc[4, "recovery"] == 2
    assert _n_not_recovered(single, hold=1) == 0


def test_zero_gap_windows_are_nan_and_break_a_hold():
    # gap 0 on steps 4, 5: G(4) is NaN; G(5..8) = 0.9 = threshold
    regret = [1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
    gap = [10, 10, 10, 10, 0, 0, 10, 10, 10, 10]
    row = _drift(regret, [4], gap=gap).loc[4]
    assert row["low_gap"] == pytest.approx(0.9)  # NaN windows ignored
    assert row["recovery"] == 1  # s = 4 fails on the NaN window


def test_zero_gap_everywhere_after_drift_is_not_recovered():
    out = _drift([1] * 10, [4], gap=[10] * 4 + [0] * 6)
    assert np.isnan(out.loc[4, "low_gap"]) and np.isnan(out.loc[4, "recovery"])
    assert _n_not_recovered(out) == 1


def test_zero_gap_before_drift_is_not_measurable():
    out = _drift([1] * 10, [4], gap=[0] * 4 + [10] * 6)
    assert np.isnan(out.loc[4, "pre_gap"]) and np.isnan(out.loc[4, "recovery"])
    assert _n_not_recovered(out) == 0


def test_segment_shorter_than_hold_windows_is_not_measurable():
    # segment [4, 7) has 3 < hold * window = 4 steps
    out = _drift([1, 1, 1, 1, 5, 5, 5, 1, 1, 1, 1], [4, 7])
    assert np.isnan(out.loc[4, "recovery"])
    assert out.loc[7, "recovery"] == 0
    assert _n_not_recovered(out) == 0
    # with hold=1 the segment is long enough, and the drop never recovers
    assert (
        _n_not_recovered(
            _drift([1, 1, 1, 1, 5, 5, 5, 1, 1, 1, 1], [4, 7], hold=1), hold=1
        )
        == 1
    )


def test_drift_at_first_step():
    out = _drift([5, 5, 1, 1, 1, 1], [0])
    row = out.loc[0]
    assert np.isnan(row["pre_gap"]) and np.isnan(row["recovery"])
    assert np.isnan(row["extra_run_rate_before"])
    assert row["peak_regret"] == 5
    assert _n_not_recovered(out) == 0


def test_recovery_searched_only_until_next_drift_point():
    out = _drift([1, 1, 1, 1, 5, 5, 5, 5, 1, 1, 1, 1], [4, 8])
    # [4, 8): all windows 0.5; windows crossing step 8 (G(7) = 0.7) not used
    assert out.loc[4, "low_gap"] == pytest.approx(0.5)
    assert np.isnan(out.loc[4, "recovery"])
    assert out.loc[4, "end"] == 8
    assert out.loc[8, "pre_gap"] == pytest.approx(0.5)
    assert out.loc[8, "recovery"] == 0 and out.loc[8, "peak_regret"] == 1
    assert _n_not_recovered(out) == 1


def test_gradual_drift_measured_from_transition_end():
    # windows a = 4..10: 0.6, 0.5, 0.6, 0.8, 0.9, 0.9, 0.9; threshold 0.86
    regret = [1, 1, 1, 1, 3, 5, 5, 3, 1, 1, 1, 1]
    gradual = _drift(regret, [4], intervals=[(4, 8)]).loc[4]
    assert gradual["start"] == 8
    assert gradual["recovery"] == 0
    assert gradual["low_gap"] == pytest.approx(0.5)  # trough counted from the drift
    assert gradual["peak_regret"] == 5
    assert _drift(regret, [4]).loc[4, "recovery"] == 4


def test_recovery_when_maximising():
    # the degradation case with every cost difference flipped around vbs = 10
    regret = [1, 1, 1, 1, 5, 5, 3, 3, 1, 1, 1, 1]
    row = _drift(regret, [4], vbs_cost=10.0, maximize=True).loc[4]
    assert row["pre_gap"] == pytest.approx(0.9)
    assert row["low_gap"] == pytest.approx(0.5)
    assert row["peak_regret"] == 5
    assert row["recovery"] == 4


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
