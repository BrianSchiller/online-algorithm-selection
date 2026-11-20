from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.collections import EventCollection  # noqa: E402
from matplotlib.container import ErrorbarContainer  # noqa: E402

from contract_frames import agg_frame, curves_frame, runs_frame  # noqa: E402
from oas_experiments.analysis import plots, style  # noqa: E402


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _vlines(ax):
    return [line for line in ax.lines if line.get_label().startswith("_")]


def _fronts(ax):
    return {c.get_label(): c for c in ax.containers if isinstance(c, ErrorbarContainer)}


def test_family_styles_are_sorted_and_stable():
    styles = style.family_styles(["b", "a", "b"])
    assert list(styles) == ["a", "b"]
    assert styles["a"]["color"] == style.PALETTE[0]
    assert styles == style.family_styles(["a", "b"])


def test_family_styles_beyond_palette_change_line_and_fill():
    styles = style.family_styles([f"f{i:02d}" for i in range(9)])
    assert styles["f08"]["color"] == styles["f00"]["color"]
    assert (styles["f08"]["dash"], styles["f08"]["fill"]) == (1, "none")


def test_regret_curves_draw_methods_drift_and_transitions():
    fig, ax = plots.regret_curves(curves_frame(), runs_frame(), "switch", "warm")
    labels = [
        line.get_label() for line in ax.lines if not line.get_label().startswith("_")
    ]
    assert sorted(labels) == ["ds-ts", "rook", "single", "unc-rate0.1", "unc-rate0.5"]
    assert len(_vlines(ax)) == 2  # drift points
    assert len(ax.patches) == 1  # transition interval
    assert ax.get_ylabel() == "Rolling regret"
    assert ax.get_legend() is not None


def test_regret_curves_family_shares_colour_and_varies_style():
    _, ax = plots.regret_curves(curves_frame(), runs_frame(), "switch", "warm")
    lines = {line.get_label(): line for line in ax.lines}
    a, b = lines["unc-rate0.1"], lines["unc-rate0.5"]
    assert a.get_color() == b.get_color()
    assert a.get_linestyle() != b.get_linestyle()


def _band_ys(ax):
    """y values of all CI bands (single-seed methods have none)."""
    return np.concatenate(
        [p.vertices[:, 1] for c in ax.collections for p in c.get_paths()]
    )


def test_bands_are_clipped_to_the_possible_range():
    curves = curves_frame()  # the first method has mean 0 and lo = -0.05
    _, ax = plots.regret_curves(curves, runs_frame(), "switch", "warm")
    lowest = _band_ys(ax).min()
    assert lowest == 0
    curves["hi"] += 1  # above a rate of 1
    _, ax = plots.extra_run_rate(curves, runs_frame(), "switch", "warm")
    highest = _band_ys(ax).max()
    assert highest == 1


def test_curves_without_drift_and_method_filter():
    _, ax = plots.extra_run_rate(
        curves_frame(), runs_frame(), "calm", "cold", methods=["rook"]
    )
    assert [line.get_label() for line in ax.lines] == ["rook"]
    assert not ax.patches
    assert ax.get_ylabel() == "Extra-run rate"


def test_curves_into_given_axes():
    fig, given = plt.subplots()
    out_fig, ax = plots.regret_curves(
        curves_frame(), runs_frame(), "switch", "cold", ax=given
    )
    assert (out_fig, ax) == (fig, given)


def test_front_one_line_per_family_and_single_markers():
    _, ax = plots.cost_quality_front(agg_frame(), "switch", "warm")
    fronts = _fronts(ax)
    assert sorted(fronts) == ["ds-ts", "rook", "single", "unc"]
    unc = fronts["unc"].lines[0]
    assert len(unc.get_xdata()) == 2 and unc.get_linestyle() == "-"
    assert list(unc.get_xdata()) == sorted(unc.get_xdata())
    assert fronts["rook"].lines[0].get_linestyle() == "None"
    assert unc.get_marker() == style.PROTOCOL_MARKERS["warm"]
    assert ax.get_xlabel() == "Solver CPU (s)" and ax.get_ylabel() == "Closed gap"


def test_front_error_bars_skip_missing_ci():
    agg = agg_frame()
    _, ax = plots.cost_quality_front(agg, "switch", "cold")
    single = _fronts(ax)["single"]  # one seed: ci95 is NaN -> zero-length bars
    for bars in single.lines[2]:
        assert np.allclose(np.ptp(np.concatenate(bars.get_segments()), axis=0), 0)
    rook = _fronts(ax)["rook"]
    assert len(rook.lines[2]) == 2  # x and y error bars


def test_front_overhead_adds_to_x():
    agg = agg_frame()
    _, plain = plots.cost_quality_front(agg, "switch", "warm")
    _, extra = plots.cost_quality_front(agg, "switch", "warm", overhead=True)
    x0 = _fronts(plain)["rook"].lines[0].get_xdata()
    x1 = _fronts(extra)["rook"].lines[0].get_xdata()
    row = agg[
        (agg.stream == "switch") & (agg.protocol == "warm") & (agg.method == "rook")
    ]
    assert np.allclose(x1 - x0, row["overhead_seconds_mean"])
    assert extra.get_xlabel() == "Solver CPU + overhead (s)"


def test_front_drops_nan_cpu_with_warning():
    agg = agg_frame()
    agg.loc[agg.method == "ds-ts", "solver_cpu_mean"] = np.nan
    with pytest.warns(UserWarning, match="ds-ts"):
        _, ax = plots.cost_quality_front(agg, "switch", "warm")
    assert "ds-ts" not in _fronts(ax)


def test_front_without_any_cpu_says_so():
    agg = agg_frame()
    agg["solver_cpu_mean"] = np.nan
    with pytest.warns(UserWarning):
        _, ax = plots.cost_quality_front(agg, "calm", "cold")
    assert not _fronts(ax)
    assert [t.get_text() for t in ax.texts] == ["no CPU data"]


def test_front_grid_has_one_panel_per_stream_and_protocol():
    fig, axes = plots.front_grid(agg_frame())
    assert axes.shape == (2, 2)  # protocols x streams
    assert [ax.get_title() for ax in axes.flat] == [
        "calm, cold",
        "switch, cold",
        "calm, warm",
        "switch, warm",
    ]
    assert all(ax.get_legend() is None for ax in axes.flat)
    assert len(fig.legends) == 1


def test_front_grid_hides_missing_panels():
    agg = agg_frame()
    agg = agg[~((agg.stream == "calm") & (agg.protocol == "warm"))]
    _, axes = plots.front_grid(agg)
    assert [ax.get_visible() for ax in axes.flat] == [True, True, False, True]


def test_raster_rows_are_detector_runs():
    _, ax = plots.detection_raster(runs_frame(), "switch", "warm")
    events = [c for c in ax.collections if isinstance(c, EventCollection)]
    assert [list(e.get_positions()) for e in events] == [[105, 210], [105, 211]]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["rook s0", "rook s1"]
    assert len(_vlines(ax)) == 2 and len(ax.patches) == 1


def test_raster_spans_the_evaluated_stream():
    runs = runs_frame().assign(n_evaluated=400)
    _, ax = plots.detection_raster(runs, "switch", "warm")
    assert ax.get_xlim() == (0, 400)


def test_raster_stream_without_drift_points():
    _, ax = plots.detection_raster(runs_frame(), "calm", "warm")
    assert not _vlines(ax) and not ax.patches
    assert len(ax.get_yticklabels()) == 2


def test_raster_without_detectors_is_empty():
    runs = runs_frame()
    runs["uses_detector"] = False
    _, ax = plots.detection_raster(runs, "switch", "warm")
    assert not ax.get_yticklabels()


def test_raster_keeps_detector_that_never_fired():
    runs = runs_frame()
    runs.loc[runs.method == "ds-ts", "uses_detector"] = True  # no drift_steps
    _, ax = plots.detection_raster(runs, "switch", "warm")
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["ds-ts s0", "ds-ts s1", "rook s0", "rook s1"]
