from __future__ import annotations

import numpy as np

from contract_frames import agg_frame
from oas_experiments.analysis import style, tables


def _row(latex: str, method: str) -> list[str]:
    line = next(x for x in latex.splitlines() if x.startswith(method + " &"))
    return [cell.strip() for cell in line.removesuffix(r" \\").split("&")]


def test_main_table_frame_shape():
    frame, _ = tables.main_table(agg_frame(), "warm")
    assert list(frame.index) == [
        "ds-ts",
        "rook",
        "single",
        "unc-rate0.1",
        "unc-rate0.5",
    ]
    assert frame.columns.names == ["stream", "metric", "stat"]
    assert frame.shape[1] == 2 * 5 * 2  # streams x metrics x (mean, ci95)


def test_main_table_bolds_best_per_column():
    _, latex = tables.main_table(agg_frame(), "warm")
    assert latex.startswith(r"\begin{tabular}{l" + "r" * 10 + "}")
    assert r"\toprule" in latex and r"\bottomrule" in latex
    # Every metric grows with the method's position in METHODS: the last one
    # (single, one seed, no CI) has the best closed gap, rook the best of the rest.
    assert _row(latex, "single")[1] == r"\textbf{0.40}"
    assert _row(latex, "unc-rate0.5")[1] == r"0.31 $\pm$ 0.01"
    rook = _row(latex, "rook")
    assert not rook[1].startswith(r"\textbf")


def test_main_table_does_not_bold_costs():
    _, latex = tables.main_table(agg_frame(), "warm")
    for method in ["rook", "single", "unc-rate0.1", "unc-rate0.5"]:
        row = _row(latex, method)
        # per stream: closed gap, mean cost, solved, CPU, extra runs
        costs = row[4::5] + row[5::5]
        assert not any(cell.startswith(r"\textbf") for cell in costs), method


def test_main_table_missing_ci_and_missing_metric():
    agg = agg_frame().drop(columns="mean_recovery_ci95")
    agg.loc[agg.method == "ds-ts", "solver_cpu_mean"] = np.nan
    _, latex = tables.main_table(agg, "cold")
    single = _row(latex, "single")
    assert "$\\pm$" not in single[1]  # one seed: ci95 is NaN
    assert _row(latex, "ds-ts")[4] == "--"
    assert _row(latex, "rook")[3] == "--"  # the frame has no solved_rate


def test_main_table_escapes_names():
    agg = agg_frame()
    agg["method"] = agg["method"].str.replace("-", "_")
    _, latex = tables.main_table(agg, "warm")
    assert r"ds\_ts &" in latex


def test_detection_table_only_detector_methods():
    frame, latex = tables.detection_table(agg_frame(), "warm")
    assert list(frame.index) == ["rook"]
    assert _row(latex, "rook")[1] == r"\textbf{5.5 $\pm$ 1.0}"
    assert "False alarms" in latex


def test_detection_table_empty_without_detectors():
    agg = agg_frame()
    agg["uses_detector"] = False
    frame, _ = tables.detection_table(agg, "warm")
    assert frame.empty


def test_single_stream_has_no_stream_header():
    agg = agg_frame()
    _, latex = tables.main_table(agg[agg.stream == "switch"], "warm")
    assert r"\multicolumn" not in latex
    assert latex.splitlines()[2].startswith("Method & Closed gap")


def test_number_formats():
    assert tables.number(12 * 3600, "hours") == "12"
    assert tables.number(1800, "hours") == "0.5"
    assert tables.number(0, "hours") == "0"
    assert tables.number(0.483, "pct") == "48"
    assert tables.number(2.0, "count") == "2"
    assert tables.number(1.5, "count") == "1.5"
    assert tables.number(0.1234, "f2") == "0.12"
    assert tables.number(np.nan, "f2") == "--"


def test_pinned_order_comes_before_names(monkeypatch):
    monkeypatch.setattr(style, "_order", ())
    agg = agg_frame()
    style.pin_order(["unc", "single"])
    frame, _ = tables.main_table(agg, "warm")
    assert list(frame.index) == [
        "unc-rate0.1",
        "unc-rate0.5",
        "single",
        "ds-ts",
        "rook",
    ]


def test_main_table_for_quality_data():
    _, latex = tables.main_table(agg_frame(), "warm", maximize=True)
    assert "Mean fitness" in latex and "Mean cost" not in latex
    assert "Solved" not in latex and "CPU" not in latex
    _, latex = tables.cost_table(agg_frame(), "warm", maximize=True)
    assert "Extra CPU" not in latex and "Runs/instance" in latex
