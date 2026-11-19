from __future__ import annotations

import numpy as np

from contract_frames import agg_frame
from oas_experiments.analysis import tables


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
    assert not rook[1].startswith(r"\textbf") and rook[2].startswith(r"\textbf")


def test_main_table_missing_ci_and_missing_metric():
    agg = agg_frame().drop(columns="mean_recovery_ci95")
    agg.loc[agg.method == "ds-ts", "solver_cpu_mean"] = np.nan
    _, latex = tables.main_table(agg, "cold")
    single = _row(latex, "single")
    assert "$\\pm$" not in single[1]  # one seed: ci95 is NaN
    assert _row(latex, "ds-ts")[2] == "--"
    assert all("$\\pm$" not in cell for cell in _row(latex, "rook")[4::5])


def test_main_table_escapes_names():
    agg = agg_frame()
    agg["method"] = agg["method"].str.replace("-", "_")
    _, latex = tables.main_table(agg, "warm")
    assert r"ds\_ts &" in latex


def test_detection_table_only_detector_methods():
    frame, latex = tables.detection_table(agg_frame(), "warm")
    assert list(frame.index) == ["rook"]
    assert _row(latex, "rook")[1] == r"\textbf{5.50 $\pm$ 0.98}"
    assert "False alarms" in latex


def test_detection_table_empty_without_detectors():
    agg = agg_frame()
    agg["uses_detector"] = False
    frame, _ = tables.detection_table(agg, "warm")
    assert frame.empty
