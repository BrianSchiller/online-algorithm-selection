from __future__ import annotations

import re
import shutil

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from conftest import FIXTURE_RESULTS  # noqa: E402
from oas_experiments import report  # noqa: E402
from oas_experiments.analysis import metrics, plots, style, tables  # noqa: E402

EXP = FIXTURE_RESULTS / "fixture_exp"


@pytest.fixture(scope="module")
def ctx():
    return report.load(EXP)


def test_main_writes_every_section_and_figure(tmp_path):
    assert report.main([str(EXP), "--out", str(tmp_path), "--no-pdf"]) == 0
    tex = (tmp_path / "report.tex").read_text(encoding="utf-8")
    for name in report.SECTIONS:
        assert f"\\input{{sections/{name}}}" in tex
        assert (tmp_path / "sections" / f"{name}.tex").exists()
    body = "".join(
        p.read_text(encoding="utf-8") for p in (tmp_path / "sections").iterdir()
    )
    included = re.findall(r"\\includegraphics\[[^]]*\]\{([^}]*)\}", body)
    assert included and all((tmp_path / f).exists() for f in included)
    assert "raster_sudden_cold.pdf" in " ".join(included)  # rook has a detector


def test_per_protocol_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(style, "_pinned", frozenset())  # main pins; undo after
    code = report.main([str(EXP), "--out", str(tmp_path), "--no-pdf", "--per-protocol"])
    assert code == 0
    for protocol in ("cold", "warm10"):
        tex = (tmp_path / protocol / "report.tex").read_text(encoding="utf-8")
        assert f"protocol {protocol}" in tex
        figures = {p.name for p in (tmp_path / protocol / "figures").iterdir()}
        other = "warm10" if protocol == "cold" else "cold"
        assert figures and not any(f"_{other}" in name for name in figures)
    # static runs only warm, yet ds-ts keeps its colour in both protocols
    assert style.family_styles(["ds-ts"]) == style.family_styles(
        ["ds-ts", "rook", "static"]
    )


def test_restrict_keeps_only_one_protocol(ctx):
    cold = report.restrict(ctx, "cold")
    assert set(cold.runs["protocol"]) == {"cold"}
    assert set(cold.steps["run_id"]) == set(cold.runs["run_id"])
    assert set(cold.frames.regret["protocol"]) == {"cold"}
    assert "static" not in set(cold.agg["method"])  # static is warm only


def test_write_report_rejects_unknown_section(ctx, tmp_path):
    with pytest.raises(ValueError, match="unknown sections"):
        report.write_report(ctx, tmp_path, ["overview", "nope"])


def test_segment_labels():
    drift = {"builder": "grouped", "params": {"schedule": ["A", ["A", "B"], "B"]}}
    assert metrics.segment_labels(drift, 3) == ["A", "A→B", "B"]
    assert metrics.segment_labels(drift, 2) == ["A→B", "B"]  # warm start ate "A"
    assert metrics.segment_labels(drift, 4) == ["S1", "S2", "S3", "S4"]
    assert metrics.segment_labels(None, 2) == ["S1", "S2"]
    assert metrics.segment_labels(float("nan"), 1) == ["S1"]


def _toy():
    runs = pd.DataFrame(
        {
            "run_id": ["r"],
            "drift_points": [[2]],
            "n_evaluated": [4],
            "maximize": [False],
            "stream_drift": [None],
        }
    )
    steps = pd.DataFrame(
        {
            "run_id": "r",
            "step": range(4),
            "user_cost": [1.0, 3.0, 10.0, 10.0],
            "vbs_cost": [1.0, 1.0, 2.0, 2.0],
            "sbs_cost": [3.0, 3.0, 10.0, 6.0],
            "n_runs": [1, 2, 1, 1],
        }
    )
    return runs, steps


def test_segment_metrics_closed_gap():
    runs, steps = _toy()
    seg = metrics.segment_metrics(steps, runs).set_index("segment")
    assert list(seg["label"]) == ["S1", "S2"]
    assert list(seg["length"]) == [2, 2]
    # segment 0: user 2, sbs 3, vbs 1 -> 0.5; segment 1: user 10, sbs 8, vbs 2
    assert seg.at[0, "closed_gap"] == pytest.approx(0.5)
    assert seg.at[1, "closed_gap"] == pytest.approx(-2 / 6)
    assert seg.at[0, "extra_run_rate"] == 0.5 and seg.at[1, "mean_runs"] == 1.0


def test_run_count_shares_sum_to_one(ctx):
    shares = plots.run_count_shares(ctx.steps, ctx.runs, "sudden", "cold", max_runs=2)
    assert list(shares.columns) == [1, 2]
    assert np.allclose(shares.sum(axis=1), 1.0)
    assert shares.loc["rook", 2] == 1.0  # rook always runs two


def test_total_runs_counts_extra_runs(ctx):
    rook = ctx.table[ctx.table["method"] == "rook"]
    assert (rook["total_runs"] == 2 * rook["n_evaluated"]).all()


def test_pivot_table_bolds_best():
    agg = pd.DataFrame(
        {
            "method": ["a", "b", "a", "b"],
            "method_family": ["a", "b", "a", "b"],
            "segment": [0, 0, 1, 1],
            "closed_gap_mean": [0.1, 0.4, 0.3, 0.2],
            "closed_gap_ci95": [np.nan, 0.05, np.nan, np.nan],
        }
    )
    frame, latex = tables.pivot_table(
        agg, "segment", "closed_gap", headers={0: "X"}, higher=True
    )
    assert frame.loc["b", (0, "mean")] == 0.4
    assert r"\textbf{0.40 $\pm$ 0.05}" in latex and r"\textbf{0.30}" in latex
    assert "Method & X & 1" in latex


def test_escape():
    assert tables.escape("a_b&c%d#e{f}~^$") == (
        r"a\_b\&c\%d\#e\{f\}\textasciitilde{}\textasciicircum{}\$"
    )
    assert tables.escape("A→B") == r"A$\to$B"


@pytest.mark.skipif(shutil.which("latexmk") is None, reason="needs latexmk")
def test_compile_pdf(ctx, tmp_path):
    tex = report.write_report(ctx, tmp_path, ["overview", "methods", "appendix"])
    pdf = report.compile_pdf(tex)
    assert pdf is not None and pdf.read_bytes().startswith(b"%PDF")
