from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from conftest import FIXTURE_RESULTS  # noqa: E402
from contract_frames import agg_frame, curves_frame, runs_frame  # noqa: E402
from oas_experiments import figures  # noqa: E402


def _frames():
    return figures.Frames(
        runs=runs_frame(),
        agg=agg_frame(),
        regret=curves_frame(),
        extra_runs=curves_frame(),
    )


def test_write_all(tmp_path):
    written = figures.write_all(_frames(), tmp_path)
    names = sorted(p.name for p in written)
    pairs = [f"{s}_{p}" for s in ("calm", "switch") for p in ("cold", "warm")]
    expected = sorted(
        [
            f"{kind}_{pair}.pdf"
            for kind in ("regret", "extra_runs", "raster")
            for pair in pairs
        ]
        + ["front.pdf", "front_overhead.pdf"]
        + [
            f"table_{t}_{p}.tex"
            for t in ("main", "detection")
            for p in ("cold", "warm")
        ]
    )
    assert names == expected
    assert sorted(p.name for p in tmp_path.iterdir()) == expected
    assert (tmp_path / "front.pdf").read_bytes().startswith(b"%PDF")


def test_write_all_only_and_no_detectors(tmp_path):
    frames = _frames()
    frames.runs["n_drift_signals"] = 0
    frames.agg["n_drift_signals_mean"] = 0.0
    assert not figures.write_all(frames, tmp_path, only="raster")
    assert not figures.write_all(frames, tmp_path, only="table_detection")
    written = figures.write_all(frames, tmp_path, only="front")
    assert [p.name for p in written] == ["front.pdf"]


def test_main_uses_d2_glue(tmp_path, monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(figures, "d2_frames", lambda d: seen.append(d) or _frames())
    code = figures.main(
        ["results/x", "--out", str(tmp_path), "--only", "table_main", "--width", "4"]
    )
    assert code == 0 and seen == ["results/x"]
    assert capsys.readouterr().out.count("table_main_") == 2


def test_main_on_fixture_results(tmp_path):
    code = figures.main([str(FIXTURE_RESULTS / "fixture_exp"), "--out", str(tmp_path)])
    names = sorted(p.name for p in tmp_path.iterdir())
    assert code == 0
    assert "front.pdf" in names and "table_main_warm10.tex" in names
    assert "raster_sudden_cold.pdf" in names  # rook signals in the fixtures
    assert "table_detection_cold.tex" in names
