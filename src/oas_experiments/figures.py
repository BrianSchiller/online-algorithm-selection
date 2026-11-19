"""Write every thesis figure (PDF) and table (.tex) of an experiment.

    python -m oas_experiments.figures results/pilot_sat12 --out figures/pilot_sat12
    python -m oas_experiments.figures results/pilot_sat12 --out figs --only front
    python -m oas_experiments.figures results/pilot_sat12 --out figs --width 4.5

Per stream and protocol: ``regret_<stream>_<protocol>.pdf``,
``extra_runs_<stream>_<protocol>.pdf`` and, if a method has a detector,
``raster_<stream>_<protocol>.pdf``. Over all streams: ``front.pdf`` and
``front_overhead.pdf``. Per protocol: ``table_main_<protocol>.tex`` and, if a
method has a detector, ``table_detection_<protocol>.tex``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from oas_experiments.analysis import metrics, plots, style, tables
from oas_experiments.results import ResultError, load_experiment

NAMES = (
    "regret",
    "extra_runs",
    "raster",
    "front",
    "front_overhead",
    "table_main",
    "table_detection",
)


class Frames(NamedTuple):
    """The D2 frames the figures are drawn from."""

    runs: pd.DataFrame  # runs of load_experiment (drift lists) plus uses_detector
    agg: pd.DataFrame  # aggregate over seeds, incl. protocol_type
    regret: pd.DataFrame  # aggregated rolling-regret curves
    extra_runs: pd.DataFrame  # aggregated rolling extra-run-rate curves


def d2_frames(experiment_dir: str | Path) -> Frames:
    """Load an experiment and compute the D2 frames (the only caller of D2)."""
    results = load_experiment(experiment_dir)
    steps = metrics.add_step_metrics(results.steps, results.runs)
    table = metrics.run_table(results.runs, results.steps)
    by = ("stream", "protocol", "protocol_type", "method")  # protocol_type: markers

    def curves(column: str) -> pd.DataFrame:
        return metrics.aggregate_curves(
            metrics.rolling_curves(steps, column), results.runs
        )

    return Frames(
        runs=results.runs.merge(table[["run_id", "uses_detector"]], on="run_id"),
        agg=metrics.aggregate(table, by=by),
        regret=curves("regret"),
        extra_runs=curves("extra_run"),
    )


def write_all(
    frames: Frames,
    out: str | Path,
    *,
    only: str | None = None,
    width: float = style.WIDTH,
) -> list[Path]:
    """Write all figures and tables (or only those named ``only``) to ``out``.

    Returns the written paths.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def want(name: str) -> bool:
        return only in (None, name)

    def save(fig: Figure, name: str) -> None:
        fig.savefig(out / name)
        plt.close(fig)
        written.append(out / name)

    runs = frames.runs
    pairs = (
        runs[["stream", "protocol"]]
        .drop_duplicates()
        .sort_values(["stream", "protocol"])
    )
    for stream, protocol in pairs.itertuples(index=False):
        tag = f"{stream}_{protocol}"
        if want("regret"):
            fig, _ = plots.regret_curves(
                frames.regret, runs, stream, protocol, width=width
            )
            save(fig, f"regret_{tag}.pdf")
        if want("extra_runs"):
            fig, _ = plots.extra_run_rate(
                frames.extra_runs, runs, stream, protocol, width=width
            )
            save(fig, f"extra_runs_{tag}.pdf")
        selected = runs[(runs["stream"] == stream) & (runs["protocol"] == protocol)]
        if want("raster") and selected["uses_detector"].any():
            fig, _ = plots.detection_raster(runs, stream, protocol, width=width)
            save(fig, f"raster_{tag}.pdf")

    if want("front"):
        save(plots.front_grid(frames.agg, width=width)[0], "front.pdf")
    if want("front_overhead"):
        fig, _ = plots.front_grid(frames.agg, overhead=True, width=width)
        save(fig, "front_overhead.pdf")

    for protocol in sorted(runs["protocol"].unique()):
        if want("table_main"):
            _, latex = tables.main_table(frames.agg, protocol)
            (out / f"table_main_{protocol}.tex").write_text(latex, encoding="utf-8")
            written.append(out / f"table_main_{protocol}.tex")
        detection, latex = tables.detection_table(frames.agg, protocol)
        if want("table_detection") and not detection.empty:
            (out / f"table_detection_{protocol}.tex").write_text(
                latex, encoding="utf-8"
            )
            written.append(out / f"table_detection_{protocol}.tex")
    return written


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m oas_experiments.figures",
        description="Write all figures and tables of an experiment.",
    )
    parser.add_argument("experiment_dir", help="results/<experiment>")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--only", choices=NAMES, help="write only this figure or table")
    parser.add_argument(
        "--width",
        type=float,
        default=style.WIDTH,
        help=f"figure width in inches (default {style.WIDTH})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mpl.use("Agg")
    style.apply()
    try:
        frames = d2_frames(args.experiment_dir)
    except ResultError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for path in write_all(frames, args.out, only=args.only, width=args.width):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
