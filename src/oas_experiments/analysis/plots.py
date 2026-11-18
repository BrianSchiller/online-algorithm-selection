"""Thesis figures from the D2 frames (run table, aggregate, curves).

Every function takes DataFrames and an optional ``ax``, returns
``(figure, axes)`` and never reads or writes files. Colours are assigned over
all families in the frame passed in, so pass the frames of the whole
experiment, not a filtered subset, to keep colours equal across figures.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from oas_experiments.analysis.style import (
    LINESTYLES,
    WIDTH,
    family_styles,
    protocol_marker,
)


def _axes(ax: Axes | None, width: float, aspect: float = 0.5) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(width, width * aspect))
    return ax.figure, ax


def _select(frame: pd.DataFrame, stream: str, protocol: str) -> pd.DataFrame:
    return frame[(frame["stream"] == stream) & (frame["protocol"] == protocol)]


def _drift_background(ax: Axes, runs: pd.DataFrame) -> None:
    """Shade transition intervals and mark drift points of a stream/protocol.

    Taken from the first run: all runs of one stream and protocol share them.
    """
    if runs.empty:
        return
    first = runs.iloc[0]
    for start, end in first["transition_intervals"]:
        ax.axvspan(start, end, color="0.9", lw=0, zorder=0)
    for point in first["drift_points"]:
        ax.axvline(point, color="0.4", ls="--", lw=0.8, zorder=1)


def _legend(ax: Axes) -> None:
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))


def _curves(
    curves: pd.DataFrame,
    runs: pd.DataFrame,
    stream: str,
    protocol: str,
    ylabel: str,
    methods: list[str] | None,
    ax: Axes | None,
    width: float,
) -> tuple[Figure, Axes]:
    fig, ax = _axes(ax, width)
    data = _select(curves, stream, protocol)
    if methods is not None:
        data = data[data["method"].isin(methods)]
    _drift_background(ax, _select(runs, stream, protocol))
    styles = family_styles(runs["method_family"])
    for family, members in data.groupby("method_family", sort=True):
        style = styles[family]
        for i, (method, line) in enumerate(members.groupby("method", sort=True)):
            line = line.sort_values("step")
            ax.plot(
                line["step"],
                line["mean"],
                color=style["color"],
                ls=LINESTYLES[(style["dash"] + i) % len(LINESTYLES)],
                label=method,
            )
            ax.fill_between(
                line["step"],
                line["lo"],
                line["hi"],
                color=style["color"],
                alpha=0.2,
                lw=0,
            )
    ax.set(xlabel="Step", ylabel=ylabel, title=f"{stream}, {protocol}")
    _legend(ax)
    return fig, ax


def regret_curves(
    curves: pd.DataFrame,
    runs: pd.DataFrame,
    stream: str,
    protocol: str,
    *,
    methods: list[str] | None = None,
    ax: Axes | None = None,
    width: float = WIDTH,
) -> tuple[Figure, Axes]:
    """Rolling regret over time, one line with a CI band per method.

    Parameters
    ----------
    curves
        Aggregated regret curves: stream, protocol, method, method_family,
        step, mean, lo, hi.
    runs
        Run table; gives drift points, transition intervals and the families.
    stream, protocol
        The panel to draw.
    methods
        Draw only these methods (default: all).
    ax
        Axes to draw into; a new figure if omitted.
    width
        Figure width in inches if a new figure is created.
    """
    return _curves(curves, runs, stream, protocol, "Rolling regret", methods, ax, width)


def extra_run_rate(
    curves: pd.DataFrame,
    runs: pd.DataFrame,
    stream: str,
    protocol: str,
    *,
    methods: list[str] | None = None,
    ax: Axes | None = None,
    width: float = WIDTH,
) -> tuple[Figure, Axes]:
    """Extra-run rate over time; same layout and parameters as
    :func:`regret_curves`, with the extra-run-rate curves as ``curves``."""
    return _curves(curves, runs, stream, protocol, "Extra-run rate", methods, ax, width)


def cost_quality_front(
    agg: pd.DataFrame,
    stream: str,
    protocol: str,
    *,
    overhead: bool = False,
    ax: Axes | None = None,
    width: float = WIDTH,
) -> tuple[Figure, Axes]:
    """Closed gap against solver CPU, one line per method family.

    The points of a family (its sweep) are connected in order of x; a family
    with one point is a single marker. Error bars are the ci95 in both
    directions and are left out where it is missing. Points without CPU time
    (maximisation scenarios) are dropped with a warning.

    Parameters
    ----------
    agg
        Aggregate over seeds with ``solver_cpu_*``, ``closed_gap_*`` and, for
        ``overhead``, ``overhead_seconds_*`` columns.
    stream, protocol
        The panel to draw.
    overhead
        Add selector and detector time (``overhead_seconds``) to x.
    ax
        Axes to draw into; a new figure if omitted.
    width
        Figure width in inches if a new figure is created.
    """
    fig, ax = _axes(ax, width, aspect=0.6)
    data = _select(agg, stream, protocol)
    x, xerr = data["solver_cpu_mean"], data["solver_cpu_ci95"]
    if overhead:
        x = x + data["overhead_seconds_mean"]
        # shortcut: treats CPU and overhead as independent; fine for error bars.
        xerr = np.hypot(xerr, data["overhead_seconds_ci95"])
    data = data.assign(x=x, xerr=xerr.fillna(0), yerr=data["closed_gap_ci95"].fillna(0))
    missing = data["x"].isna()
    if missing.any():
        warnings.warn(
            f"{stream}, {protocol}: no CPU time for {sorted(data['method'][missing])}; "
            "left out of the front.",
            stacklevel=2,
        )
        data = data[~missing]

    styles = family_styles(agg["method_family"])
    for family, points in data.groupby("method_family", sort=True):
        points = points.sort_values("x")
        style = styles[family]
        ax.errorbar(
            points["x"],
            points["closed_gap_mean"],
            xerr=points["xerr"],
            yerr=points["yerr"],
            fmt=protocol_marker(points["protocol_type"].iloc[0]),
            ls=LINESTYLES[style["dash"]] if len(points) > 1 else "none",
            color=style["color"],
            mfc=style["color"] if style["fill"] == "full" else "none",
            elinewidth=0.6,
            capsize=1.5,
            label=family,
        )
    if data.empty:
        ax.text(0.5, 0.5, "no CPU data", transform=ax.transAxes, ha="center")
    xlabel = "Solver CPU + overhead (s)" if overhead else "Solver CPU (s)"
    ax.set(xlabel=xlabel, ylabel="Closed gap", title=f"{stream}, {protocol}")
    _legend(ax)
    return fig, ax


def front_grid(
    agg: pd.DataFrame, *, overhead: bool = False, width: float = WIDTH
) -> tuple[Figure, np.ndarray]:
    """:func:`cost_quality_front` for every stream (columns) and protocol
    (rows), with one shared legend. Returns the figure and the axes grid."""
    streams = sorted(agg["stream"].unique())
    protocols = sorted(agg["protocol"].unique())
    fig, axes = plt.subplots(
        len(protocols),
        len(streams),
        figsize=(width, 0.45 * width * len(protocols)),
        sharey=True,
        squeeze=False,
    )
    handles = {}
    for row, protocol in zip(axes, protocols, strict=True):
        for ax, stream in zip(row, streams, strict=True):
            if _select(agg, stream, protocol).empty:
                ax.set_visible(False)
                continue
            cost_quality_front(agg, stream, protocol, overhead=overhead, ax=ax)
            if ax.get_legend():
                ax.get_legend().remove()
            artists, labels = ax.get_legend_handles_labels()
            handles.update(zip(labels, artists, strict=True))
    for ax in axes[:, 1:].flat:  # y is shared, x is not
        ax.set_ylabel("")
    if handles:
        fig.legend(
            handles.values(), handles.keys(), loc="outside lower center", ncols=4
        )
    return fig, axes


def detection_raster(
    runs: pd.DataFrame,
    stream: str,
    protocol: str,
    *,
    ax: Axes | None = None,
    width: float = WIDTH,
) -> tuple[Figure, Axes]:
    """Detector signals per method and seed against the true drift.

    One row per run of a method that signalled drift in at least one run of
    this stream and protocol; marks at its ``drift_steps``, drawn over the
    drift points and transition intervals.

    Parameters
    ----------
    runs
        Run table with drift_steps, drift_points, transition_intervals,
        n_drift_signals, method, method_family, seed.
    stream, protocol
        The panel to draw.
    ax
        Axes to draw into; a new figure if omitted.
    width
        Figure width in inches if a new figure is created.
    """
    selected = _select(runs, stream, protocol)
    signals = selected.groupby("method")["n_drift_signals"].max()
    data = selected[selected["method"].isin(signals.index[signals > 0])]
    data = data.sort_values(["method_family", "method", "seed"])
    fig, ax = _axes(ax, width, aspect=0.12 + 0.04 * len(data))
    _drift_background(ax, selected)
    styles = family_styles(runs["method_family"])
    rows = range(len(data))
    if len(data):
        ax.eventplot(
            [list(steps) for steps in data["drift_steps"]],
            lineoffsets=list(rows),
            linelengths=0.7,
            linewidths=1.0,
            colors=[styles[f]["color"] for f in data["method_family"]],
        )
    ax.set_yticks(
        list(rows),
        [f"{m} s{s}" for m, s in zip(data["method"], data["seed"], strict=True)],
    )
    if len(data):
        ax.set_ylim(len(data) - 0.5, -0.5)
    ax.set_xlim(left=0)
    ax.grid(False, axis="y")
    ax.set(xlabel="Step", title=f"{stream}, {protocol}")
    return fig, ax
