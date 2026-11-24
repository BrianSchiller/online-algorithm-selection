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
from matplotlib.colors import BoundaryNorm
from matplotlib.figure import Figure

from oas_experiments.analysis.style import (
    LINESTYLES,
    WIDTH,
    family_rank,
    family_styles,
    method_order,
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
        # above the grid (zorder 2), which would hide points on a tick
        ax.axvline(point, color="0.4", ls="--", lw=0.8, zorder=2.1)


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
    bounds: tuple[float | None, float | None] = (None, None),
) -> tuple[Figure, Axes]:
    """Mean line and CI band per method; the band is clipped to ``bounds``
    (the range the value can take, e.g. regret >= 0)."""
    fig, ax = _axes(ax, width)
    data = _select(curves, stream, protocol)
    if methods is not None:
        data = data[data["method"].isin(methods)]
    _drift_background(ax, _select(runs, stream, protocol))
    styles = family_styles(runs["method_family"])
    families = sorted(data["method_family"].unique(), key=family_rank)
    for family in families:
        members = data[data["method_family"] == family]
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
                line["lo"].clip(*bounds),
                line["hi"].clip(*bounds),
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
    return _curves(
        curves, runs, stream, protocol, "Rolling regret", methods, ax, width, (0, None)
    )


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
    return _curves(
        curves, runs, stream, protocol, "Extra-run rate", methods, ax, width, (0, 1)
    )


def cost_quality_front(
    agg: pd.DataFrame,
    stream: str,
    protocol: str,
    *,
    overhead: bool = False,
    hours: bool = False,
    by_runs: bool = False,
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
    hours
        CPU time in hours instead of seconds.
    by_runs
        Solver runs per instance (``mean_runs_per_instance``) instead of CPU
        time on x, for quality data without runtimes; ignores ``overhead``.
    ax
        Axes to draw into; a new figure if omitted.
    width
        Figure width in inches if a new figure is created.
    """
    fig, ax = _axes(ax, width, aspect=0.6)
    data = _select(agg, stream, protocol)
    cost = "mean_runs_per_instance" if by_runs else "solver_cpu"
    x, xerr = data[f"{cost}_mean"], data[f"{cost}_ci95"]
    if by_runs:
        overhead = hours = False
    if overhead:
        x = x + data["overhead_seconds_mean"]
        # shortcut: treats CPU and overhead as independent; fine for error bars.
        xerr = np.hypot(xerr, data["overhead_seconds_ci95"])
    if hours:
        x, xerr = x / 3600, xerr / 3600
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
    for family in sorted(data["method_family"].unique(), key=family_rank):
        points = data[data["method_family"] == family].sort_values("x")
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
    unit = "h" if hours else "s"
    xlabel = f"Solver CPU + overhead ({unit})" if overhead else f"Solver CPU ({unit})"
    xlabel = "Solver runs per instance" if by_runs else xlabel
    ax.set(xlabel=xlabel, ylabel="Closed gap", title=f"{stream}, {protocol}")
    _legend(ax)
    return fig, ax


def front_grid(
    agg: pd.DataFrame,
    *,
    overhead: bool = False,
    hours: bool = False,
    by_runs: bool = False,
    width: float = WIDTH,
    aspect: float = 0.45,
) -> tuple[Figure, np.ndarray]:
    """:func:`cost_quality_front` for every stream (columns) and protocol
    (rows), with one shared legend. Returns the figure and the axes grid;
    ``aspect`` is the height of one row of panels over the width."""
    streams = sorted(agg["stream"].unique())
    protocols = sorted(agg["protocol"].unique())
    fig, axes = plt.subplots(
        len(protocols),
        len(streams),
        figsize=(width, aspect * width * len(protocols)),
        sharey=True,
        squeeze=False,
    )
    handles = {}
    for row, protocol in zip(axes, protocols, strict=True):
        for ax, stream in zip(row, streams, strict=True):
            if _select(agg, stream, protocol).empty:
                ax.set_visible(False)
                continue
            cost_quality_front(
                agg,
                stream,
                protocol,
                overhead=overhead,
                hours=hours,
                by_runs=by_runs,
                ax=ax,
            )
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

    One row per run of a method with a drift detector (``uses_detector``),
    also if it never fired; marks at its ``drift_steps``, drawn over the
    drift points and transition intervals.

    Parameters
    ----------
    runs
        Run table with drift_steps, drift_points, transition_intervals,
        uses_detector, method, method_family, seed.
    stream, protocol
        The panel to draw.
    ax
        Axes to draw into; a new figure if omitted.
    width
        Figure width in inches if a new figure is created.
    """
    selected = _select(runs, stream, protocol)
    data = selected[selected["uses_detector"]]
    data = data.assign(rank=data["method_family"].map(family_rank)).sort_values(
        ["rank", "method", "seed"]
    )
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
    # the whole evaluated stream, not just up to the last signal
    end = selected["n_evaluated"].max() if "n_evaluated" in selected else None
    ax.set_xlim(0, None if pd.isna(end) else end)
    ax.grid(False, axis="y")
    ax.set(xlabel="Step", title=f"{stream}, {protocol}")
    return fig, ax


# --- report figures -------------------------------------------------------

SEQUENTIAL = "Blues"  # one hue, light to dark: shares and counts
# Colour bins of share heatmaps: most shares are small, so the bins are fine
# below 0.1 and coarse above; everything above 0.5 gets the darkest colour.
SHARE_BINS = (0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 1)
SHARE_TICKS = (0, 0.05, 0.1, 0.2, 0.5, 1)  # labelled on the colour bar


def cumulative_regret(
    curves: pd.DataFrame,
    runs: pd.DataFrame,
    stream: str,
    protocol: str,
    *,
    methods: list[str] | None = None,
    ax: Axes | None = None,
    width: float = WIDTH,
) -> tuple[Figure, Axes]:
    """Cumulative regret over time; same layout and parameters as
    :func:`regret_curves`, with cumulative-regret curves as ``curves``."""
    return _curves(
        curves,
        runs,
        stream,
        protocol,
        "Cumulative regret",
        methods,
        ax,
        width,
        (0, None),
    )


def _stream_steps(
    steps: pd.DataFrame, runs: pd.DataFrame, stream: str, protocol: str
) -> pd.DataFrame:
    """Steps of one stream and protocol with ``method``, ``method_family`` and
    ``seed`` joined on."""
    selected = _select(runs, stream, protocol)[
        ["run_id", "method", "method_family", "seed"]
    ]
    return steps.merge(selected, on="run_id")


def first_run(
    steps: pd.DataFrame, runs: pd.DataFrame, stream: str, protocol: str
) -> tuple[pd.Series, pd.DataFrame]:
    """One run and its steps; for columns that do not depend on the selector
    (VBS, SBS), which all runs of a stream and protocol share."""
    run = _select(runs, stream, protocol).sort_values("run_id").iloc[0]
    return run, steps[steps["run_id"] == run["run_id"]].sort_values("step")


def run_count_shares(
    steps: pd.DataFrame,
    runs: pd.DataFrame,
    stream: str,
    protocol: str,
    max_runs: int = 4,
) -> pd.DataFrame:
    """Share of instances with 1, 2, ... solver runs per method (rows sum to 1).

    Counts above ``max_runs`` go into the column ``max_runs``; seeds are pooled.
    """
    data = _stream_steps(steps, runs, stream, protocol)
    counts = data["n_runs"].clip(upper=max_runs)
    shares = pd.crosstab(data["method"], counts, normalize="index")
    shares = shares.reindex(index=method_order(data))
    return shares.reindex(columns=range(1, max_runs + 1), fill_value=0.0)


def runs_per_instance(
    steps: pd.DataFrame,
    runs: pd.DataFrame,
    stream: str,
    protocol: str,
    *,
    max_runs: int = 4,
    ax: Axes | None = None,
    width: float = WIDTH,
) -> tuple[Figure, Axes]:
    """Stacked bar per method: share of instances with 1, 2, ..., >= ``max_runs``
    solver runs (see :func:`run_count_shares`)."""
    shares = run_count_shares(steps, runs, stream, protocol, max_runs)
    fig, ax = _axes(ax, width, aspect=0.12 + 0.05 * len(shares))
    colors = plt.get_cmap(SEQUENTIAL)(np.linspace(0.3, 0.9, max_runs))
    left = np.zeros(len(shares))
    for k, color in zip(shares.columns, colors, strict=True):
        label = f"$\\geq${k}" if k == max_runs else str(k)
        ax.barh(
            shares.index,
            shares[k],
            left=left,
            color=color,
            label=label,
            edgecolor="white",
            linewidth=0.5,
        )
        left += shares[k].to_numpy()
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.grid(False, axis="y")
    ax.set(
        xlabel="Share of instances",
        title=f"{stream}, {protocol}: solver runs per instance",
    )
    ax.legend(title="Runs", loc="upper left", bbox_to_anchor=(1.01, 1))
    return fig, ax


def _heatmap(ax: Axes, frame: pd.DataFrame, label: str) -> None:
    """Shares in [0, 1] as cells, every non-zero share printed (``.04``), so a
    row can be read to its sum of 1. Colours come in :data:`SHARE_BINS`: fine
    below 0.1, one colour for 0.4-0.5 and the darkest for everything above."""
    cmap = plt.get_cmap(SEQUENTIAL, len(SHARE_BINS) - 1)
    norm = BoundaryNorm(SHARE_BINS, cmap.N)
    image = ax.imshow(frame.to_numpy(float), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(frame.shape[1]), frame.columns, rotation=60, ha="right")
    ax.set_yticks(range(frame.shape[0]), frame.index)
    ax.grid(False)
    for (i, j), value in np.ndenumerate(frame.to_numpy(float)):
        if value >= 0.005:  # prints as at least .01
            ax.text(
                j,
                i,
                f"{value:.2f}".removeprefix("0"),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value >= 0.3 else "0.15",
            )
    # uniform spacing: every bin gets the same height, so the fine low bins show
    bar = ax.figure.colorbar(image, ax=ax, label=label, shrink=0.8, spacing="uniform")
    bar.set_ticks(SHARE_TICKS, labels=[f"{b:g}" for b in SHARE_TICKS])
    bar.minorticks_off()


def _top_columns(
    frame: pd.DataFrame, top: int, rank: pd.Series | None = None
) -> pd.DataFrame:
    """Keep the ``top`` columns with the largest share; sum the rest as other.

    The kept columns are ordered by ``rank`` (descending; e.g. how often an
    algorithm is the VBS, so heatmaps of one stream share the column order),
    columns without a rank after them by their largest share."""
    order = frame.max().sort_values(ascending=False).index
    kept = list(order[:top])
    if rank is not None:
        kept.sort(key=lambda c: (-rank.get(c, -1.0), -frame[c].max()))
    result = frame[kept]
    if len(order) > top:
        result = result.assign(other=frame[order[top:]].sum(axis=1))
    return result


def selection_heatmap(
    steps: pd.DataFrame,
    runs: pd.DataFrame,
    stream: str,
    protocol: str,
    *,
    top: int = 12,
    ax: Axes | None = None,
    width: float = WIDTH,
) -> tuple[Figure, Axes]:
    """Share of instances on which each algorithm was the primary choice, per
    method (seeds pooled), with the oracle-best algorithm (VBS) as first row.
    The ``top`` algorithms with the largest share are shown, the rest as other.
    """
    data = _stream_steps(steps, runs, stream, protocol)
    chosen = pd.crosstab(data["method"], data["primary"], normalize="index")
    chosen = chosen.reindex(index=method_order(data))
    _, first = first_run(steps, runs, stream, protocol)
    vbs = first["vbs_algorithm"].value_counts(normalize=True).rename("VBS")
    frame = pd.concat([vbs.to_frame().T, chosen]).fillna(0.0)
    frame = _top_columns(frame, top, rank=vbs)
    fig, ax = _axes(ax, width, aspect=0.15 + 0.05 * len(frame))
    _heatmap(ax, frame, "Share of instances")
    ax.set(title=f"{stream}, {protocol}: primary choice", xlabel="Algorithm")
    return fig, ax


def oracle_costs(
    steps: pd.DataFrame,
    runs: pd.DataFrame,
    stream: str,
    protocol: str,
    *,
    window: int = 50,
    ax: Axes | None = None,
    width: float = WIDTH,
) -> tuple[Figure, Axes]:
    """Rolling mean cost of the VBS and the stream SBS (log scale; fitness on a
    linear scale for quality data), with the drift points and transition
    intervals of the stream."""
    fig, ax = _axes(ax, width, aspect=0.35)
    _drift_background(ax, _select(runs, stream, protocol))
    run, first = first_run(steps, runs, stream, protocol)
    for column, label, color in (
        ("sbs_cost", "SBS", "0.45"),
        ("vbs_cost", "VBS", "#0072B2"),
    ):
        ax.plot(
            first["step"],
            first[column].rolling(window, min_periods=1).mean(),
            color=color,
            label=label,
        )
    if not run["maximize"]:
        ax.set_yscale("log")
    ax.set(
        xlabel="Step",
        ylabel=f"{'Fitness' if run['maximize'] else 'Cost'} (rolling mean, {window})",
        title=f"{stream}, {protocol}: oracle costs",
    )
    _legend(ax)
    return fig, ax


def segment_oracle_share(
    steps: pd.DataFrame,
    runs: pd.DataFrame,
    stream: str,
    protocol: str,
    *,
    top: int = 10,
    ax: Axes | None = None,
    width: float = WIDTH,
) -> tuple[Figure, Axes]:
    """Share of instances per segment on which each algorithm is the VBS: what
    changes at each drift point. The ``top`` algorithms, the rest as other."""
    from oas_experiments.analysis.metrics import segments, step_segments

    run, first = first_run(steps, runs, stream, protocol)
    run_frame = run.to_frame().T
    first = first.assign(segment=step_segments(first, run_frame))
    frame = pd.crosstab(first["segment"], first["vbs_algorithm"], normalize="index")
    seg = segments(run_frame).set_index("segment")
    frame.index = [
        f"{seg.at[i, 'label']} [{seg.at[i, 'start']}, {seg.at[i, 'end']})"
        for i in frame.index
    ]
    vbs = first["vbs_algorithm"].value_counts(normalize=True)
    frame = _top_columns(frame, top, rank=vbs)
    fig, ax = _axes(ax, width, aspect=0.15 + 0.05 * len(frame))
    _heatmap(ax, frame, "Share of instances")
    ax.set(
        title=f"{stream}, {protocol}: oracle-best algorithm per segment",
        xlabel="Algorithm",
    )
    return fig, ax


def small_multiples(
    curves: pd.DataFrame,
    runs: pd.DataFrame,
    stream: str,
    protocol: str,
    ylabel: str,
    *,
    methods: list[str] | None = None,
    ncols: int = 2,
    width: float = WIDTH,
    panel_aspect: float = 0.32,
) -> tuple[Figure, np.ndarray]:
    """One panel per method: its curve (mean and CI band) in its colour, the
    other methods in light grey behind it, drift points and transitions, and
    the drift signals of detector methods as ticks along the bottom.

    Parameters
    ----------
    curves
        Aggregated curves as in :func:`regret_curves`.
    runs
        Run table; gives drift points, transitions, signals and the families.
    stream, protocol
        The stream and protocol to draw.
    ylabel
        Label of the y axis.
    methods
        Panels for these methods (default: all, in :func:`method_order`).
    ncols
        Panels per row.
    width
        Figure width in inches.
    panel_aspect
        Height of one row of panels over the figure width.
    """
    data = _select(curves, stream, protocol)
    selected = _select(runs, stream, protocol)
    methods = methods or method_order(data)
    nrows = -(-len(methods) // ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(width, panel_aspect * width * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    styles = family_styles(runs["method_family"])
    lines = {m: line.sort_values("step") for m, line in data.groupby("method")}
    for ax, method in zip(axes.flat, methods, strict=False):
        _drift_background(ax, selected)
        for other, line in lines.items():
            if other != method:
                ax.plot(line["step"], line["mean"], color="0.85", lw=0.6)
        line = lines[method]
        family = line["method_family"].iloc[0]
        color = styles[family]["color"]
        ax.fill_between(
            line["step"],
            line["lo"].clip(0),
            line["hi"].clip(0),
            color=color,
            alpha=0.2,
            lw=0,
        )
        ax.plot(line["step"], line["mean"], color=color, lw=1.0)
        signals = selected.loc[selected["method"] == method, "drift_steps"]
        steps = [s for run in signals for s in run]
        if steps:
            ax.plot(
                steps,
                [0.04] * len(steps),
                "|",
                color="black",
                ms=5,
                transform=ax.get_xaxis_transform(),
            )
        ax.text(
            0.01,
            0.97,
            method,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"fc": "white", "ec": "none", "pad": 1, "alpha": 0.8},
        )
        ax.set_ylim(bottom=0)
    for ax in axes[-1]:
        ax.set_xlabel("Step")
    for i, ax in enumerate(axes.flat[len(methods) :], start=len(methods)):
        ax.set_visible(False)
        above = axes.flat[i - ncols]  # now the lowest panel of its column
        above.tick_params(labelbottom=True)
        above.set_xlabel("Step")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    return fig, axes
