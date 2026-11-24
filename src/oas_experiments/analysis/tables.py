"""Thesis tables (LaTeX, booktabs) from the D2 aggregate over seeds.

Each table function returns ``(frame, latex)``: the numbers as a DataFrame
(index: method; columns: stream, metric, ``mean``/``ci95``) and a
``tabular`` environment to ``\\input`` inside a ``table``. Cells read
"mean ± ci95" in the column's format; the best mean per quality column is
bold (costs such as CPU and extra runs are traded against quality, so they
have no "best").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from oas_experiments.analysis.style import method_order

# metric -> (column header, whether higher is better (None = no bold), format);
# the formats are those of :func:`number`.
MAIN_METRICS = {
    "closed_gap": ("Closed gap", True, "f2"),
    "mean_user_cost": ("Mean cost", False, "f0"),
    "solved_rate": (r"Solved (\%)", True, "pct"),
    "solver_cpu": ("CPU (h)", None, "hours"),
    "extra_run_rate": (r"Extra runs (\%)", None, "pct"),
}
# What a method spends besides the solver runs the user asked for.
COST_METRICS = {
    "mean_runs_per_instance": ("Runs/instance", None, "f2"),
    "extra_cpu": ("Extra CPU (h)", None, "hours"),
    "overhead_seconds": ("Overhead (s)", None, "f0"),
    "n_retrains": ("Retrains", None, "count"),
    "n_drift_signals": ("Drift signals", None, "count"),
}
# Quality data (maximize): the user's result is a fitness, and there is no
# cutoff or runtime, hence nothing solved and no CPU time.
QUALITY = {"mean_user_cost": ("Mean fitness", True, "f3")}
RUNTIME_ONLY = {"solved_rate", "solver_cpu", "extra_cpu"}
DETECTION_METRICS = {
    "mean_delay": ("Delay", False, "f1"),
    "n_missed": ("Missed", False, "count"),
    "n_false_alarms": ("False alarms", False, "count"),
}


def number(value: float, fmt: str) -> str:
    """A number as text: ``f0``-``f3`` fixed decimals, ``count`` an integer
    (one decimal if a mean over seeds is not whole), ``pct`` a share in
    percent, ``hours`` seconds in whole hours. NaN is ``--``."""
    if pd.isna(value):
        return "--"
    if fmt == "pct":
        return f"{100 * value:.0f}"
    if fmt == "hours":  # one decimal below 10 h
        hours = value / 3600
        return "0" if hours == 0 else f"{hours:.1f}" if hours < 10 else f"{hours:.0f}"
    if fmt == "count":
        return f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}"
    return f"{value:.{int(fmt[1:])}f}"


def main_table(
    agg: pd.DataFrame, protocol: str, maximize: bool = False
) -> tuple[pd.DataFrame, str]:
    """Closed gap, mean cost, share of solved instances, solver CPU and
    extra-run rate, per method and stream.

    Parameters
    ----------
    agg
        Aggregate over seeds (``<metric>_mean``, ``<metric>_ci95`` per stream,
        protocol and method).
    protocol
        The protocol to tabulate.
    maximize
        Quality data: mean fitness instead of mean cost, no solved or CPU columns.
    """
    return _table(agg[agg["protocol"] == protocol], _for(MAIN_METRICS, maximize))


def cost_table(
    agg: pd.DataFrame, protocol: str, maximize: bool = False
) -> tuple[pd.DataFrame, str]:
    """:data:`COST_METRICS` per method and stream; parameters as in
    :func:`main_table`. ``agg`` must aggregate these metrics."""
    return _table(agg[agg["protocol"] == protocol], _for(COST_METRICS, maximize))


def _for(metrics: Metrics, maximize: bool) -> Metrics:
    """``metrics`` as they apply to quality data if ``maximize``."""
    if not maximize:
        return metrics
    return {k: QUALITY.get(k, v) for k, v in metrics.items() if k not in RUNTIME_ONLY}


def pivot_table(
    agg: pd.DataFrame,
    column: str,
    metric: str,
    *,
    headers: dict | None = None,
    higher: bool | None = None,
    fmt: str = "f2",
) -> tuple[pd.DataFrame, str]:
    """One metric per method (rows) and value of ``column`` (columns).

    Parameters
    ----------
    agg
        Aggregate of one stream and protocol with ``method``, ``method_family``,
        ``column`` and ``<metric>_mean`` / ``<metric>_ci95``.
    column
        Column whose values become table columns, e.g. ``segment``.
    metric
        The metric in the cells ("mean ± ci95").
    headers
        LaTeX header per value of ``column`` (default: the escaped value).
    higher
        Whether higher is better; the best mean per column is bold. None: no bold.
    fmt
        Number format, see :func:`number`.
    """
    keys = sorted(agg[column].unique())
    methods = method_order(agg)
    rows = agg.set_index(["method", column])
    frame = pd.DataFrame(
        {
            (key, stat): rows[f"{metric}_{stat}"].xs(key, level=column)
            for key in keys
            for stat in ("mean", "ci95")
        },
        index=pd.Index(methods, name="method"),
    )
    headers = headers or {}
    lines = [
        rf"\begin{{tabular}}{{l{'r' * len(keys)}}}",
        r"\toprule",
        " & ".join(["Method"] + [headers.get(k, escape(str(k))) for k in keys])
        + r" \\",
        r"\midrule",
    ]
    best = {}
    if higher is not None:
        best = {
            k: frame[(k, "mean")].max() if higher else frame[(k, "mean")].min()
            for k in keys
        }
    for method, row in frame.iterrows():
        cells = [escape(method)] + [
            _cell(
                row[(k, "mean")],
                row[(k, "ci95")],
                fmt,
                bold=row[(k, "mean")] == best.get(k),
            )
            for k in keys
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return frame, "\n".join(lines) + "\n"


def detection_table(agg: pd.DataFrame, protocol: str) -> tuple[pd.DataFrame, str]:
    """Detection delay, missed drifts and false alarms of detector methods.

    Only methods with ``uses_detector``. Parameters as in :func:`main_table`.
    """
    data = agg[(agg["protocol"] == protocol) & agg["uses_detector"]]
    return _table(data, DETECTION_METRICS)


Metrics = dict[str, tuple[str, bool | None, str]]


def _table(data: pd.DataFrame, metrics: Metrics) -> tuple[pd.DataFrame, str]:
    methods = method_order(data)
    streams = sorted(data["stream"].unique())
    rows = data.set_index(["method", "stream"])
    columns = {}
    for stream in streams:
        for metric in metrics:
            for stat in ("mean", "ci95"):
                name = f"{metric}_{stat}"
                values = rows[name] if name in rows else pd.Series(np.nan, rows.index)
                columns[(stream, metric, stat)] = values.xs(stream, level="stream")
    frame = pd.DataFrame(
        columns,
        index=pd.Index(methods, name="method"),
        columns=pd.MultiIndex.from_product(
            [streams, list(metrics), ["mean", "ci95"]],
            names=["stream", "metric", "stat"],
        ),
    )
    return frame, _latex(frame, streams, metrics)


def _latex(frame: pd.DataFrame, streams: list[str], metrics: Metrics) -> str:
    """Stream names head groups of columns; with one stream they are left out."""
    k = len(metrics)
    best = {}
    for stream in streams:
        for metric, (_, higher, _) in metrics.items():
            means = frame[(stream, metric, "mean")]
            if higher is not None:
                best[(stream, metric)] = means.max() if higher else means.min()

    lines = [rf"\begin{{tabular}}{{l{'r' * k * len(streams)}}}", r"\toprule"]
    if len(streams) > 1:
        lines += [
            " & ".join(
                [""] + [rf"\multicolumn{{{k}}}{{c}}{{{_escape(s)}}}" for s in streams]
            )
            + r" \\",
            " ".join(
                rf"\cmidrule(lr){{{2 + i * k}-{1 + (i + 1) * k}}}"
                for i in range(len(streams))
            ),
        ]
    lines += [
        " & ".join(
            ["Method"] + [label for _ in streams for label, *_ in metrics.values()]
        )
        + r" \\",
        r"\midrule",
    ]
    for method, row in frame.iterrows():
        cells = [_escape(method)]
        for stream in streams:
            for metric, (_, _, fmt) in metrics.items():
                mean = row[(stream, metric, "mean")]
                cells.append(
                    _cell(
                        mean,
                        row[(stream, metric, "ci95")],
                        fmt,
                        bold=mean == best.get((stream, metric)),
                    )
                )
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _cell(mean: float, ci: float, fmt: str, bold: bool) -> str:
    if pd.isna(mean):
        return "--"
    text = number(mean, fmt)
    if not pd.isna(ci):
        text += rf" $\pm$ {number(ci, fmt)}"
    return rf"\textbf{{{text}}}" if bold else text


_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "→": r"$\to$",
}


def escape(text: str) -> str:
    """Any text as LaTeX (special characters escaped)."""
    return "".join(_SPECIAL.get(c, c) for c in str(text))


def _escape(name: str) -> str:
    # Config names are letters, digits and single '.', '_' or '-'.
    return name.replace("_", r"\_")
