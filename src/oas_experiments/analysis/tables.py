"""Thesis tables (LaTeX, booktabs) from the D2 aggregate over seeds.

Each table function returns ``(frame, latex)``: the numbers as a DataFrame
(index: method; columns: stream, metric, ``mean``/``ci95``) and a
``tabular`` environment to ``\\input`` inside a ``table``. Cells read
"mean ± ci95"; the best mean per quality column is bold (costs such as CPU
and extra runs are traded against quality, so they have no "best").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# metric -> (column header, whether higher is better; None = no bold)
MAIN_METRICS = {
    "closed_gap": ("Closed gap", True),
    "solver_cpu": ("Solver CPU", None),
    "extra_run_rate": ("Extra runs", None),
    "mean_recovery": ("Recovery", False),
    # Recovery averages only recovered drift points, so show the others too.
    "n_not_recovered": ("Not recovered", False),
}
DETECTION_METRICS = {
    "mean_delay": ("Delay", False),
    "n_missed": ("Missed", False),
    "n_false_alarms": ("False alarms", False),
}


def main_table(
    agg: pd.DataFrame, protocol: str, digits: int = 2
) -> tuple[pd.DataFrame, str]:
    """Closed gap, solver CPU, extra-run rate, recovery and drift points not
    recovered from, per method and stream.

    Parameters
    ----------
    agg
        Aggregate over seeds (``<metric>_mean``, ``<metric>_ci95`` per stream,
        protocol and method).
    protocol
        The protocol to tabulate.
    digits
        Decimal places.
    """
    return _table(agg[agg["protocol"] == protocol], MAIN_METRICS, digits)


def detection_table(
    agg: pd.DataFrame, protocol: str, digits: int = 2
) -> tuple[pd.DataFrame, str]:
    """Detection delay, missed drifts and false alarms of detector methods.

    Only methods with ``uses_detector``. Parameters as in :func:`main_table`.
    """
    data = agg[(agg["protocol"] == protocol) & agg["uses_detector"]]
    return _table(data, DETECTION_METRICS, digits)


def _table(
    data: pd.DataFrame, metrics: dict[str, tuple[str, bool | None]], digits: int
) -> tuple[pd.DataFrame, str]:
    methods = data.sort_values(["method_family", "method"])["method"].unique()
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
    return frame, _latex(frame, streams, metrics, digits)


def _latex(
    frame: pd.DataFrame,
    streams: list[str],
    metrics: dict[str, tuple[str, bool | None]],
    digits: int,
) -> str:
    k = len(metrics)
    best = {}
    for stream in streams:
        for metric, (_, higher) in metrics.items():
            means = frame[(stream, metric, "mean")]
            if higher is not None:
                best[(stream, metric)] = means.max() if higher else means.min()

    lines = [
        rf"\begin{{tabular}}{{l{'r' * k * len(streams)}}}",
        r"\toprule",
        " & ".join(
            ["Method"] + [rf"\multicolumn{{{k}}}{{c}}{{{_escape(s)}}}" for s in streams]
        )
        + r" \\",
        " ".join(
            rf"\cmidrule(lr){{{2 + i * k}-{1 + (i + 1) * k}}}"
            for i in range(len(streams))
        ),
        " & ".join([""] + [label for _ in streams for label, _ in metrics.values()])
        + r" \\",
        r"\midrule",
    ]
    for method, row in frame.iterrows():
        cells = [_escape(method)]
        for stream in streams:
            for metric in metrics:
                mean = row[(stream, metric, "mean")]
                cells.append(
                    _cell(
                        mean,
                        row[(stream, metric, "ci95")],
                        digits,
                        bold=mean == best.get((stream, metric)),
                    )
                )
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _cell(mean: float, ci: float, digits: int, bold: bool) -> str:
    if pd.isna(mean):
        return "--"
    text = f"{mean:.{digits}f}"
    if not pd.isna(ci):
        text += rf" $\pm$ {ci:.{digits}f}"
    return rf"\textbf{{{text}}}" if bold else text


def _escape(name: str) -> str:
    # Config names are letters, digits and single '.', '_' or '-'.
    return name.replace("_", r"\_")
