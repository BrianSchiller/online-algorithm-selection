"""A LaTeX report (and PDF) of one experiment: stream, methods, performance,
drift adaptation, statistics and run details.

    python -m oas_experiments.report results/test_sat12 --out reports/test_sat12
    python -m oas_experiments.report results/x --out r/x --sections overview,stream
    python -m oas_experiments.report results/x --out reports/x --no-pdf

Writes ``report.tex``, one ``sections/<name>.tex`` per section (each can be
``\\input`` on its own), ``figures/*.pdf`` and, with latexmk installed,
``report.pdf``.

A section is a function ``(ctx: Context, out: Out) -> str`` that returns
LaTeX and saves its figures through ``out``. :data:`SECTIONS` lists them in
report order; a report for another scenario passes its own list of names to
:func:`write_report`, or adds a function to :data:`SECTIONS`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from oas_experiments import figures
from oas_experiments.analysis import metrics, plots, stats, style, tables
from oas_experiments.analysis.tables import escape
from oas_experiments.results import ResultError, load_experiment

BY = ("stream", "protocol", "protocol_type", "method")
COST_METRICS = ["mean_runs_per_instance", "n_retrains", "total_runs"]
DRIFT_METRICS = ["recovery", "peak_regret", "extra_run_rate_change"]
CONFIG_KEYS = ["run_id", "stream", "protocol", "method", "method_family"]


class Context(NamedTuple):
    """Everything a section draws from (all frames of the whole experiment)."""

    runs: pd.DataFrame  # load_experiment runs plus uses_detector
    steps: pd.DataFrame  # steps with the step metrics of add_step_metrics
    frames: figures.Frames  # the D2 frames of the figures CLI
    table: pd.DataFrame  # run table plus COST_METRICS
    agg: pd.DataFrame  # aggregate of ``table`` over seeds
    segments: pd.DataFrame  # segment metrics aggregated over seeds
    drift: pd.DataFrame  # drift-point metrics aggregated over seeds
    cum_regret: pd.DataFrame  # aggregated cumulative-regret curves


def load(experiment_dir: str | Path) -> Context:
    """Load an experiment and compute every frame of the report."""
    results = load_experiment(experiment_dir)
    runs = results.runs
    frames = figures.frames_from(results)
    steps = metrics.add_step_metrics(results.steps, runs)
    table = metrics.run_table(runs, results.steps).merge(
        runs[["run_id", "mean_runs_per_instance", "n_retrains"]], on="run_id"
    )
    # every solver run, extra runs included
    table["total_runs"] = table["mean_runs_per_instance"] * table["n_evaluated"]
    keys = table[[*CONFIG_KEYS, "protocol_type", "uses_detector"]]

    segment = metrics.segment_metrics(steps, runs).merge(keys, on="run_id")
    drift = metrics.drift_metrics(steps, runs).merge(keys, on="run_id")
    drift["extra_run_rate_change"] = (
        drift["extra_run_rate_after"] - drift["extra_run_rate_before"]
    )
    return Context(
        runs=frames.runs,
        steps=steps,
        frames=frames,
        table=table,
        agg=metrics.aggregate(table, by=BY, metrics=metrics.METRICS + COST_METRICS),
        segments=metrics.aggregate(
            segment, by=(*BY, "segment", "label"), metrics=metrics.SEGMENT_METRICS
        ),
        drift=metrics.aggregate(
            drift, by=(*BY, "drift_index", "drift_point"), metrics=DRIFT_METRICS
        ),
        cum_regret=metrics.aggregate_curves(
            metrics.rolling_curves(steps, "cum_regret", window=1), runs
        ),
    )


def restrict(ctx: Context, protocol: str) -> Context:
    """The context of one protocol (every frame filtered to its runs)."""

    def keep(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[frame["protocol"] == protocol]

    ids = keep(ctx.runs)["run_id"]
    return Context(
        runs=keep(ctx.runs),
        steps=ctx.steps[ctx.steps["run_id"].isin(ids)],
        frames=figures.Frames(*(keep(f) for f in ctx.frames)),
        table=keep(ctx.table),
        agg=keep(ctx.agg),
        segments=keep(ctx.segments),
        drift=keep(ctx.drift),
        cum_regret=keep(ctx.cum_regret),
    )


class Out(NamedTuple):
    """Where a section saves its figures."""

    dir: Path
    width: float = style.WIDTH

    def figure(self, fig: Figure, name: str, caption: str) -> str:
        """Save ``fig`` as ``figures/<name>.pdf``; return a LaTeX figure."""
        (self.dir / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(self.dir / "figures" / f"{name}.pdf")
        plt.close(fig)
        return (
            "\\begin{figure}[H]\n\\centering\n"
            f"\\includegraphics[width=\\linewidth]{{figures/{name}.pdf}}\n"
            f"\\caption{{{caption}}}\n\\end{{figure}}\n"
        )


Section = Callable[[Context, Out], str]


# --- LaTeX helpers ----------------------------------------------------------


def table_env(tabular: str, caption: str) -> str:
    """A ``table`` around ``tabular``, shrunk to the text width if wider."""
    return (
        "\\begin{table}[H]\n\\centering\n"
        f"\\caption{{{caption}}}\n"
        "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%\n"
        f"{tabular}}}\n\\end{{table}}\n"
    )


def simple_tabular(
    header: Sequence[str], rows: Sequence[Sequence], spec: str = ""
) -> str:
    """A booktabs ``tabular``; cells are escaped, except header cells."""
    spec = spec or "l" * len(header)
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + " \\\\"]
    lines.append("\\midrule")
    lines += [" & ".join(_fmt(c) for c in row) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _fmt(value: object) -> str:
    if isinstance(value, float | np.floating):
        return "--" if pd.isna(value) else f"{value:.4g}"
    return escape(value)


def _facts(items: dict[str, str]) -> str:
    """A two-column table of facts (values are LaTeX); empty values are left out."""
    rows = "\n".join(f"{k} & {v} \\\\" for k, v in items.items() if v)
    return (
        "\\begin{center}\n\\begin{tabular}{@{}>{\\bfseries}p{0.22\\linewidth}"
        ">{\\raggedright\\arraybackslash}p{0.72\\linewidth}@{}}\n\\toprule\n"
        f"{rows}\n\\bottomrule\n\\end{{tabular}}\n\\end{{center}}\n"
    )


def _join(values: Sequence[object], sep: str = ", ") -> str:
    return sep.join(escape(v) for v in values)


def _value(value: object) -> str:
    """A config value as readable escaped text (lists joined, placeholders named)."""
    if isinstance(value, dict):
        return _call(value) if "class" in value else _kv(value)
    if isinstance(value, list):
        return _join(value)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return escape(PLACEHOLDERS.get(value, value) if isinstance(value, str) else value)


def _kv(params: dict, skip: Sequence[str] = ()) -> str:
    """``key: value`` pairs, underscores in keys read as spaces."""
    return "; ".join(
        f"{escape(k.replace('_', ' '))}: {_value(v)}"
        for k, v in params.items()
        if k not in skip
    )


def _call(block: dict) -> str:
    """A ``{class, params}`` config block as ``Name(key=value, ...)``, with
    the module path and random seeds left out."""
    name = escape(str(block.get("class") or block.get("type")).rsplit(".", 1)[-1])
    params = {
        k: v for k, v in (block.get("params") or {}).items() if k != "random_state"
    }
    if not params:
        return name
    inner = ", ".join(f"{escape(k)}={_value(v)}" for k, v in params.items())
    return f"{name}({inner})"


COMMITS = ("repo_commit", "asf_commit")
PLACEHOLDERS = {"${stream.cutoff}": "cutoff", "${seed}": "seed"}
RESULT_MODES = {
    "best_of_scheduled": "the user gets the best result of all solvers run on an "
    "instance",
    "primary": "the user gets the result of the selected solver only; extra runs "
    "only inform the selector",
}
METRIC_DEFINITIONS = {
    "Cost": "Penalised runtime of an instance: the runtime, or the penalty factor "
    "times the cutoff on a timeout.",
    "Regret": "Cost the user sees minus the cost of the VBS on that instance.",
    "Closed gap": "Share of the gap between SBS and VBS (mean cost) that a method "
    "closes: 1 = VBS, 0 = SBS, negative = worse than the SBS.",
    "Mean cost": "Mean cost per instance as the user sees it.",
    "Solver CPU": "Total runtime of all solver runs, extra runs included, in "
    "seconds; a timeout counts the cutoff, not the penalty.",
    "Solver runs": "Number of solver runs over the stream, extra runs included; "
    "runs per instance is its mean.",
    "Extra runs": "Share of instances with more than one solver run.",
    "Overhead": "Total time the selector and the drift detector spend, in seconds.",
    "Recovery": f"Steps after a drift point until the rolling regret "
    f"({metrics.WINDOW} steps) is back to its level before the drift, up to "
    f"{escape(f'{metrics.TOLERANCE:.0%}')} of the jump; 0 = no visible "
    "degradation. Averaged "
    "over the drift points that recovered.",
    "Not recovered": "Number of drift points after which the rolling regret does "
    "not come back before the next drift point.",
    "Peak regret": "Highest rolling regret between a drift point and the next one.",
    "Detection": f"A drift signal within {metrics.DETECTION_TOLERANCE} steps after "
    "a drift point detects it (delay in steps); a drift point without one is "
    "missed, other signals are false alarms.",
}


def _pairs(runs: pd.DataFrame) -> list[tuple[str, str]]:
    pairs = runs[["stream", "protocol"]].drop_duplicates()
    return list(pairs.sort_values(["stream", "protocol"]).itertuples(index=False))


def _select(frame: pd.DataFrame, stream: str, protocol: str) -> pd.DataFrame:
    return frame[(frame["stream"] == stream) & (frame["protocol"] == protocol)]


def _tag(*parts: str) -> str:
    return "_".join(parts)


def _count(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _protocol_text(row: pd.Series) -> str:
    if row["protocol_type"] == "warm":
        return (
            f"{escape(row['protocol'])}: fitted on the first {row['n_warmstart']} "
            "instances (all solvers observed), evaluated on the rest"
        )
    return f"{escape(row['protocol'])}: no warm start, the whole stream is evaluated"


# --- sections ---------------------------------------------------------------


def overview(ctx: Context, out: Out) -> str:
    """Experiment grid, evaluation settings, provenance and metric definitions."""
    runs = ctx.runs
    first = runs.iloc[0]
    protocols = runs.drop_duplicates("protocol").sort_values("protocol")
    env = first["environment"] or {}
    final = int((runs["final"] == True).sum())  # noqa: E712 (None is not final)
    dirty = int(runs["repo_dirty"].fillna(True).sum() + runs["asf_dirty"].sum())
    if final == len(runs) and not dirty:
        status = "Final: every run is traceable to committed code."
    else:
        status = (
            f"\\textbf{{Pilot}}: {len(runs) - final} of {len(runs)} runs were not "
            "started as final runs (clean, committed code with the pinned ASF "
            f"version), {dirty} with uncommitted changes. Not for the thesis."
        )
    repo, asf = (_join(runs[c].dropna().str[:10].unique()) for c in COMMITS)
    commits = f"experiments \\texttt{{{repo}}}, ASF \\texttt{{{asf}}}"
    items = {
        "Experiment": f"{escape(first['experiment'])} (config "
        f"{escape(first['config_file'])})",
        "Runs": f"{len(runs)}: "
        + ", ".join(
            _count(runs[column].nunique(), word)
            for column, word in (
                ("stream", "stream"),
                ("protocol", "protocol"),
                ("method", "method"),
                ("seed", "selector seed"),
                ("stream_seed", "stream seed"),
            )
        ),
        "Protocols": " \\newline ".join(
            _protocol_text(p) for _, p in protocols.iterrows()
        ),
        "Objective": "maximise performance"
        if first["maximize"]
        else f"minimise runtime; a timeout costs {first['par_factor']:g} times the "
        "cutoff (PAR" + f"{first['par_factor']:g})",
        "Result": RESULT_MODES.get(first["result_mode"], escape(first["result_mode"])),
        "Status": status,
        "Code": commits,
        "Environment": escape(
            f"Python {env.get('python', '?')}; "
            + ", ".join(f"{k} {v}" for k, v in (env.get("packages") or {}).items())
        ),
        "Run dates": f"{runs['created_utc'].min()[:10]} to "
        f"{runs['created_utc'].max()[:10]}",
    }
    return (
        "\\section{Overview}\n"
        + _facts(items)
        + "\\subsection{Metrics}\n"
        + _facts(METRIC_DEFINITIONS)
    )


def stream(ctx: Context, out: Out) -> str:
    """Per stream: its construction; per protocol: segments and oracle figures."""
    text = "\\section{Streams}\n"
    for name, runs in ctx.runs.groupby("stream", sort=True):
        first = runs.sort_values("run_id").iloc[0]
        drift = first["stream_drift"] if isinstance(first["stream_drift"], dict) else {}
        drift_params = dict(drift.get("params") or {})
        schedule = drift_params.pop("schedule", None)
        text += f"\\subsection{{{escape(name)}}}\n" + _facts(
            {
                "Source": f"{escape(first['stream_kind'])}; "
                + _kv(first["stream_params"] or {}, skip=("root",)),
                "Instances": f"{first['n_instances']} instances, "
                f"{first['n_features']} features, {first['n_algorithms']} "
                f"algorithms, cutoff {first['cutoff']:g}",
                "Ordering": (
                    f"{escape(drift.get('builder'))} ({_kv(drift_params)})"
                    if drift
                    else "no drift block (random order or the stream kind's own)"
                ),
                "Schedule": _join(
                    ["→".join(e) if isinstance(e, list) else e for e in schedule],
                    " → ",
                )
                if schedule
                else "",
                "SBS": f"{escape(first['sbs_algorithm_stream'])} (best single "
                "solver on the whole stream)",
                "Stream seeds": _join(sorted(runs["stream_seed"].unique())),
            }
        )
        for protocol in sorted(runs["protocol"].unique()):
            text += _stream_protocol(ctx, out, name, protocol)
    return text


def _stream_protocol(ctx: Context, out: Out, name: str, protocol: str) -> str:
    run, steps = plots.first_run(ctx.steps, ctx.runs, name, protocol)
    run_frame = run.to_frame().T
    steps = steps.assign(segment=metrics.step_segments(steps, run_frame))
    unsolved = run["par_factor"] * run["cutoff"]
    rows = []
    for seg in metrics.segments(run_frame).itertuples():
        part = steps[steps["segment"] == seg.segment]
        best = part["vbs_algorithm"].value_counts(normalize=True)
        rows.append(
            [
                seg.label,
                f"[{seg.start}, {seg.end})",
                seg.length,
                f"{best.index[0]} ({best.iloc[0]:.0%})" if len(best) else "--",
                float(part["vbs_cost"].mean()),
                float(part["sbs_cost"].mean()),
                float("nan")
                if run["maximize"]
                else float((part["vbs_cost"] >= unsolved).mean()),
            ]
        )
    header = [
        "Segment",
        "Steps",
        "Length",
        "Most often VBS",
        "VBS cost",
        "SBS cost",
        "Unsolved",
    ]
    tag = _tag(name, protocol)
    caption = escape(f"{name}, {protocol}")
    seed = f"stream seed {run['stream_seed']}"
    return (
        f"\\subsubsection{{Protocol {escape(protocol)}}}\n"
        + _facts(
            {
                "Evaluated": f"{run['n_evaluated']} instances"
                + (
                    f" after {run['n_warmstart']} warm-start instances (SBS of "
                    f"the warm start: {escape(run['sbs_algorithm_warmstart'])})"
                    if run["n_warmstart"]
                    else ""
                ),
                "Drift points": "steps " + _join(run["drift_points"])
                if run["drift_points"]
                else "none",
                "Gradual transitions": _join(
                    [f"[{a}, {b})" for a, b in run["transition_intervals"]]
                ),
            }
        )
        + table_env(
            simple_tabular(header, rows, "lllrrrr"),
            f"{caption}: segments between drift points ({seed}). Costs are means "
            "over the segment; unsolved is the share of instances no algorithm "
            "solves within the cutoff.",
        )
        + out.figure(
            plots.oracle_costs(ctx.steps, ctx.runs, name, protocol, width=out.width)[0],
            f"oracle_costs_{tag}",
            f"{caption}: mean cost of the VBS and the SBS over the last "
            f"{metrics.WINDOW} steps (fewer at the start; {seed}).",
        )
        + out.figure(
            plots.segment_oracle_share(
                ctx.steps, ctx.runs, name, protocol, width=out.width
            )[0],
            f"segment_oracle_{tag}",
            f"{caption}: how often each algorithm is the VBS, per segment.",
        )
    )


def methods(ctx: Context, out: Out) -> str:
    """Methods, their protocols and selector configuration as written in the
    config (module paths, seeds and defaults left out)."""
    runs = ctx.runs.sort_values(["method_family", "method"])
    family = (runs["method_family"] != runs["method"]).any()  # only with sweeps
    rows = []
    for name, group in runs.groupby("method", sort=False):
        first = group.iloc[0]
        cells = [escape(name)]
        cells += [escape(first["method_family"])] if family else []
        cells += [
            _join(sorted(group["protocol"].unique())),
            _call(first["selector_config"]),
        ]
        rows.append(" & ".join(cells) + r" \\")
    header = ["Method"] + (["Family"] if family else []) + ["Protocols", "Selector"]
    spec = "l" * (len(header) - 1) + r">{\raggedright\arraybackslash}p{0.62\linewidth}"
    tabular = "\n".join(
        [
            rf"\begin{{tabular}}{{{spec}}}",
            r"\toprule",
            " & ".join(header) + r" \\",
            r"\midrule",
            ("\n" + r"\addlinespace" + "\n").join(rows),
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return (
        r"\section{Methods}"
        + "\n"
        + table_env(
            tabular + "\n",
            "Methods and their selector configuration as set in the config; "
            "everything else is at its default.",
        )
    )


def performance(ctx: Context, out: Out) -> str:
    """Cost and quality tables per protocol, fronts, and per stream and
    protocol: cumulative regret, solver runs per instance, primary choices."""
    text = "\\section{Performance}\n"
    for protocol in sorted(ctx.runs["protocol"].unique()):
        p = escape(protocol)
        text += (
            f"\\subsection{{Protocol {p}}}\n"
            + table_env(
                tables.cost_table(ctx.agg, protocol)[1],
                f"{p}: quality and cost, mean $\\pm$ ci95 over seeds. Mean cost "
                "is the PAR cost the user sees; solver runs count every run, extra "
                "runs included; overhead is selector plus detector time.",
            )
            + table_env(
                tables.main_table(ctx.frames.agg, protocol)[1],
                f"{p}: main table (recovery in steps; not recovered counts drift "
                "points).",
            )
        )
    text += "\\subsection{Cost against quality}\n"
    text += out.figure(
        plots.front_grid(ctx.frames.agg, width=out.width)[0],
        "front",
        "Closed gap against solver CPU, mean and ci95 over seeds.",
    )
    text += out.figure(
        plots.front_grid(ctx.frames.agg, overhead=True, width=out.width)[0],
        "front_overhead",
        "As before, with selector and detector time added to x.",
    )
    for name, protocol in _pairs(ctx.runs):
        tag, caption = _tag(name, protocol), escape(f"{name}, {protocol}")
        text += (
            f"\\subsection{{{caption}}}\n"
            + out.figure(
                plots.cumulative_regret(
                    ctx.cum_regret, ctx.runs, name, protocol, width=out.width
                )[0],
                f"cum_regret_{tag}",
                f"{caption}: cumulative regret, mean and ci95 over seeds.",
            )
            + out.figure(
                plots.runs_per_instance(
                    ctx.steps, ctx.runs, name, protocol, width=out.width
                )[0],
                f"runs_per_instance_{tag}",
                f"{caption}: share of instances by number of solver runs.",
            )
            + out.figure(
                plots.selection_heatmap(
                    ctx.steps, ctx.runs, name, protocol, width=out.width
                )[0],
                f"selection_{tag}",
                f"{caption}: primary choice per method against the VBS.",
            )
        )
    return text


def drift(ctx: Context, out: Out) -> str:
    """Per stream and protocol with drift: regret and extra-run curves,
    detection, and per-segment and per-drift-point tables."""
    text = "\\section{Drift adaptation}\n"
    for protocol in sorted(ctx.runs["protocol"].unique()):
        detection, latex = tables.detection_table(ctx.frames.agg, protocol)
        if not detection.empty:
            text += table_env(
                latex,
                f"{escape(protocol)}: detection delay (steps), missed drifts and "
                f"false alarms (tolerance {metrics.DETECTION_TOLERANCE} steps).",
            )
    for name, protocol in _pairs(ctx.runs):
        runs = _select(ctx.runs, name, protocol)
        if not runs["drift_points"].map(len).any():
            continue
        tag, caption = _tag(name, protocol), escape(f"{name}, {protocol}")
        text += f"\\subsection{{{caption}}}\n"
        text += out.figure(
            plots.regret_curves(
                ctx.frames.regret, ctx.runs, name, protocol, width=out.width
            )[0],
            f"regret_{tag}",
            f"{caption}: rolling regret ({metrics.WINDOW} steps); drift points "
            "dashed, transitions shaded.",
        )
        text += out.figure(
            plots.extra_run_rate(
                ctx.frames.extra_runs, ctx.runs, name, protocol, width=out.width
            )[0],
            f"extra_runs_{tag}",
            f"{caption}: rolling share of instances with more than one run.",
        )
        if runs["uses_detector"].any():
            text += out.figure(
                plots.detection_raster(ctx.runs, name, protocol, width=out.width)[0],
                f"raster_{tag}",
                f"{caption}: drift signals per method and seed.",
            )
        segments = _select(ctx.segments, name, protocol)
        labels = segments.drop_duplicates("segment").set_index("segment")["label"]
        headers = {i: escape(label) for i, label in labels.items()}
        text += table_env(
            tables.pivot_table(
                segments, "segment", "closed_gap", headers=headers, higher=True
            )[1],
            f"{caption}: closed gap per segment (SBS of the whole stream).",
        )
        points = _select(ctx.drift, name, protocol)
        point_labels = points.drop_duplicates("drift_index").set_index("drift_index")
        headers = {i: f"$p={row.drift_point}$" for i, row in point_labels.iterrows()}
        legend = "; ".join(
            f"$p={row.drift_point}$: {escape(labels.get(i, ''))}$\\to$"
            f"{escape(labels.get(i + 1, ''))}"
            for i, row in point_labels.iterrows()
        )
        for metric, higher, what in (
            (
                "recovery",
                False,
                "recovery in steps (-- never recovered or not measurable)",
            ),
            ("peak_regret", False, "peak rolling regret after the drift point"),
            (
                "extra_run_rate_change",
                None,
                "change of the extra-run rate, "
                f"{metrics.WINDOW} steps after minus before the drift point",
            ),
        ):
            text += table_env(
                tables.pivot_table(
                    points, "drift_index", metric, headers=headers, higher=higher
                )[1],
                f"{caption}: {what}. Drift points: {legend}.",
            )
    return text


def statistics(ctx: Context, out: Out, reference: str = "online-pairwise") -> str:
    """Mean ranks, Friedman and Wilcoxon against ``reference`` on closed gap,
    over the (stream, protocol) cells that every compared method has."""
    table = ctx.table
    cells = table.groupby(["stream", "protocol"]).ngroups
    present = table.groupby("method")[["stream", "protocol"]].apply(
        lambda f: len(f.drop_duplicates())
    )
    compared = table[table["method"].isin(present[present == cells].index)]
    n = compared.groupby(["stream", "protocol"]).ngroups
    text = "\\section{Statistics}\n"
    if n < 3 or compared["method"].nunique() < 3:
        return text + (
            f"Only {n} (stream, protocol) cells with every method; the tests need "
            "at least three cells and three methods, so none are reported.\n"
        )
    ranks = stats.mean_ranks(compared)
    text += table_env(
        simple_tabular(["Method", "Mean rank", "Cells"], ranks.values.tolist(), "lrr"),
        f"Mean rank on closed gap (1 = best) over {n} cells.",
    )
    friedman = stats.friedman(compared)
    text += (
        f"Friedman test: $\\chi^2 = {friedman['statistic']:.2f}$, "
        f"$p = {friedman['p_value']:.3g}$.\n"
    )
    if reference not in set(compared["method"]):
        reference = ranks["method"].iloc[0]
    wilcoxon = stats.wilcoxon_vs_reference(compared, reference)
    header = ["Method", "Cells", "Median diff.", "Statistic", "p", "p (Holm)"]
    return text + table_env(
        simple_tabular(header, wilcoxon.values.tolist(), "lrrrrr"),
        f"Wilcoxon signed-rank test on closed gap against {escape(reference)}.",
    )


def appendix(ctx: Context, out: Out) -> str:
    """Every run, and the resolved selector config of every method."""
    rows = [
        [r.run_id, r.closed_gap, r.solver_cpu, r.n_drift_signals, r.n_retrains]
        for r in ctx.table.sort_values("run_id").itertuples()
    ]
    header = ["Run", "Closed gap", "Solver CPU", "Signals", "Retrains"]
    tabular = (
        simple_tabular(header, rows, "lrrrr")
        .replace("\\begin{tabular}", "\\begin{longtable}")
        .replace("\\end{tabular}", "\\end{longtable}")
    )
    text = "\\appendix\n\\section{Runs}\n{\\small\n" + tabular + "}\n"
    text += "\\section{Selector configurations}\n"
    for name, runs in ctx.runs.groupby("method", sort=True):
        config = json.dumps(runs.iloc[0]["selector_config"], indent=2)
        text += (
            f"\\subsection*{{{escape(name)}}}\n{{\\footnotesize\n"
            f"\\begin{{verbatim}}\n{config}\n\\end{{verbatim}}\n}}\n"
        )
    return text


SECTIONS: dict[str, Section] = {
    "overview": overview,
    "stream": stream,
    "methods": methods,
    "performance": performance,
    "drift": drift,
    "statistics": statistics,
    "appendix": appendix,
}

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{array,booktabs,graphicx,float,longtable}
\usepackage[hidelinks]{hyperref}
"""


def write_report(
    ctx: Context,
    out: str | Path,
    sections: Sequence[str] = tuple(SECTIONS),
    width: float = style.WIDTH,
    title: str | None = None,
) -> Path:
    """Write ``report.tex``, ``sections/<name>.tex`` and ``figures/`` to ``out``.

    ``title`` defaults to the experiment name. Returns the path of
    ``report.tex``.
    """
    unknown = sorted(set(sections) - set(SECTIONS))
    if unknown:
        raise ValueError(f"unknown sections {unknown}; known: {list(SECTIONS)}")
    target = Out(Path(out), width)
    (target.dir / "sections").mkdir(parents=True, exist_ok=True)
    for name in sections:
        latex = SECTIONS[name](ctx, target)
        (target.dir / "sections" / f"{name}.tex").write_text(latex, encoding="utf-8")
    title = escape(title or ctx.runs["experiment"].iloc[0])
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body = "\n".join(f"\\input{{sections/{name}}}" for name in sections)
    tex = target.dir / "report.tex"
    tex.write_text(
        PREAMBLE
        + f"\\title{{Experiment report: {title}}}\n\\date{{{date}}}\n"
        + "\\begin{document}\n\\maketitle\n\\tableofcontents\n\\clearpage\n"
        + body
        + "\n\\end{document}\n",
        encoding="utf-8",
    )
    return tex


def compile_pdf(tex: Path, timeout: float = 600) -> Path | None:
    """Build the PDF with latexmk; None if latexmk is not installed."""
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        return None
    done = subprocess.run(
        [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex.name],
        cwd=tex.parent,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if done.returncode != 0:
        tail = "\n".join(done.stdout.splitlines()[-30:])
        raise RuntimeError(f"latexmk failed on {tex}:\n{tail}")
    return tex.with_suffix(".pdf")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m oas_experiments.report",
        description="Write a LaTeX report (and PDF) of an experiment.",
    )
    parser.add_argument("experiment_dir", help="results/<experiment>")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--sections",
        default=",".join(SECTIONS),
        help=f"comma-separated, in order (default: {','.join(SECTIONS)})",
    )
    parser.add_argument("--no-pdf", action="store_true", help="only write LaTeX")
    parser.add_argument(
        "--per-protocol",
        action="store_true",
        help="one report per protocol, in <out>/<protocol>/",
    )
    parser.add_argument(
        "--width", type=float, default=style.WIDTH, help="figure width in inches"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mpl.use("Agg")
    style.apply()
    try:
        ctx = load(args.experiment_dir)
    except ResultError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    sections = args.sections.split(",")
    if args.per_protocol:
        # one colour per family across all protocol reports
        style.pin_families(ctx.runs["method_family"])
        texs = [
            write_report(
                restrict(ctx, protocol),
                Path(args.out) / protocol,
                sections,
                args.width,
                title=f"{ctx.runs['experiment'].iloc[0]}, protocol {protocol}",
            )
            for protocol in sorted(ctx.runs["protocol"].unique())
        ]
    else:
        texs = [write_report(ctx, args.out, sections, args.width)]
    for tex in texs:
        print(tex)
        if not args.no_pdf:
            pdf = compile_pdf(tex)
            print(pdf or "latexmk not found; wrote LaTeX only", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
