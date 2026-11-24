"""A LaTeX report (and PDF) of one experiment: summary with the main results,
stream, methods, results, drift adaptation, statistics and an appendix
(metrics, provenance, runs).

    python -m oas_experiments.report results/test_sat12 --out reports/test_sat12
    python -m oas_experiments.report results/x --out r/x --sections summary,stream
    python -m oas_experiments.report results/x --out reports/x --no-pdf

Writes ``report.tex``, one ``sections/<name>.tex`` per section (each can be
``\\input`` on its own), ``figures/*.pdf`` and, with latexmk installed,
``report.pdf``.

A section is a function ``(ctx: Context, out: Out) -> str`` that returns
LaTeX and saves its figures through ``out``. :data:`SECTIONS` lists them in
report order; a report for another scenario passes its own list of names to
:func:`write_report`, or adds a function to :data:`SECTIONS`. Methods are
listed in the order of the experiment's config file.
"""

from __future__ import annotations

import argparse
import os
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
import yaml
from matplotlib.figure import Figure

from oas_experiments import figures
from oas_experiments.analysis import metrics, plots, stats, style, tables
from oas_experiments.analysis.tables import escape
from oas_experiments.results import ResultError, load_experiment

BY = ("stream", "protocol", "protocol_type", "method")
COST_METRICS = ["mean_runs_per_instance", "n_retrains", "total_runs", "solved_rate"]
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
    table = table.merge(solved_rate(results.steps, runs), on="run_id", how="left")
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


def solved_rate(steps: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Share of instances on which the user gets a result within the cutoff
    (cost below the timeout penalty), per run; NaN for maximisation."""
    penalty = runs.set_index("run_id").eval("par_factor * cutoff")
    penalty[runs.set_index("run_id")["maximize"].astype(bool)] = np.nan
    limit = steps["run_id"].map(penalty)
    solved = (steps["user_cost"] < limit).where(limit.notna())
    rate = solved.groupby(steps["run_id"]).mean()
    return rate.rename("solved_rate").rename_axis("run_id").reset_index()


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

    def figure(
        self, fig: Figure, name: str, caption: str, keep_titles: bool = False
    ) -> str:
        """Save ``fig`` as ``figures/<name>.pdf``; return a LaTeX figure.

        Axes titles are removed (the caption and the heading say what the
        figure shows), unless ``keep_titles`` (e.g. panels of a grid)."""
        if not keep_titles:
            for ax in fig.axes:
                ax.set_title("")
        (self.dir / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(self.dir / "figures" / f"{name}.pdf")
        plt.close(fig)
        return (
            "\\begin{figure}[!htb]\n\\centering\n"
            f"\\includegraphics[width=\\linewidth]{{figures/{name}.pdf}}\n"
            f"\\caption{{{caption}}}\n\\end{{figure}}\n"
        )


Section = Callable[[Context, Out], str]


# --- LaTeX helpers ----------------------------------------------------------


def table_env(tabular: str, caption: str) -> str:
    """A ``table`` around ``tabular`` in small type, shrunk to the text width
    only if it is still wider."""
    return (
        "\\begin{table}[!htb]\n\\centering\n\\small\n"
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
    "Solved": "Share of instances on which the user gets a result within the cutoff.",
    "Solver CPU": "Total runtime of all solver runs, extra runs included, in "
    "hours; a timeout counts the cutoff, not the penalty.",
    "Extra CPU": "The part of the solver CPU spent on extra runs.",
    "Runs/instance": "Mean number of solver runs per instance, extra runs included.",
    "Extra runs": "Share of instances with more than one solver run.",
    "Overhead": "Total time the selector and the drift detector spend, in seconds.",
    "Recovery": f"Steps after a drift point until the closed gap per window "
    f"({metrics.WINDOW} steps) is back to its level before the drift, up to "
    f"{escape(f'{metrics.TOLERANCE:.0%}')} of the drop and at most "
    f"{metrics.CAP:g} below it, and stays there for "
    f"{metrics.HOLD} consecutive windows; 0 = no visible degradation. Averaged "
    "over the drift points that recovered.",
    "Not recovered": "Number of drift points after which the closed gap per "
    f"window does not come back and hold for {metrics.HOLD} windows before the "
    "next drift point (drift points without a level before the drift or with a "
    "too short segment are not counted).",
    "Peak regret": "Highest rolling regret between a drift point and the next one.",
    "Detection": f"A drift signal within {metrics.DETECTION_TOLERANCE} steps after "
    "a drift point detects it (delay in steps); a drift point without one is "
    "missed, other signals are false alarms.",
}


# Quality data (maximize): fitness replaces cost; no cutoff, so no solved
# instances and no CPU time.
QUALITY_DEFINITIONS = {
    "Fitness": "Quality of the result on an instance, higher is better (bin "
    "packing: Falkenauer fitness of the packing).",
    "Regret": "Fitness of the VBS minus the fitness the user gets on that instance.",
    "Closed gap": "Share of the gap between SBS and VBS (mean fitness) that a "
    "method closes: 1 = VBS, 0 = SBS, negative = worse than the SBS.",
    "Mean fitness": "Mean fitness per instance as the user sees it.",
}
QUALITY_DROPPED = ("Cost", "Mean cost", "Solved", "Solver CPU", "Extra CPU")


def _maximize(runs: pd.DataFrame) -> bool:
    """Whether the runs maximise a quality (e.g. bin-packing fitness)."""
    return bool(runs["maximize"].astype(bool).all())


def _pairs(runs: pd.DataFrame) -> list[tuple[str, str]]:
    pairs = runs[["stream", "protocol"]].drop_duplicates()
    return list(pairs.sort_values(["stream", "protocol"]).itertuples(index=False))


def _select(frame: pd.DataFrame, stream: str, protocol: str) -> pd.DataFrame:
    return frame[(frame["stream"] == stream) & (frame["protocol"] == protocol)]


def _tag(*parts: str) -> str:
    return "_".join(parts)


def _count(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


# --- sections ---------------------------------------------------------------


def _single(ctx: Context) -> bool:
    """Whether the report has one (stream, protocol) pair: then headings and
    captions need not name it."""
    return len(_pairs(ctx.runs)) == 1


def _part(ctx: Context, name: str, protocol: str) -> str:
    """A subsection heading per (stream, protocol) pair; none if there is only
    one (headings without text of their own float away from the figures)."""
    if _single(ctx):
        return ""
    return f"\\subsection{{{escape(f'{name}, {protocol}')}}}\n"


def _seeds(runs: pd.DataFrame) -> str:
    n = runs["seed"].nunique() * runs["stream_seed"].nunique()
    return "one seed" if n == 1 else f"mean $\\pm$ 95\\% CI over {n} seeds"


def _short(labels: Sequence[str]) -> list[str]:
    """Labels without the prefix they all share (``SAT12-HAND`` -> ``HAND``),
    cut at a ``-``, ``_`` or space."""
    labels = [str(label) for label in labels]
    prefix = os.path.commonprefix(labels) if len(set(labels)) > 1 else ""
    cut = max(prefix.rfind(c) for c in "-_ ") + 1
    return [label[cut:] for label in labels]


def _protocol_text(row: pd.Series) -> str:
    if row["protocol_type"] == "warm":
        return (
            f"{escape(row['protocol'])}: fitted on the first {row['n_warmstart']} "
            "instances (all solvers observed), evaluated on the rest"
        )
    return f"{escape(row['protocol'])}: no warm start, the whole stream is evaluated"


def _is_final(runs: pd.DataFrame) -> tuple[bool, str]:
    """Whether every run is a final run on clean code, and why not."""
    final = int((runs["final"] == True).sum())  # noqa: E712 (None is not final)
    dirty = int(runs["repo_dirty"].fillna(True).sum() + runs["asf_dirty"].sum())
    if final == len(runs) and not dirty:
        return True, "every run is traceable to committed code"
    reasons = []
    if final < len(runs):
        reasons.append(f"{len(runs) - final} of {len(runs)} runs not started as final")
    if dirty:
        reasons.append(f"{dirty} with uncommitted changes")
    return False, "; ".join(reasons)


def _cells(ctx: Context) -> tuple[pd.DataFrame, int]:
    """Runs of the methods that every (stream, protocol) cell has, and the
    number of cells."""
    table = ctx.table
    cells = table.groupby(["stream", "protocol"]).ngroups
    present = table.groupby("method")[["stream", "protocol"]].apply(
        lambda f: len(f.drop_duplicates())
    )
    compared = table[table["method"].isin(present[present == cells].index)]
    return compared, compared.groupby(["stream", "protocol"]).ngroups


def _testable(ctx: Context) -> bool:
    compared, n = _cells(ctx)
    return n >= 3 and compared["method"].nunique() >= 3


def summary(ctx: Context, out: Out) -> str:
    """The experiment in a few lines, the main table and the cost-quality front."""
    runs = ctx.runs
    first = runs.iloc[0]
    protocols = runs.drop_duplicates("protocol").sort_values("protocol")
    final, why = _is_final(runs)
    text = "\\section{Summary}\n"
    if not final:
        text += (
            "\\begin{center}\\fbox{\\parbox{0.96\\linewidth}{\\textbf{Pilot run, "
            f"not for the thesis}}: {escape(why)} (details in the appendix).}}}}"
            "\\end{center}\n"
        )
    items = {
        "Experiment": f"{escape(first['experiment'])} (config "
        f"{escape(first['config_file'])}), {len(runs)} runs",
        "Grid": ", ".join(
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
        "Objective": "maximise a quality (e.g. fitness); no runtimes, so cost is "
        "counted in solver runs"
        if first["maximize"]
        else f"minimise runtime; a timeout costs {first['par_factor']:g} times the "
        f"cutoff (PAR{first['par_factor']:g})",
        "Result": RESULT_MODES.get(first["result_mode"], escape(first["result_mode"])),
        "Tests": ""
        if _testable(ctx)
        else "no significance tests: they need at least three (stream, protocol) "
        "cells and three methods",
    }
    text += _facts(items)
    quality = _maximize(runs)
    columns = (
        "Extra runs: instances with more than one solver run, so a method can be "
        "better for the user and still cost more."
        if quality
        else "Solved: instances finished within the cutoff. CPU counts every "
        "solver run, extra runs included, so a method can be better for the "
        "user and still cost more."
    )
    for protocol in sorted(runs["protocol"].unique()):
        which = "" if protocols.shape[0] == 1 else f", protocol {escape(protocol)}"
        text += table_env(
            tables.main_table(ctx.agg, protocol, quality)[1],
            f"Main results{which} ({_seeds(runs)}). Closed gap: 1 = VBS, 0 = SBS. "
            f"{columns} Best value per column in bold.",
        )
    cost = "solver runs per instance" if quality else "solver CPU"
    text += out.figure(
        plots.front_grid(
            ctx.agg, hours=True, by_runs=quality, width=out.width, aspect=0.38
        )[0],
        "front",
        f"Closed gap against {cost}: up is better, left is cheaper.",
        keep_titles=not _single(ctx),
    )
    return text


def stream(ctx: Context, out: Out) -> str:
    """Per stream: its construction; per protocol: segments and oracle figures."""
    names = sorted(ctx.runs["stream"].unique())
    text = "\\section{Stream}\n" if len(names) == 1 else "\\section{Streams}\n"
    for name, runs in ctx.runs.groupby("stream", sort=True):
        first = runs.sort_values("run_id").iloc[0]
        drift = first["stream_drift"] if isinstance(first["stream_drift"], dict) else {}
        drift_params = dict(drift.get("params") or {})
        schedule = drift_params.pop("schedule", None)
        if len(names) > 1:
            text += f"\\subsection{{{escape(name)}}}\n"
        protocols = sorted(runs["protocol"].unique())
        parts = [_stream_protocol(ctx, out, name, p) for p in protocols]
        several = ctx.runs["protocol"].nunique() > 1
        text += _facts(
            {
                "Name": escape(name) if len(names) == 1 else "",
                "Source": _source(first),
                "Instances": f"{first['n_instances']} instances, "
                f"{first['n_features']} features, {first['n_algorithms']} "
                "algorithms"
                + ("" if first["maximize"] else f", cutoff {first['cutoff']:g}"),
                "Ordering": (
                    f"{escape(drift.get('builder'))} ({_kv(drift_params)})"
                    # a sort key on performance columns is named ``runtime``
                    .replace("runtime", "fitness" if first["maximize"] else "runtime")
                    if drift
                    else "no drift block (random order or the stream kind's own)"
                ),
                # "+" mixes groups within a segment, "→" separates segments
                "Schedule": _join(
                    ["+".join(e) if isinstance(e, list) else e for e in schedule],
                    " → ",
                )
                if schedule
                else "",
                "SBS": f"{escape(first['sbs_algorithm_stream'])} (best single "
                "solver on the whole stream)",
                "Stream seeds": _join(sorted(runs["stream_seed"].unique())),
                **({} if several else parts[0][0]),
            }
        )
        for protocol, (facts, body) in zip(protocols, parts, strict=True):
            if several:
                text += f"\\subsection{{Protocol {escape(protocol)}}}\n"
                text += _facts(facts)
            text += body
    return text


def _source(run: pd.Series) -> str:
    """Stream kind and parameters; of ``root`` only the last folder (the
    dataset, e.g. ``E_120_N_40_60``)."""
    params = dict(run["stream_params"] or {})
    if "root" in params:
        params["root"] = Path(str(params["root"]).replace("\\", "/")).name
    return "; ".join(filter(None, [escape(run["stream_kind"]), _kv(params)]))


def _stream_protocol(
    ctx: Context, out: Out, name: str, protocol: str
) -> tuple[dict[str, str], str]:
    """Facts of a stream under a protocol, and its segment table and figures."""
    run, steps = plots.first_run(ctx.steps, ctx.runs, name, protocol)
    run_frame = run.to_frame().T
    steps = steps.assign(segment=metrics.step_segments(steps, run_frame))
    unsolved = run["par_factor"] * run["cutoff"]
    quality = bool(run["maximize"])
    value = "{:.3f}" if quality else "{:.0f}"
    rows = []
    for seg in metrics.segments(run_frame).itertuples():
        part = steps[steps["segment"] == seg.segment]
        best = part["vbs_algorithm"].value_counts(normalize=True)
        rows.append(
            [
                seg.label,
                f"{seg.start}--{seg.end}",
                seg.length,
                f"{best.index[0]} ({best.iloc[0]:.0%})" if len(best) else "--",
                value.format(part["vbs_cost"].mean()),
                value.format(part["sbs_cost"].mean()),
            ]
            + ([] if quality else [f"{(part['vbs_cost'] >= unsolved).mean():.0%}"])
        )
    measure = "fitness" if quality else "cost"
    header = [
        "Segment",
        "Steps",
        "Length",
        "Most often VBS",
        f"VBS {measure}",
        f"SBS {measure}",
    ] + ([] if quality else ["Unsolved"])
    tag = _tag(name, protocol)
    seed = "" if run["stream_seed"] == 0 else f" (stream seed {run['stream_seed']})"
    facts = {
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
    body = (
        table_env(
            simple_tabular(header, rows, "llrlrr" + ("" if quality else "r")),
            f"Segments between drift points{seed}. "
            + (
                "Fitness values are means over the segment."
                if quality
                else "Costs are means over the segment; unsolved is the share of "
                "instances no algorithm solves within the cutoff."
            ),
        )
        + out.figure(
            plots.oracle_costs(ctx.steps, ctx.runs, name, protocol, width=out.width)[0],
            f"oracle_costs_{tag}",
            f"Mean {measure} of the VBS and the SBS over the last "
            f"{metrics.WINDOW} steps{'' if quality else ' (log scale)'}; drift "
            "points dashed.",
        )
        + out.figure(
            plots.segment_oracle_share(
                ctx.steps, ctx.runs, name, protocol, width=out.width
            )[0],
            f"segment_oracle_{tag}",
            "How often each algorithm is the VBS, per segment (rows sum to 1).",
        )
    )
    return facts, body


def methods(ctx: Context, out: Out) -> str:
    """Methods, their protocols and selector configuration as written in the
    config (module paths, seeds and defaults left out)."""
    runs = ctx.runs
    family = (runs["method_family"] != runs["method"]).any()  # only with sweeps
    several = runs["protocol"].nunique() > 1
    rows = []
    for name in style.method_order(runs):
        group = runs[runs["method"] == name]
        first = group.iloc[0]
        cells = [escape(name)]
        cells += [escape(first["method_family"])] if family else []
        cells += [_join(sorted(group["protocol"].unique()))] if several else []
        cells += [_call(first["selector_config"])]
        rows.append(" & ".join(cells) + r" \\")
    header = (
        ["Method"]
        + (["Family"] if family else [])
        + (["Protocols"] if several else [])
        + ["Selector"]
    )
    spec = "l" * (len(header) - 1) + r">{\raggedright\arraybackslash}p{0.7\linewidth}"
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
            "Methods and their selector settings as set in the config; "
            "everything else is at its default.",
        )
    )


def results(ctx: Context, out: Out) -> str:
    """Per protocol the cost table; per stream and protocol cumulative regret,
    solver runs per instance and primary choices."""
    text = "\\section{Results}\n"
    several = ctx.runs["protocol"].nunique() > 1
    quality = _maximize(ctx.runs)
    extra_cpu = "" if quality else "Extra CPU is the solver time of extra runs; "
    for protocol in sorted(ctx.runs["protocol"].unique()):
        which = f", protocol {escape(protocol)}" if several else ""
        text += table_env(
            tables.cost_table(ctx.agg, protocol, quality)[1],
            f"What the methods spend besides one solver run per instance{which} "
            f"({_seeds(ctx.runs)}). {extra_cpu}Overhead is selector plus "
            "detector time.",
        )
    for name, protocol in _pairs(ctx.runs):
        tag = _tag(name, protocol)
        text += (
            _part(ctx, name, protocol)
            + out.figure(
                plots.cumulative_regret(
                    ctx.cum_regret, ctx.runs, name, protocol, width=out.width
                )[0],
                f"cum_regret_{tag}",
                f"Cumulative regret ({_seeds(ctx.runs)}); drift points dashed.",
            )
            + out.figure(
                plots.runs_per_instance(
                    ctx.steps, ctx.runs, name, protocol, width=out.width
                )[0],
                f"runs_per_instance_{tag}",
                "Share of instances by number of solver runs.",
            )
            + out.figure(
                plots.selection_heatmap(
                    ctx.steps, ctx.runs, name, protocol, width=out.width
                )[0],
                f"selection_{tag}",
                "Share of instances on which each algorithm was the primary "
                "choice, per method, and how often it is the VBS (first row). "
                "Rows sum to 1.",
            )
        )
    return text


def drift(ctx: Context, out: Out) -> str:
    """Per stream and protocol with drift: regret per method, extra-run rate,
    detection, and per-segment and per-drift-point tables."""
    # its own page: it starts with floats, which would leave the heading alone
    text = "\\clearpage\n\\section{Drift adaptation}\n"
    several = ctx.runs["protocol"].nunique() > 1
    for protocol in sorted(ctx.runs["protocol"].unique()):
        detection, latex = tables.detection_table(ctx.agg, protocol)
        if not detection.empty:
            which = f", protocol {escape(protocol)}" if several else ""
            text += table_env(
                latex,
                f"Drift detection{which}: delay in steps, missed drift points and "
                f"false alarms (a signal counts within "
                f"{metrics.DETECTION_TOLERANCE} steps of a drift point).",
            )
    for name, protocol in _pairs(ctx.runs):
        runs = _select(ctx.runs, name, protocol)
        if not runs["drift_points"].map(len).any():
            continue
        tag = _tag(name, protocol)
        signals = (
            " Ticks at the bottom are drift signals."
            if runs["uses_detector"].any()
            else ""
        )
        text += _part(ctx, name, protocol)
        text += out.figure(
            plots.small_multiples(
                ctx.frames.regret,
                ctx.runs,
                name,
                protocol,
                "Rolling regret",
                width=out.width,
            )[0],
            f"regret_{tag}",
            f"Rolling regret ({metrics.WINDOW} steps), one panel per method with "
            f"the others in grey; drift points dashed.{signals}",
        )
        extra = ctx.agg[
            (ctx.agg["stream"] == name)
            & (ctx.agg["protocol"] == protocol)
            & (ctx.agg["extra_run_rate_mean"] > 0)
        ]
        if not extra.empty:
            text += out.figure(
                plots.extra_run_rate(
                    ctx.frames.extra_runs,
                    ctx.runs,
                    name,
                    protocol,
                    methods=style.method_order(extra),
                    width=out.width,
                )[0],
                f"extra_runs_{tag}",
                f"Share of instances with more than one solver run (last "
                f"{metrics.WINDOW} steps), for the methods that run extra solvers.",
            )

        segments = _select(ctx.segments, name, protocol)
        labels = segments.drop_duplicates("segment").set_index("segment")["label"]
        short = dict(zip(labels.index, _short(labels.tolist()), strict=True))
        text += table_env(
            tables.pivot_table(
                segments,
                "segment",
                "closed_gap",
                headers={i: escape(s) for i, s in short.items()},
                higher=True,
            )[1],
            "Closed gap per segment (against the SBS of the whole stream).",
        )
        points = _select(ctx.drift, name, protocol)
        point_labels = points.drop_duplicates("drift_index").set_index("drift_index")
        headers = {
            i: f"\\shortstack{{{escape(short.get(i, ''))}$\\to$"
            f"{escape(short.get(i + 1, ''))}\\\\ \\footnotesize step "
            f"{row.drift_point}}}"
            for i, row in point_labels.iterrows()
        }
        extra_methods = set(extra["method"])
        for metric, higher, fmt, rows, what in (
            (
                "recovery",
                False,
                "f0",
                points,
                f"Steps until the closed gap per window ({metrics.WINDOW} steps) "
                "is back to its level before the drift point and holds for "
                f"{metrics.HOLD} windows (-- = not before the next drift point, "
                "or not measurable).",
            ),
            (
                "peak_regret",
                False,
                "f3" if _maximize(runs) else "f0",
                points,
                "Highest rolling regret between a drift point and the next one.",
            ),
            (
                "extra_run_rate_change",
                None,
                "pct",
                points[points["method"].isin(extra_methods)],
                f"Change of the extra-run rate in percentage points: "
                f"{metrics.WINDOW} steps after minus before the drift point.",
            ),
        ):
            if rows.empty:
                continue
            text += table_env(
                tables.pivot_table(
                    rows,
                    "drift_index",
                    metric,
                    headers=headers,
                    higher=higher,
                    fmt=fmt,
                )[1],
                what,
            )
    return text


def statistics(ctx: Context, out: Out, reference: str = "online-pairwise") -> str:
    """Mean ranks, Friedman and Wilcoxon against ``reference`` on closed gap,
    over the (stream, protocol) cells that every compared method has. Empty if
    there are too few cells or methods (the summary says so)."""
    if not _testable(ctx):
        return ""
    compared, n = _cells(ctx)
    text = "\\section{Statistics}\n"
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
    """Metric definitions, provenance and every run."""
    runs = ctx.runs
    first = runs.iloc[0]
    env = first["environment"] or {}
    final, why = _is_final(runs)
    repo, asf = (_join(runs[c].dropna().str[:10].unique()) for c in COMMITS)
    provenance = {
        "Status": ("Final: " if final else "Pilot: ") + escape(why) + ".",
        "Config": f"{escape(first['config_file'])}, hash "
        f"\\texttt{{{escape(str(first['config_hash'])[:10])}}}",
        "Code": f"experiments \\texttt{{{repo}}}, ASF \\texttt{{{asf}}}",
        "Environment": escape(
            f"Python {env.get('python', '?')}; "
            + ", ".join(f"{k} {v}" for k, v in (env.get("packages") or {}).items())
        ),
        "Run dates": f"{runs['created_utc'].min()[:10]} to "
        f"{runs['created_utc'].max()[:10]}",
    }
    order = {m: i for i, m in enumerate(style.method_order(ctx.table))}
    table = ctx.table.assign(order=ctx.table["method"].map(order)).sort_values(
        ["stream", "protocol", "order", "seed", "stream_seed"]
    )
    keys = [c for c in ("stream", "protocol") if table[c].nunique() > 1]
    seeds = "stream_seed" if table["stream_seed"].nunique() > 1 else None
    header = [k.capitalize() for k in keys] + ["Method", "Seed"]
    header += ["Stream seed"] if seeds else []
    quality = _maximize(runs)
    header += ["Closed gap"] + ([] if quality else ["CPU (h)"])
    header += ["Signals", "Retrains"]
    rows = [
        [getattr(r, k) for k in keys]
        + [r.method, r.seed]
        + ([r.stream_seed] if seeds else [])
        + [tables.number(r.closed_gap, "f2")]
        + ([] if quality else [tables.number(r.solver_cpu, "hours")])
        + [
            tables.number(r.n_drift_signals, "count"),
            tables.number(r.n_retrains, "count"),
        ]
        for r in table.itertuples()
    ]
    n_numbers = 3 if quality else 4
    spec = "l" * (len(header) - n_numbers) + "r" * n_numbers
    definitions = METRIC_DEFINITIONS
    if quality:
        definitions = QUALITY_DEFINITIONS | {
            k: v
            for k, v in METRIC_DEFINITIONS.items()
            if k not in QUALITY_DROPPED and k not in QUALITY_DEFINITIONS
        }
    tabular = (
        simple_tabular(header, rows, spec)
        .replace("\\begin{tabular}", "\\begin{longtable}")
        .replace("\\end{tabular}", "\\end{longtable}")
    )
    return (
        "\\appendix\n\\section{Metrics}\n"
        + _facts(definitions)
        + "\\section{Provenance}\n"
        + _facts(provenance)
        + "\\section{Runs}\n{\\small\n"
        + tabular
        + "}\n"
    )


SECTIONS: dict[str, Section] = {
    "summary": summary,
    "stream": stream,
    "methods": methods,
    "results": results,
    "drift": drift,
    "statistics": statistics,
    "appendix": appendix,
}

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{array,booktabs,graphicx,longtable}
\usepackage[section]{placeins}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage{needspace}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}
% many floats: let them fill pages, and start float-only pages at the top
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.75}
\setcounter{topnumber}{3}
\setcounter{totalnumber}{5}
\makeatletter\setlength{\@fptop}{0pt}\makeatother
% a section heading never sits alone at the bottom of a page
\let\oldsection\section
\renewcommand{\section}{\Needspace{0.3\textheight}\oldsection}
\renewcommand{\arraystretch}{1.1}
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
        + "\\begin{document}\n"
        + f"{{\\LARGE\\bfseries Experiment report: {title}}}\\\\[0.3em]\n"
        + f"{{\\small Generated {date}}}\n\\vspace{{1em}}\n\n"
        + body
        + "\n\\end{document}\n",
        encoding="utf-8",
    )
    return tex


def method_order_from_config(experiment_dir: str | Path, runs: pd.DataFrame) -> list:
    """The method families in the order of the experiment's config file, looked
    up in ``configs/`` next to ``results/`` (empty if it is not found)."""
    names = runs["config_file"].dropna().unique()
    if len(names) != 1:
        return []
    root = Path(experiment_dir).resolve().parent.parent
    for path in (root / "configs" / names[0], Path("configs") / names[0]):
        if path.is_file():
            config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return [m.get("name") for m in config.get("methods") or []]
    return []


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
    style.pin_order(method_order_from_config(args.experiment_dir, ctx.runs))
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
