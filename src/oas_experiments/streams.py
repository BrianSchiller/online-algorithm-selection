"""Stream kinds: turn the ``streams`` entries of a config into ASF streams.

A stream kind is a function ``(params, seed, drift) -> InstanceStream``.
``aslib`` builds streams from ASlib scenarios, with the drift construction in
the ``drift`` block of a stream entry (``{builder, params}``). Bin packing and
SATCOMP are converted to ASlib format and then use the same kind.

ASF is imported inside the builders so that importing this module, and hence
listing the runs of a config, stays cheap.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from asf.online.streams import InstanceStream

    from oas_experiments.config import RunSpec

StreamBuilder = Callable[
    [dict[str, Any], int, "dict[str, Any] | None"], "InstanceStream"
]

STREAM_KINDS: dict[str, StreamBuilder] = {}


def stream_kind(name: str) -> Callable[[StreamBuilder], StreamBuilder]:
    """Register a stream builder under ``kind: <name>``."""

    def register(builder: StreamBuilder) -> StreamBuilder:
        if name in STREAM_KINDS:
            raise ValueError(f"Stream kind {name!r} is already registered.")
        STREAM_KINDS[name] = builder
        return builder

    return register


@stream_kind("aslib")
def _aslib(
    params: dict[str, Any], seed: int, drift: dict[str, Any] | None
) -> InstanceStream:
    """ASlib scenario(s) as a stream (raw runtimes, no PAR penalty).

    ``params``: ``root`` (folder holding the scenarios; relative paths count
    from the working directory), ``scenarios`` (one name or a list; several
    are stacked with ``combine_scenarios``) and optionally ``standardize: n``
    (see :func:`standardize`). Without a ``drift`` block the instances arrive
    in random order. ``drift`` is
    ``{builder: grouped, params: {labels, schedule, ...}}`` or
    ``{builder: sorted, params: {key, ...}}``; see the README.
    """
    from pathlib import Path

    import numpy as np
    from asf.online.streams import (
        InstanceStream,
        combine_scenarios,
        grouped_stream,
        sorted_stream,
    )
    from asf.online.streams.loaders import load_aslib_raw

    try:
        root, names = Path(params["root"]), params["scenarios"]
    except KeyError as exc:
        raise ValueError(f"Stream kind 'aslib' needs the parameter {exc}.") from None
    unknown = sorted(set(params) - {"root", "scenarios", "standardize"})
    if unknown:
        raise ValueError(f"Stream kind 'aslib' has no parameter(s) {unknown}.")
    n_standardize = params.get("standardize")
    names = [names] if isinstance(names, str) else list(names)
    features, performance, cutoff, scenario = combine_scenarios(
        {name: load_aslib_raw(str(root / name)) for name in names}
    )
    if drift is None:
        order = np.random.default_rng(seed).permutation(len(features))
        stream = InstanceStream(features.iloc[order], performance.iloc[order], cutoff)
    elif drift["builder"] == "grouped":
        rest = dict(drift.get("params") or {})
        spec = rest.pop("labels", "scenario")
        labels = _labels(spec, features, performance, scenario)
        stream = grouped_stream(
            features, performance, cutoff, labels, seed=seed, **rest
        )
    elif drift["builder"] == "sorted":
        rest = dict(drift.get("params") or {})
        if "key" not in rest:
            raise ValueError("The sorted builder needs `key`.")
        key = _key(rest.pop("key"), features, performance)
        stream = sorted_stream(features, performance, cutoff, key, seed=seed, **rest)
    else:
        raise ValueError(
            f"Unknown drift builder {drift['builder']!r} (use grouped or sorted)."
        )
    if n_standardize is not None:
        standardize(stream, n_standardize)
    return stream


def standardize(stream: InstanceStream, n: int) -> None:
    """Impute and standardise the features with statistics of the first ``n``
    instances (in place).

    Missing values become the median of the first ``n`` instances (0 if that
    is missing too); then every feature is shifted and scaled by the mean and
    standard deviation of the first ``n`` instances (constant features are
    only shifted). Only features are used, never runtimes. Linear bandits need
    this to get usable contexts; tree-based selectors are unaffected by the
    scaling. On a cold start the selector thereby sees the scale of the first
    ``n`` instances in advance, as in ASF's ``examples/online/4_aslib_sat12.py``.
    """
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= len(stream):
        raise ValueError(
            f"standardize must be an integer between 1 and {len(stream)}, got {n!r}."
        )
    features = stream.features
    features = features.fillna(features.iloc[:n].median().fillna(0.0))
    head = features.iloc[:n]
    std = head.std(ddof=0)
    stream.features = (features - head.mean()) / std.where(std > 0, 1.0)


def _labels(spec: Any, features: Any, performance: Any, scenario: Any) -> Any:
    """Group labels of a ``grouped`` builder."""
    import pandas as pd
    from asf.online.streams import cluster_labels

    if spec == "scenario":
        return scenario
    if spec == "best_algorithm":
        return performance.idxmin(axis=1)  # ties: first algorithm
    if isinstance(spec, dict) and set(spec) == {"clusters"}:
        # fixed seed: cluster ids stay the same across stream seeds
        return cluster_labels(features, spec["clusters"], seed=0)
    if isinstance(spec, dict) and set(spec) == {"feature", "quantiles"}:
        return pd.qcut(features.eval(spec["feature"]), spec["quantiles"], labels=False)
    raise ValueError(
        "labels must be scenario, best_algorithm, {clusters: n} or "
        f"{{feature: <expr>, quantiles: n}}, got {spec!r}."
    )


def _key(spec: Any, features: Any, performance: Any) -> Any:
    """Sort key of a ``sorted`` builder."""
    if isinstance(spec, dict) and set(spec) == {"feature"}:
        return features.eval(spec["feature"])
    if isinstance(spec, dict) and set(spec) <= {"runtime", "agg"} and "runtime" in spec:
        algos = spec["runtime"]
        cols = performance.columns if algos == "all" else list(algos)
        return performance[cols].agg(spec.get("agg", "max"), axis=1)
    raise ValueError(
        "key must be {feature: <column or expression>} or "
        f"{{runtime: [algorithms] | all, agg: max|min|mean}}, got {spec!r}."
    )


def build_stream(run: RunSpec) -> InstanceStream:
    """Build the full stream of a run (warm-start prefix included)."""
    builder = STREAM_KINDS[run.stream_kind]
    stream = builder(dict(run.stream_params), run.stream_seed, run.stream_drift)
    stream.name = run.stream
    return stream
