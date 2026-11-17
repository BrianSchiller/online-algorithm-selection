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


def _no_drift_block(kind: str, drift: dict[str, Any] | None) -> None:
    if drift is not None:
        raise ValueError(
            f"Stream kind {kind!r} constructs its own drift; remove the `drift` block."
        )


@stream_kind("synthetic_scenario")
def _synthetic_scenario(
    params: dict[str, Any], seed: int, drift: dict[str, Any] | None
) -> InstanceStream:
    """``asf.online.streams.make_synthetic_scenario(**params, random_state=seed)``."""
    from asf.online.streams import make_synthetic_scenario

    _no_drift_block("synthetic_scenario", drift)
    return make_synthetic_scenario(**params, random_state=seed)


@stream_kind("synthetic")
def _synthetic(
    params: dict[str, Any], seed: int, drift: dict[str, Any] | None
) -> InstanceStream:
    """``asf.online.streams.make_synthetic_stream`` with explicit regimes and phases.

    ``regimes`` maps a name to ``{mean, scale}``, ``phases`` is a list of
    ``{regime, length, transition_length}`` and ``performance_model`` is
    ``{intercepts, coefficients}``; all other keys are passed through.
    """
    from asf.online.streams import (
        FeatureRegime,
        PerformanceModelConfig,
        StreamPhase,
        make_synthetic_stream,
    )

    _no_drift_block("synthetic", drift)
    rest = dict(params)
    try:
        regimes = {k: FeatureRegime(**v) for k, v in rest.pop("regimes").items()}
        phases = [StreamPhase(**p) for p in rest.pop("phases")]
        model = PerformanceModelConfig(**rest.pop("performance_model"))
        cutoff = rest.pop("cutoff")
    except KeyError as exc:
        raise ValueError(
            f"Stream kind 'synthetic' needs the parameter {exc}."
        ) from None
    return make_synthetic_stream(
        regimes, phases, model, cutoff, random_state=seed, **rest
    )


@stream_kind("aslib")
def _aslib(
    params: dict[str, Any], seed: int, drift: dict[str, Any] | None
) -> InstanceStream:
    """ASlib scenario(s) as a stream (raw runtimes, no PAR penalty).

    ``params``: ``root`` (folder holding the scenarios; relative paths count
    from the working directory) and ``scenarios`` (one name or a list; several
    are stacked with ``combine_scenarios``). Without a ``drift`` block the
    instances arrive in random order. ``drift`` is
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
    names = [names] if isinstance(names, str) else list(names)
    features, performance, cutoff, scenario = combine_scenarios(
        {name: load_aslib_raw(str(root / name)) for name in names}
    )
    if drift is None:
        order = np.random.default_rng(seed).permutation(len(features))
        return InstanceStream(features.iloc[order], performance.iloc[order], cutoff)
    rest = dict(drift.get("params") or {})
    if drift["builder"] == "grouped":
        spec = rest.pop("labels", "scenario")
        labels = _labels(spec, features, performance, scenario)
        return grouped_stream(features, performance, cutoff, labels, seed=seed, **rest)
    if drift["builder"] == "sorted":
        if "key" not in rest:
            raise ValueError("The sorted builder needs `key`.")
        key = _key(rest.pop("key"), features, performance)
        return sorted_stream(features, performance, cutoff, key, seed=seed, **rest)
    raise ValueError(
        f"Unknown drift builder {drift['builder']!r} (use grouped or sorted)."
    )


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
