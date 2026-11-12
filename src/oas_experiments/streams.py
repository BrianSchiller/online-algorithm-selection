"""Stream kinds: turn the ``streams`` entries of a config into ASF streams.

A stream kind is a function ``(params, seed, drift) -> InstanceStream``.
Dataset-backed kinds (ASlib scenario + drift construction, bin packing,
SATCOMP) are registered here by later work packages; the ``drift`` block of a
stream entry (``{builder, params}``) is reserved for them.

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


def build_stream(run: RunSpec) -> InstanceStream:
    """Build the full stream of a run (warm-start prefix included)."""
    builder = STREAM_KINDS[run.stream_kind]
    stream = builder(dict(run.stream_params), run.stream_seed, run.stream_drift)
    stream.name = run.stream
    return stream
