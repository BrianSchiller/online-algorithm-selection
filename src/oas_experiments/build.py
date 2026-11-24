"""Turn the ``selector`` block of a config into a Python object.

Four forms are understood anywhere inside ``params``:

``{class: pkg.mod.Name, params: {...}}``
    an instance, ``Name(**params)``, with the params built recursively;
``{type: pkg.mod.Name}``
    the class or function itself (e.g. ``model_class``);
``{type: pkg.mod.Name, params: {...}}``
    the class with preset arguments, ``functools.partial(Name, **params)``
    (e.g. a ``model_class`` with ``n_estimators: 50``);
``"${seed}"`` / ``"${stream.cutoff}"``
    the selector seed of the run / the cutoff of the built stream.

Everything else is passed through unchanged.
"""

from __future__ import annotations

import functools
import importlib
import re
from typing import TYPE_CHECKING, Any

from oas_experiments.streams import MAXIMIZE_KINDS

if TYPE_CHECKING:
    from asf.online.base import AbstractOnlineSelector
    from asf.online.streams import InstanceStream

    from oas_experiments.config import RunSpec

_PLACEHOLDER = re.compile(r"\$\{([^}]*)\}")


class BuildError(ValueError):
    """A config block could not be turned into an object."""


def import_object(path: str) -> Any:
    """Import ``pkg.mod.Name`` and return ``Name``."""
    module_name, _, attribute = path.rpartition(".")
    if not module_name:
        raise BuildError(f"{path!r} is not a dotted import path.")
    try:
        return getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise BuildError(f"Cannot import {path!r}: {exc}") from exc


def resolve_placeholders(value: Any, context: dict[str, Any]) -> Any:
    """Replace ``${key}`` by ``context[key]`` everywhere in a nested structure.

    A string that is exactly one placeholder takes the value's type; a
    placeholder inside a longer string is formatted into it.
    """
    if isinstance(value, dict):
        return {k: resolve_placeholders(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_placeholders(v, context) for v in value]
    if not isinstance(value, str):
        return value

    def lookup(key: str) -> Any:
        if key not in context:
            raise BuildError(
                f"Unknown placeholder ${{{key}}}; available: "
                + ", ".join(f"${{{k}}}" for k in sorted(context))
            )
        return context[key]

    whole = _PLACEHOLDER.fullmatch(value)
    if whole:
        return lookup(whole.group(1))
    return _PLACEHOLDER.sub(lambda m: str(lookup(m.group(1))), value)


def build(spec: Any) -> Any:
    """Recursively build ``{class, params}``, ``{type}`` and ``{type, params}``."""
    if isinstance(spec, list):
        return [build(item) for item in spec]
    if not isinstance(spec, dict):
        return spec
    if "class" in spec:
        extra = sorted(set(spec) - {"class", "params"})
        if extra:
            raise BuildError(
                f"A {{class, params}} block has unexpected key(s) {extra}: {spec!r}"
            )
        cls = import_object(spec["class"])
        params = build(spec.get("params") or {})
        try:
            return cls(**params)
        except TypeError as exc:
            raise BuildError(f"Cannot construct {spec['class']}: {exc}") from exc
    if set(spec) == {"type"}:
        return import_object(spec["type"])
    if set(spec) == {"type", "params"}:
        params = build(spec["params"] or {})
        if not isinstance(params, dict):
            raise BuildError(f"{{type, params}}: params must be a mapping: {spec!r}")
        return functools.partial(import_object(spec["type"]), **params)
    return {key: build(item) for key, item in spec.items()}


def build_selector(run: RunSpec, stream: InstanceStream) -> AbstractOnlineSelector:
    """Build the unfitted selector of a run; ``random_state`` is the run's seed."""
    context = {"seed": run.seed, "stream.cutoff": stream.cutoff}
    params = build(resolve_placeholders(run.selector["params"], context))
    cls = import_object(run.selector["class"])
    try:
        selector = cls(**params, random_state=run.seed)
    except TypeError as exc:
        raise BuildError(f"Cannot construct {run.selector['class']}: {exc}") from exc
    # a minimising selector on quality data would silently pick the worst algorithm
    if run.stream_kind in MAXIMIZE_KINDS and not selector.maximize:
        raise BuildError(
            f"Stream kind {run.stream_kind!r} holds qualities to maximise; "
            f"set `maximize: true` in the params of method {run.method!r}."
        )
    return selector
