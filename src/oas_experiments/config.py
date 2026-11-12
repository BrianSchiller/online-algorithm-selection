"""Experiment configs: one YAML file -> a grid of single runs.

A config describes a grid over streams (x stream seeds), protocols, methods
and selector seeds. :func:`load_config` validates the file and expands the grid
into :class:`RunSpec` objects, each with a stable, human-readable run id::

    <stream>-s<stream seed>__<protocol>__<method>__seed<selector seed>

Run ids are built from names only, so they do not change when methods, streams
or seeds are added to a config. This module does not import ASF, so listing the
runs of a config is fast.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from oas_experiments.streams import STREAM_KINDS

RESULT_MODES = ("primary", "best_of_scheduled")
PROTOCOL_TYPES = ("warm", "cold")

# Letters and digits, single '.', '_' or '-' in between. This keeps names safe
# as directory names on every platform and keeps '__' free as the separator.
_NAME = re.compile(r"^[A-Za-z0-9]+([._-][A-Za-z0-9]+)*$")
_SEED_PLACEHOLDER = "${seed}"


class ConfigError(ValueError):
    """The experiment config is invalid."""


@dataclass(frozen=True)
class RunSpec:
    """One cell of the grid: everything needed for a single run."""

    experiment: str
    run_id: str
    method: str
    selector: dict[str, Any]
    stream: str
    stream_kind: str
    stream_params: dict[str, Any]
    stream_drift: dict[str, Any] | None
    stream_seed: int
    protocol: str
    protocol_type: str
    n_warmstart: int
    seed: int
    par_factor: float
    result_mode: str

    def to_dict(self) -> dict[str, Any]:
        """The resolved config of this run, as stored in the result metadata.

        ``${seed}`` is already substituted. ``${stream.cutoff}`` stays symbolic
        because its value is only known once the stream is built; the value
        itself is recorded by ASF as ``meta.stream.cutoff``.
        """
        return jsonable(
            {
                "experiment": self.experiment,
                "run_id": self.run_id,
                "method": {"name": self.method, "selector": self.selector},
                "stream": {
                    "name": self.stream,
                    "kind": self.stream_kind,
                    "seed": self.stream_seed,
                    "params": self.stream_params,
                    "drift": self.stream_drift,
                },
                "protocol": {
                    "name": self.protocol,
                    "type": self.protocol_type,
                    "n_warmstart": self.n_warmstart,
                },
                "seed": self.seed,
                "evaluation": {
                    "par_factor": self.par_factor,
                    "result_mode": self.result_mode,
                },
            }
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """A validated config and its expanded runs (in a deterministic order)."""

    name: str
    path: Path | None
    raw: dict[str, Any]
    runs: tuple[RunSpec, ...]
    _by_id: dict[str, RunSpec] = field(repr=False, compare=False, default_factory=dict)

    @property
    def run_ids(self) -> list[str]:
        return [run.run_id for run in self.runs]

    @property
    def config_hash(self) -> str:
        """SHA-256 of the parsed config (independent of formatting and comments)."""
        return hashlib.sha256(canonical_json(self.raw).encode()).hexdigest()

    def get(self, run_id: str) -> RunSpec:
        try:
            return self._by_id[run_id]
        except KeyError:
            raise ConfigError(
                f"Unknown run id {run_id!r} for experiment {self.name!r}. "
                "List the valid ids with: python -m oas_experiments.list <config>"
            ) from None


# --------------------------------------------------------------------- helpers
def jsonable(value: Any) -> Any:
    """Copy ``value`` into plain JSON types; non-finite floats become ``None``.

    ASF writes non-finite numbers as null, so the same rule is applied here to
    keep a stored config comparable with a freshly expanded one.
    """
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ConfigError(f"Config values must be plain YAML types, got {value!r}.")


def canonical_json(value: Any) -> str:
    """Deterministic JSON text, used for hashing and comparing configs."""
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"))


def _substitute_seed(value: Any, seed: int) -> Any:
    if isinstance(value, dict):
        return {k: _substitute_seed(v, seed) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_seed(v, seed) for v in value]
    if value == _SEED_PLACEHOLDER:
        return seed
    if isinstance(value, str) and _SEED_PLACEHOLDER in value:
        return value.replace(_SEED_PLACEHOLDER, str(seed))
    return value


def _mapping(value: Any, where: str, allowed: set[str], required: set[str]) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a mapping.")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(
            f"{where} has unknown key(s) {unknown}; allowed: {sorted(allowed)}."
        )
    missing = sorted(required - set(value))
    if missing:
        raise ConfigError(f"{where} is missing required key(s) {missing}.")
    return value


def _name(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _NAME.match(value):
        raise ConfigError(
            f"{where}: {value!r} is not a valid name. Use letters and digits, "
            "with single '.', '_' or '-' in between."
        )
    return value


def _entries(value: Any, where: str) -> list:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{where} must be a non-empty list.")
    return value


def _unique(names: list[Any], where: str) -> None:
    duplicates = sorted({str(n) for n in names if names.count(n) > 1})
    if duplicates:
        raise ConfigError(f"{where} contains duplicates: {duplicates}.")


def _seeds(value: Any, where: str) -> list[int]:
    seeds = _entries(value, where)
    if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ConfigError(f"{where} must be non-negative integers.")
    _unique(seeds, where)
    return seeds


def _spec(value: Any, where: str) -> dict[str, Any]:
    spec = _mapping(value, where, {"class", "params"}, {"class"})
    if not isinstance(spec["class"], str) or "." not in spec["class"]:
        raise ConfigError(f"{where}.class must be a dotted import path.")
    params = spec.get("params") or {}
    if not isinstance(params, dict):
        raise ConfigError(f"{where}.params must be a mapping.")
    return {"class": spec["class"], "params": params}


# ------------------------------------------------------------------ validation
def parse_config(raw: Any, path: Path | None = None) -> ExperimentConfig:
    """Validate a parsed config and expand its grid."""
    top = _mapping(
        raw,
        "config",
        {"name", "evaluation", "seeds", "protocols", "streams", "methods"},
        {"name", "seeds", "protocols", "streams", "methods"},
    )
    name = _name(top["name"], "name")

    evaluation = _mapping(
        top.get("evaluation") or {}, "evaluation", {"par_factor", "result_mode"}, set()
    )
    par_factor = evaluation.get("par_factor", 10.0)
    if isinstance(par_factor, bool) or not isinstance(par_factor, (int, float)):
        raise ConfigError("evaluation.par_factor must be a number.")
    if not par_factor >= 1:
        raise ConfigError("evaluation.par_factor must be >= 1.")
    result_mode = evaluation.get("result_mode", "best_of_scheduled")
    if result_mode not in RESULT_MODES:
        raise ConfigError(f"evaluation.result_mode must be one of {RESULT_MODES}.")

    seeds = _seeds(top["seeds"], "seeds")

    protocols: list[dict[str, Any]] = []
    for i, entry in enumerate(_entries(top["protocols"], "protocols")):
        where = f"protocols[{i}]"
        p = _mapping(entry, where, {"name", "type", "n_warmstart"}, {"name", "type"})
        if p["type"] not in PROTOCOL_TYPES:
            raise ConfigError(f"{where}.type must be one of {PROTOCOL_TYPES}.")
        n_warm = p.get("n_warmstart")
        if p["type"] == "warm":
            if isinstance(n_warm, bool) or not isinstance(n_warm, int) or n_warm < 1:
                raise ConfigError(
                    f"{where}: a warm protocol needs an integer n_warmstart >= 1."
                )
        elif n_warm not in (None, 0):
            raise ConfigError(f"{where}: a cold protocol has no n_warmstart.")
        protocols.append(
            {
                "name": _name(p["name"], f"{where}.name"),
                "type": p["type"],
                "n_warmstart": n_warm if p["type"] == "warm" else 0,
            }
        )
    _unique([p["name"] for p in protocols], "protocols")
    protocol_names = [p["name"] for p in protocols]

    streams: list[dict[str, Any]] = []
    for i, entry in enumerate(_entries(top["streams"], "streams")):
        where = f"streams[{i}]"
        s = _mapping(
            entry, where, {"name", "kind", "seeds", "params", "drift"}, {"name", "kind"}
        )
        if s["kind"] not in STREAM_KINDS:
            raise ConfigError(
                f"{where}.kind {s['kind']!r} is unknown; "
                f"available: {sorted(STREAM_KINDS)}."
            )
        params = s.get("params") or {}
        if not isinstance(params, dict):
            raise ConfigError(f"{where}.params must be a mapping.")
        drift = s.get("drift")
        if drift is not None:
            drift = _mapping(
                drift, f"{where}.drift", {"builder", "params"}, {"builder"}
            )
        streams.append(
            {
                "name": _name(s["name"], f"{where}.name"),
                "kind": s["kind"],
                "seeds": _seeds(s.get("seeds", [0]), f"{where}.seeds"),
                "params": params,
                "drift": drift,
            }
        )
    _unique([s["name"] for s in streams], "streams")

    methods: list[dict[str, Any]] = []
    for i, entry in enumerate(_entries(top["methods"], "methods")):
        where = f"methods[{i}]"
        m = _mapping(
            entry, where, {"name", "selector", "protocols"}, {"name", "selector"}
        )
        selector = _spec(m["selector"], f"{where}.selector")
        if "random_state" in selector["params"]:
            raise ConfigError(
                f"{where}.selector.params: random_state is set from the `seeds` "
                "axis; remove it. Use ${seed} to seed nested components."
            )
        allowed = m.get("protocols", protocol_names)
        if not isinstance(allowed, list) or not allowed:
            raise ConfigError(f"{where}.protocols must be a non-empty list.")
        unknown = [p for p in allowed if p not in protocol_names]
        if unknown:
            raise ConfigError(
                f"{where}.protocols refers to undefined protocol(s) {unknown}."
            )
        _unique(allowed, f"{where}.protocols")
        methods.append(
            {
                "name": _name(m["name"], f"{where}.name"),
                "selector": selector,
                "protocols": allowed,
            }
        )
    _unique([m["name"] for m in methods], "methods")

    runs: list[RunSpec] = []
    for s in streams:
        for stream_seed in s["seeds"]:
            for p in protocols:
                for m in methods:
                    if p["name"] not in m["protocols"]:
                        continue
                    for seed in seeds:
                        run_id = (
                            f"{s['name']}-s{stream_seed}__{p['name']}"
                            f"__{m['name']}__seed{seed}"
                        )
                        runs.append(
                            RunSpec(
                                experiment=name,
                                run_id=run_id,
                                method=m["name"],
                                selector=jsonable(
                                    _substitute_seed(m["selector"], seed)
                                ),
                                stream=s["name"],
                                stream_kind=s["kind"],
                                stream_params=jsonable(s["params"]),
                                stream_drift=jsonable(s["drift"]),
                                stream_seed=stream_seed,
                                protocol=p["name"],
                                protocol_type=p["type"],
                                n_warmstart=p["n_warmstart"],
                                seed=seed,
                                par_factor=float(par_factor),
                                result_mode=result_mode,
                            )
                        )
    if not runs:
        raise ConfigError("The config expands to zero runs.")
    return ExperimentConfig(
        name=name,
        path=path,
        raw=jsonable(top),
        runs=tuple(runs),
        _by_id={run.run_id: run for run in runs},
    )


def load_config(path: str | Path) -> ExperimentConfig:
    """Read, validate and expand an experiment config file."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    try:
        return parse_config(raw, path=path)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from None
