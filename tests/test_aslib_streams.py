"""The ``aslib`` stream kind, with the ASlib reader replaced by toy data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oas_experiments.config import parse_config
from oas_experiments.streams import build_stream

SIZES = {"HAND": 10, "INDU": 12, "RAND": 8}


@pytest.fixture(autouse=True)
def fake_aslib(monkeypatch):
    import asf.online.streams.loaders as loaders

    def load(path: str):
        name = path.replace("\\", "/").rsplit("/", 1)[-1]
        n, rng = SIZES[name], np.random.default_rng(len(name) + SIZES[name])
        idx = pd.Index([f"{name}/{i}" for i in range(n)])
        features = pd.DataFrame(
            {"nvars": rng.uniform(1, 9, n), "nclauses": rng.uniform(1, 9, n)}, idx
        )
        performance = pd.DataFrame(rng.uniform(1, 9, (n, 2)), idx, ["a", "b"])
        return features, performance, 100.0

    monkeypatch.setattr(loaders, "load_aslib_raw", load)


def _stream(minimal_config, drift=None, scenarios=("HAND", "INDU", "RAND"), seed=0):
    entry = {
        "name": "sat",
        "kind": "aslib",
        "seeds": [seed],
        "params": {"root": "aslib_data", "scenarios": list(scenarios)},
    }
    if drift is not None:
        entry["drift"] = drift
    minimal_config["streams"] = [entry]
    return build_stream(parse_config(minimal_config).runs[0])


def _source(stream) -> list[str]:
    return [i.split("/")[0] for i in stream.features.index]


def test_no_drift_block_is_a_shuffle(minimal_config):
    stream = _stream(minimal_config)
    assert len(stream) == 30 and stream.drift_points == () and stream.cutoff == 100.0
    assert stream.name == "sat"
    assert list(stream.features.index) != sorted(stream.features.index)


def test_halves_of_each_scenario(minimal_config):
    drift = {
        "builder": "grouped",
        "params": {"schedule": ["HAND", "INDU", "RAND"] * 2},
    }
    stream = _stream(minimal_config, drift)
    assert _source(stream) == (["HAND"] * 5 + ["INDU"] * 6 + ["RAND"] * 4) * 2
    assert stream.drift_points == (5, 11, 15, 20, 26)


def test_mixed_segment_from_yaml_list(minimal_config):
    drift = {"builder": "grouped", "params": {"schedule": [["HAND", "INDU"], "RAND"]}}
    stream = _stream(minimal_config, drift)
    assert set(_source(stream)[:22]) == {"HAND", "INDU"}
    assert stream.drift_points == (22,)


@pytest.mark.parametrize(
    "labels", ["best_algorithm", {"clusters": 2}, {"feature": "nvars", "quantiles": 2}]
)
def test_other_labels(minimal_config, labels):
    drift = {"builder": "grouped", "params": {"labels": labels, "schedule": [0, 1]}}
    if labels == "best_algorithm":
        drift["params"]["schedule"] = ["a", "b"]
    stream = _stream(minimal_config, drift)
    assert len(stream) == 30 and len(stream.drift_points) == 1


def test_sorted_by_feature_expression_and_runtime(minimal_config):
    drift = {"builder": "sorted", "params": {"key": {"feature": "nclauses / nvars"}}}
    stream = _stream(minimal_config, drift)
    ratio = stream.features.nclauses / stream.features.nvars
    assert (np.diff(ratio) >= 0).all() and stream.transition_intervals == ((0, 30),)
    drift = {
        "builder": "sorted",
        "params": {"key": {"runtime": "all"}, "ascending": False},
    }
    stream = _stream(minimal_config, drift)
    assert (np.diff(stream.performance.max(axis=1)) <= 0).all()


def test_bad_blocks_are_rejected(minimal_config):
    with pytest.raises(ValueError, match="Unknown drift builder"):
        _stream(minimal_config, {"builder": "abrupt_drift"})
    with pytest.raises(ValueError, match="labels must be"):
        bad = {"builder": "grouped", "params": {"labels": "x", "schedule": [0]}}
        _stream(minimal_config, bad)
    with pytest.raises(ValueError, match="needs `key`"):
        _stream(minimal_config, {"builder": "sorted"})


def test_standardize_uses_the_first_instances_only(minimal_config):
    raw = _stream(minimal_config, seed=3)
    minimal_config["streams"][0]["params"]["standardize"] = 10
    stream = build_stream(parse_config(minimal_config).runs[0])
    assert list(stream.features.index) == list(raw.features.index)
    head = stream.features.iloc[:10]
    assert np.allclose(head.mean(), 0) and np.allclose(head.std(ddof=0), 1)
    assert not np.allclose(stream.features.iloc[10:].mean(), 0)
    assert stream.performance.equals(raw.performance)


def test_standardize_imputes_and_keeps_constant_features(minimal_config):
    from oas_experiments.streams import standardize

    stream = _stream(minimal_config)
    stream.features.iloc[0, 0] = np.nan
    stream.features["const"] = 4.0
    standardize(stream, 5)
    assert not stream.features.isna().any().any()
    assert (stream.features["const"] == 0).all()


@pytest.mark.parametrize("bad", [0, 31, 2.5, True])
def test_bad_standardize(minimal_config, bad):
    minimal_config["streams"] = [
        {
            "name": "sat",
            "kind": "aslib",
            "params": {"root": "x", "scenarios": ["HAND"], "standardize": bad},
        }
    ]
    with pytest.raises(ValueError, match="standardize"):
        build_stream(parse_config(minimal_config).runs[0])


def test_unknown_aslib_parameter(minimal_config):
    minimal_config["streams"] = [
        {
            "name": "sat",
            "kind": "aslib",
            "params": {"root": "x", "scenarios": ["HAND"], "scale": 1},
        }
    ]
    with pytest.raises(ValueError, match="scale"):
        build_stream(parse_config(minimal_config).runs[0])
