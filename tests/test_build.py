from __future__ import annotations

import pytest

from oas_experiments.build import (
    BuildError,
    build,
    build_selector,
    import_object,
    resolve_placeholders,
)
from oas_experiments.config import parse_config
from oas_experiments.streams import build_stream


def test_import_object():
    from collections import OrderedDict

    assert import_object("collections.OrderedDict") is OrderedDict
    with pytest.raises(BuildError, match="Cannot import"):
        import_object("collections.Nope")
    with pytest.raises(BuildError, match="dotted"):
        import_object("nodots")


def test_resolve_placeholders_keeps_types():
    context = {"seed": 7, "stream.cutoff": 100.0}
    spec = {"a": "${seed}", "b": ["${stream.cutoff}", "run-${seed}"], "c": 3}
    assert resolve_placeholders(spec, context) == {
        "a": 7,
        "b": [100.0, "run-7"],
        "c": 3,
    }
    with pytest.raises(BuildError, match="Unknown placeholder"):
        resolve_placeholders("${sede}", context)


def test_build_class_type_and_passthrough():
    from fractions import Fraction

    built = build(
        {
            "instance": {"class": "fractions.Fraction", "params": {"numerator": 3}},
            "cls": {"type": "fractions.Fraction"},
            "plain": {"x": [1, {"class": "fractions.Fraction"}]},
        }
    )
    assert built["instance"] == Fraction(3)
    assert built["cls"] is Fraction
    assert built["plain"] == {"x": [1, Fraction(0)]}
    with pytest.raises(BuildError, match="unexpected key"):
        build({"class": "fractions.Fraction", "parms": {}})
    with pytest.raises(BuildError, match="Cannot construct"):
        build({"class": "fractions.Fraction", "params": {"nope": 1}})


def test_build_stream_and_selectors(minimal_config):
    from asf.online.selectors import DiscountedTSSelector, StaticOnlineSelector
    from sklearn.tree import DecisionTreeClassifier

    cfg = parse_config(minimal_config)
    run = cfg.get("sudden-s0__warm20__static__seed1")
    stream = build_stream(run)
    assert len(stream) == 80
    assert stream.name == "sudden"
    assert stream.cutoff == 100.0
    assert len(stream.drift_points) == 1

    static = build_selector(run, stream)
    assert isinstance(static, StaticOnlineSelector)
    assert static.random_state == 1
    assert static.selector.random_state == 1
    # ASF wraps plain sklearn classes in a partial of its SklearnWrapper
    model_class = static.selector.model_class
    assert DecisionTreeClassifier in getattr(model_class, "args", (model_class,))

    bandit = build_selector(cfg.get("sudden-s0__cold__ds-ts__seed0"), stream)
    assert isinstance(bandit, DiscountedTSSelector)
    assert bandit.random_state == 0
    assert bandit.budget == 100.0  # ${stream.cutoff}


def test_stream_seed_controls_the_stream(minimal_config):
    minimal_config["streams"][0]["seeds"] = [0, 1]
    cfg = parse_config(minimal_config)
    s0 = build_stream(cfg.get("sudden-s0__cold__ds-ts__seed0"))
    s0_again = build_stream(cfg.get("sudden-s0__cold__ds-ts__seed1"))
    s1 = build_stream(cfg.get("sudden-s1__cold__ds-ts__seed0"))
    assert s0.features.equals(s0_again.features)
    assert not s0.features.equals(s1.features)


def test_explicit_synthetic_stream(minimal_config):
    minimal_config["streams"][0] = {
        "name": "explicit",
        "kind": "synthetic",
        "params": {
            "regimes": {"A": {"mean": [0.0, 0.0]}, "B": {"mean": [2.0, 0.0]}},
            "phases": [
                {"regime": "A", "length": 30},
                {"regime": "B", "length": 30, "transition_length": 10},
            ],
            "performance_model": {
                "intercepts": [1.0, 1.2],
                "coefficients": [[0.5, -0.5], [0.0, 0.0]],
            },
            "cutoff": 50.0,
            "performance_noise": 0.1,
        },
    }
    stream = build_stream(parse_config(minimal_config).runs[0])
    assert len(stream) == 60
    assert stream.drift_points == (30,)
    assert stream.transition_intervals == ((30, 40),)


def test_synthetic_streams_reject_a_drift_block(minimal_config):
    minimal_config["streams"][0]["drift"] = {"builder": "abrupt_drift"}
    with pytest.raises(ValueError, match="remove the `drift` block"):
        build_stream(parse_config(minimal_config).runs[0])
