from __future__ import annotations

import copy

import pytest
import yaml

from conftest import EXAMPLE_CONFIG
from oas_experiments.config import (
    ConfigError,
    canonical_json,
    jsonable,
    load_config,
    parse_config,
)


def test_grid_expansion_respects_protocol_restrictions(minimal_config):
    cfg = parse_config(minimal_config)
    assert cfg.run_ids == [
        "sudden-s0__warm20__static__seed0",
        "sudden-s0__warm20__static__seed1",
        "sudden-s0__warm20__ds-ts__seed0",
        "sudden-s0__warm20__ds-ts__seed1",
        "sudden-s0__cold__ds-ts__seed0",
        "sudden-s0__cold__ds-ts__seed1",
    ]
    assert not any("cold__static" in run_id for run_id in cfg.run_ids)


def test_run_spec_fields(minimal_config):
    cfg = parse_config(minimal_config)
    warm = cfg.get("sudden-s0__warm20__static__seed1")
    assert (warm.protocol_type, warm.n_warmstart, warm.seed) == ("warm", 20, 1)
    assert (warm.par_factor, warm.result_mode) == (10.0, "best_of_scheduled")
    # ${seed} is substituted at expansion, ${stream.cutoff} stays symbolic
    assert warm.selector["params"]["selector"]["params"]["random_state"] == 1
    cold = cfg.get("sudden-s0__cold__ds-ts__seed0")
    assert (cold.protocol_type, cold.n_warmstart) == ("cold", 0)
    assert cold.selector["params"]["budget"] == "${stream.cutoff}"


def test_run_ids_are_stable_when_the_grid_grows(minimal_config):
    before = set(parse_config(minimal_config).run_ids)
    grown = copy.deepcopy(minimal_config)
    grown["seeds"].append(2)
    grown["streams"][0]["seeds"].append(1)
    grown["methods"].insert(
        0,
        {
            "name": "sw-ts",
            "selector": {"class": "asf.online.selectors.SlidingWindowTSSelector"},
        },
    )
    after = parse_config(grown)
    assert before < set(after.run_ids)
    assert len(after.runs) == 2 * (2 * 3 + 3 + 2 * 3)
    assert len(set(after.run_ids)) == len(after.runs)


def test_to_dict_is_json_safe_and_complete(minimal_config):
    minimal_config["streams"][0]["params"]["cutoff"] = float("inf")
    run = parse_config(minimal_config).get("sudden-s0__cold__ds-ts__seed1")
    resolved = run.to_dict()
    assert resolved["stream"]["params"]["cutoff"] is None  # as ASF writes it
    assert resolved["run_id"] == run.run_id
    assert resolved["method"] == {"name": "ds-ts", "selector": run.selector}
    assert resolved["protocol"] == {"name": "cold", "type": "cold", "n_warmstart": 0}
    assert resolved["evaluation"] == {
        "par_factor": 10.0,
        "result_mode": "best_of_scheduled",
    }
    assert canonical_json(resolved) == canonical_json(jsonable(resolved))


def test_config_hash_ignores_formatting(tmp_path, minimal_config):
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text(yaml.safe_dump(minimal_config, sort_keys=True))
    b.write_text("# a comment\n" + yaml.safe_dump(minimal_config, sort_keys=False))
    assert load_config(a).config_hash == load_config(b).config_hash
    minimal_config["seeds"] = [0]
    a.write_text(yaml.safe_dump(minimal_config))
    assert load_config(a).config_hash != load_config(b).config_hash


def _broken(config: dict, edit) -> dict:
    config = copy.deepcopy(config)
    edit(config)
    return config


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda c: c.update(typo=1), "unknown key"),
        (lambda c: c.pop("methods"), "missing required"),
        (lambda c: c.update(name="has space"), "not a valid name"),
        (lambda c: c["methods"][0].update(name="a__b"), "not a valid name"),
        (lambda c: c.update(seeds=[0, 0]), "duplicates"),
        (lambda c: c.update(seeds=[]), "non-empty list"),
        (lambda c: c.update(seeds=[0.5]), "non-negative integers"),
        (lambda c: c["protocols"][0].pop("n_warmstart"), "n_warmstart >= 1"),
        (lambda c: c["protocols"][1].update(n_warmstart=5), "no n_warmstart"),
        (lambda c: c["protocols"][1].update(type="lukewarm"), "must be one of"),
        (lambda c: c["protocols"][1].update(name="warm20"), "duplicates"),
        (lambda c: c["streams"][0].update(kind="nope"), "is unknown"),
        (lambda c: c["methods"][0].update(protocols=["hot"]), "undefined protocol"),
        (lambda c: c["methods"][1].update(name="static"), "duplicates"),
        (lambda c: c["methods"][1]["selector"].update({"class": "NoDots"}), "dotted"),
        (
            lambda c: c["methods"][1]["selector"]["params"].update(random_state=3),
            "random_state is set from",
        ),
        (lambda c: c.update(evaluation={"result_mode": "best"}), "result_mode"),
        (lambda c: c.update(evaluation={"par_factor": 0}), "par_factor"),
        (
            lambda c: (
                c["methods"][1].update(protocols=["warm20"])
                or c.update(protocols=[c["protocols"][1]])
            ),
            "undefined protocol",
        ),
    ],
)
def test_invalid_configs_are_rejected(minimal_config, edit, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(_broken(minimal_config, edit))


def test_unknown_run_id(minimal_config):
    with pytest.raises(ConfigError, match="Unknown run id"):
        parse_config(minimal_config).get("nope")


def test_load_config_reports_the_file(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\n")
    with pytest.raises(ConfigError, match="bad.yaml"):
        load_config(path)


def test_example_config_expands_to_15_runs():
    cfg = load_config(EXAMPLE_CONFIG)
    assert cfg.name == "example_synthetic_sudden"
    assert len(cfg.runs) == 15
    by_method: dict[str, set[str]] = {}
    for run in cfg.runs:
        by_method.setdefault(run.method, set()).add(run.protocol_type)
    assert by_method == {
        "static-pairwise": {"warm"},
        "rook-pairwise": {"warm", "cold"},
        "ds-ts": {"warm", "cold"},
    }
    assert {run.seed for run in cfg.runs} == {0, 1, 2}


def _swept(minimal_config, sweep):
    method = minimal_config["methods"][1]
    method["sweep"] = sweep
    return parse_config(minimal_config)


def test_sweep_expands_one_method_per_value(minimal_config):
    config = _swept(minimal_config, {"gamma": [0.9, 0.99]})
    runs = [r for r in config.runs if r.family == "ds-ts"]
    assert sorted({r.method for r in runs}) == ["ds-ts-gamma0.9", "ds-ts-gamma0.99"]
    run = config.get("sudden-s0__cold__ds-ts-gamma0.9__seed1")
    assert run.selector["params"] == {"gamma": 0.9, "budget": "${stream.cutoff}"}
    assert run.to_dict()["method"] == {
        "name": "ds-ts-gamma0.9",
        "selector": run.selector,
        "family": "ds-ts",
        "sweep": {"gamma": 0.9},
    }
    unswept = config.get("sudden-s0__warm20__static__seed0")
    assert set(unswept.to_dict()["method"]) == {"name", "selector"}
    assert len(config.runs) == 2 + 2 * 2 * 2  # static: warm only


def test_sweep_over_nested_paths_is_a_product(minimal_config):
    minimal_config["methods"][1]["selector"]["params"]["gate"] = {
        "class": "asf.online.gate.UncertaintyGate",
        "params": {"rate": 0.1},
    }
    config = _swept(
        minimal_config, {"gate.params.rate": [0.1, 0.5], "gate.params.n": [1, None]}
    )
    names = sorted({r.method for r in config.runs if r.family == "ds-ts"})
    assert names == [
        "ds-ts-rate0.1-n1",
        "ds-ts-rate0.1-nNone",
        "ds-ts-rate0.5-n1",
        "ds-ts-rate0.5-nNone",
    ]
    run = config.get("sudden-s0__cold__ds-ts-rate0.5-nNone__seed0")
    assert run.selector["params"]["gate"]["params"] == {"rate": 0.5, "n": None}
    other = config.get("sudden-s0__cold__ds-ts-rate0.1-n1__seed0")
    assert other.selector["params"]["gate"]["params"] == {"rate": 0.1, "n": 1}


@pytest.mark.parametrize(
    ("sweep", "message"),
    [
        ({}, "non-empty mapping"),
        ({"gamma": 0.9}, "non-empty list"),
        ({"gamma": []}, "non-empty list"),
        ({"gamma": [{"a": 1}]}, "plain values"),
        ({"gamma": [0.9, 0.9]}, "duplicates"),
        ({"gate.params.rate": [0.1]}, "does not lead to a mapping"),
        ({"gamma": ["a b"]}, "generated name"),
    ],
)
def test_bad_sweeps_are_rejected(minimal_config, sweep, message):
    with pytest.raises(ConfigError, match=message):
        _swept(minimal_config, sweep)


def test_swept_name_clashing_with_a_method_is_rejected(minimal_config):
    minimal_config["methods"][0]["name"] = "ds-ts-gamma0.9"
    with pytest.raises(ConfigError, match="duplicates"):
        _swept(minimal_config, {"gamma": [0.9]})
