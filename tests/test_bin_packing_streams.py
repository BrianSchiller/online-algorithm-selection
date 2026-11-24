"""The ``bin_packing`` stream kind and the maximize guard, on toy fitness data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from oas_experiments.build import BuildError, build_selector
from oas_experiments.config import parse_config
from oas_experiments.streams import build_stream

HEURISTICS = ["BF", "FF", "NF", "WF"]


@pytest.fixture(autouse=True)
def fake_bin_packing(monkeypatch: pytest.MonkeyPatch) -> None:
    import asf.online.streams.loaders as loaders

    def load(path: str) -> tuple[pd.DataFrame, pd.DataFrame, float]:
        rng = np.random.default_rng(0)
        idx = pd.Index([f"E_{h}{i:04d}" for h in HEURISTICS for i in range(10)])
        # every heuristic is best on its own 10 instances
        performance = pd.DataFrame(rng.uniform(0.5, 0.7, (40, 4)), idx, HEURISTICS)
        for h in HEURISTICS:
            performance.loc[idx.str.contains(h), h] = 0.9
        features = pd.DataFrame({"ts__sum_values": rng.uniform(1, 9, 40)}, idx)
        return features, performance, float("inf")

    monkeypatch.setattr(loaders, "load_bin_packing", load)


def _config(minimal_config: dict, drift: dict | None = None) -> dict:
    entry = {"name": "bpp", "kind": "bin_packing", "params": {"root": "bpp"}}
    if drift is not None:
        entry["drift"] = drift
    minimal_config["streams"] = [entry]
    return minimal_config


def test_best_algorithm_labels_use_the_highest_fitness(minimal_config: dict) -> None:
    # Rook's sudden stream: a mix of BF, FF, WF, then NF
    drift = {
        "builder": "grouped",
        "params": {"schedule": [["BF", "FF", "WF"], "NF"]},  # default labels
    }
    stream = build_stream(parse_config(_config(minimal_config, drift)).runs[0])
    winners = stream.performance.idxmax(axis=1).tolist()
    assert set(winners[:30]) == {"BF", "FF", "WF"} and winners[30:] == ["NF"] * 10
    assert stream.drift_points == (30,) and stream.cutoff == np.inf


def test_sorted_by_max_fitness_descending(minimal_config: dict) -> None:
    key = {"runtime": ["FF", "WF"], "agg": "max"}
    drift = {"builder": "sorted", "params": {"key": key, "ascending": False}}
    stream = build_stream(parse_config(_config(minimal_config, drift)).runs[0])
    assert (np.diff(stream.performance[["FF", "WF"]].max(axis=1)) <= 0).all()


def test_scenario_labels_and_unknown_params_are_rejected(minimal_config: dict) -> None:
    drift = {"builder": "grouped", "params": {"labels": "scenario", "schedule": [0]}}
    with pytest.raises(ValueError, match="labels must be"):
        build_stream(parse_config(_config(minimal_config, drift)).runs[0])
    cfg = _config(minimal_config)
    cfg["streams"][0]["params"]["scenarios"] = ["x"]
    with pytest.raises(ValueError, match="scenarios"):
        build_stream(parse_config(cfg).runs[0])


def _ds_ts_run(cfg: dict) -> Any:
    return next(r for r in parse_config(cfg).runs if r.method == "ds-ts")


def test_selectors_must_maximize(minimal_config: dict) -> None:
    run = _ds_ts_run(_config(minimal_config))
    stream = build_stream(run)
    with pytest.raises(BuildError, match="maximize: true"):
        build_selector(run, stream)
    minimal_config["methods"][1]["selector"]["params"]["maximize"] = True
    run = _ds_ts_run(minimal_config)
    assert build_selector(run, stream).maximize
