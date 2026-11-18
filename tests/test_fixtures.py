"""The fixture results must follow OnlineResult schema v1 exactly."""

from __future__ import annotations

import json
import math

import pandas as pd
import pyarrow.parquet as pq
import pytest
from asf.online.evaluation import STEP_COLUMNS

from conftest import FIXTURE_RESULTS
from oas_experiments.config import load_config

META_KEYS = {
    "schema_version",
    "asf",
    "selector",
    "stream",
    "protocol",
    "n_warmstart",
    "seed",
    "result_mode",
    "par_factor",
    "maximize",
    "sbs_algorithm_stream",
    "sbs_algorithm_warmstart",
    "retrain_steps",
    "drift_steps",
    "retrain_seconds",
    "created_utc",
    "extra",
}
FIXTURE_CONFIG = load_config(FIXTURE_RESULTS.parent / "fixture_exp.yaml")
RUN_DIRS = sorted((FIXTURE_RESULTS / "fixture_exp").iterdir())
PARQUET_TYPES = {
    "string": {"string", "large_string"},
    "int64": {"int64"},
    "float64": {"double"},
    "bool": {"bool"},
}


def test_fixtures_cover_the_fixture_config():
    assert [p.name for p in RUN_DIRS] == sorted(FIXTURE_CONFIG.run_ids)


@pytest.mark.parametrize("run_dir", RUN_DIRS, ids=lambda p: p.name)
def test_fixture_follows_schema_v1(run_dir):
    assert sorted(p.name for p in run_dir.iterdir()) == ["meta.json", "steps.parquet"]

    def no_constants(name):
        raise AssertionError(f"meta.json contains {name}")

    meta = json.loads((run_dir / "meta.json").read_text(), parse_constant=no_constants)
    assert set(meta) == META_KEYS
    assert meta["schema_version"] == 1
    assert meta["protocol"] == ("warm" if meta["n_warmstart"] else "cold")
    assert (meta["sbs_algorithm_warmstart"] is None) == (meta["protocol"] == "cold")
    assert meta["extra"]["run"] == FIXTURE_CONFIG.get(run_dir.name).to_dict()

    schema = pq.read_schema(run_dir / "steps.parquet")
    assert schema.names == list(STEP_COLUMNS)
    for column, kind in STEP_COLUMNS.items():
        assert str(schema.field(column).type) in PARQUET_TYPES[kind], column

    steps = pd.read_parquet(run_dir / "steps.parquet")
    stream = meta["stream"]
    assert len(steps) == stream["n_evaluated"]
    assert stream["n_evaluated"] == stream["n_instances"] - meta["n_warmstart"]
    assert steps["step"].tolist() == list(range(len(steps)))
    assert steps["sbs_cost_warmstart"].isna().all() == (meta["protocol"] == "cold")
    assert (steps["user_cost"] == steps[f"cost_{meta['result_mode']}"]).all()
    assert (steps["vbs_cost"] <= steps["cost_best_of_scheduled"]).all()
    assert steps.index[steps["retrained"]].tolist() == meta["retrain_steps"]
    assert steps.index[steps["drift_signal"]].tolist() == meta["drift_steps"]
    for row in steps.itertuples():
        runs = json.loads(row.runs)
        assert list(runs) == [row.primary, *json.loads(row.extras)]
        assert row.n_runs == len(runs)
        assert math.isclose(row.compute_cost, sum(r["runtime"] for r in runs.values()))
        assert runs[row.primary]["censored"] == row.primary_censored
