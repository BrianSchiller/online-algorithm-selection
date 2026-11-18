from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
from asf.online.evaluation import STEP_COLUMNS

from conftest import FIXTURE_RESULTS
from oas_experiments.results import (
    ResultError,
    SchemaVersionError,
    collect_runs,
    collect_steps,
    flatten_params,
    load_experiment,
)

EXPERIMENT = FIXTURE_RESULTS / "fixture_exp"
RUN = "sudden-s0__cold__rook__seed0"


@pytest.fixture
def experiment_copy(tmp_path):
    return shutil.copytree(EXPERIMENT, tmp_path / "fixture_exp")


def _edit_meta(run_dir, edit):
    meta = json.loads((run_dir / "meta.json").read_text())
    edit(meta)
    (run_dir / "meta.json").write_text(json.dumps(meta))


def test_runs_frame_has_one_row_per_run_with_config_columns():
    runs = collect_runs(EXPERIMENT)
    assert len(runs) == 10
    assert runs["run_id"].is_unique
    assert set(runs["experiment"]) == {"fixture_exp"}
    assert (runs["method_family"] == runs["method"]).all()  # nothing swept
    counts = runs.groupby(["method", "protocol_type"]).size().to_dict()
    assert counts == {
        ("ds-ts", "cold"): 2,
        ("ds-ts", "warm"): 2,
        ("rook", "cold"): 2,
        ("rook", "warm"): 2,
        ("static", "warm"): 2,
    }
    row = runs.set_index("run_id").loc[RUN]
    assert row["method"] == "rook"
    assert row["selector_class"] == "asf.online.selectors.OnlinePairwiseSelector"
    assert (row["stream"], row["stream_kind"], row["stream_seed"]) == (
        "sudden",
        "synthetic_scenario",
        0,
    )
    assert (row["protocol"], row["n_warmstart"], row["seed"]) == ("cold", 0, 0)
    assert (row["result_mode"], row["par_factor"]) == ("best_of_scheduled", 10.0)
    assert (row["cutoff"], row["n_evaluated"]) == (100.0, 40)
    assert row["repo_commit"] == "1" * 40 and not row["repo_dirty"]
    assert row["asf_commit"] == "0" * 40 and not row["final"]


def test_runs_frame_metrics_match_the_steps():
    results = load_experiment(EXPERIMENT)
    row = results.runs.set_index("run_id").loc[RUN]
    steps = results.steps[results.steps["run_id"] == RUN]
    assert row["mean_user_cost"] == pytest.approx(steps["user_cost"].mean())
    assert row["mean_runs_per_instance"] == 2.0
    assert row["total_compute_cost"] == pytest.approx(steps["compute_cost"].sum())
    assert row["total_user_cost"] == pytest.approx(steps["user_cost"].sum())
    assert row["total_selector_seconds"] == pytest.approx(
        steps["selector_seconds"].sum()
    )
    assert row["total_detector_seconds"] > 0
    gap = (steps["sbs_cost"].mean() - steps["user_cost"].mean()) / (
        steps["sbs_cost"].mean() - steps["vbs_cost"].mean()
    )
    assert row["closed_gap"] == pytest.approx(gap)
    assert row["n_retrains"] == 1
    assert np.isnan(row["closed_gap_warmstart"])  # cold start


def test_runs_frame_has_flattened_method_params():
    runs = collect_runs(EXPERIMENT).set_index("run_id")
    rook = runs.loc[RUN]
    assert rook["param.n_additional"] == 1
    assert rook["param.detector.class"].endswith("PCAReconstructionDetector")
    assert rook["param.detector.chunk_size"] == 5
    bandit = runs.loc["sudden-s0__warm10__ds-ts__seed1"]
    assert bandit["param.gamma"] == 0.95
    assert np.isnan(bandit["param.n_additional"])
    static = runs.loc["sudden-s0__warm10__static__seed1"]
    assert static["param.selector.class"] == "asf.selectors.PairwiseClassifier"
    assert static["param.selector.random_state"] == 1
    assert static["param.selector.model_class"].endswith("DecisionTreeClassifier")


def test_flatten_params():
    assert flatten_params(
        {"a": 1, "b": {"type": "x.Y"}, "c": [1, 2], "d": {"e": {"f": None}}}
    ) == {"param.a": 1, "param.b": "x.Y", "param.c": "[1, 2]", "param.d.e.f": None}


def test_steps_frame_is_long_and_joins_on_run_id():
    results = load_experiment(EXPERIMENT)
    steps = results.steps
    assert list(steps.columns) == ["run_id", *STEP_COLUMNS]
    assert len(steps) == 4 * 40 + 6 * 30
    assert steps.groupby("run_id").size().to_dict() == (
        results.runs.set_index("run_id")["n_evaluated"].to_dict()
    )
    merged = steps.merge(results.runs[["run_id", "method"]], on="run_id")
    assert merged.groupby("method")["n_runs"].mean().to_dict() == {
        "ds-ts": 1.0,
        "rook": 2.0,
        "static": 1.0,
    }


def test_collect_steps_column_subset():
    steps = collect_steps(EXPERIMENT, columns=["step", "user_cost"])
    assert list(steps.columns) == ["run_id", "step", "user_cost"]


def test_unknown_schema_version_is_rejected(experiment_copy):
    _edit_meta(experiment_copy / RUN, lambda meta: meta.update(schema_version=0))
    with pytest.raises(SchemaVersionError, match="schema_version 0"):
        collect_runs(experiment_copy)


def test_newer_schema_version_is_rejected(experiment_copy):
    _edit_meta(experiment_copy / RUN, lambda meta: meta.update(schema_version=2))
    with pytest.raises(Exception, match="schema_version"):
        collect_runs(experiment_copy)


def test_result_without_runner_metadata_is_rejected(experiment_copy):
    _edit_meta(experiment_copy / RUN, lambda meta: meta.update(extra={}))
    with pytest.raises(ResultError, match="meta.extra.run"):
        collect_runs(experiment_copy)


def test_renamed_run_directory_is_rejected(experiment_copy):
    (experiment_copy / RUN).rename(experiment_copy / "renamed")
    with pytest.raises(ResultError, match="does not match"):
        collect_runs(experiment_copy)


def test_incomplete_and_temporary_directories(experiment_copy):
    (experiment_copy / f"{RUN}.tmp-123").mkdir()
    assert len(collect_runs(experiment_copy)) == 10  # temporary: ignored silently
    (experiment_copy / RUN / "steps.parquet").unlink()
    with pytest.warns(UserWarning, match="1 incomplete"):
        assert len(collect_runs(experiment_copy)) == 9


def test_empty_or_missing_experiment(tmp_path):
    with pytest.raises(ResultError, match="no results"):
        collect_runs(tmp_path)
    with pytest.raises(ResultError, match="not a directory"):
        collect_runs(tmp_path / "nope")
