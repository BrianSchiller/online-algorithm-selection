"""End to end: every run of the example config, then load the experiment.

``test_example_config_end_to_end`` needs ASF with the F1 interfaces
(``OnlineResult.save`` / ``load``) and is skipped until they are installed.
Until then ``test_example_config_with_stand_in`` runs the same flow with the
schema-v1 stand-in around today's evaluator.
"""

from __future__ import annotations

import time

import pytest

from conftest import EXAMPLE_CONFIG, FIXTURE_RESULTS
from oas_experiments import io, run
from oas_experiments import list as list_cli
from oas_experiments.config import load_config
from oas_experiments.results import load_experiment
from support.schema_v1 import STEP_COLUMNS

needs_f1 = pytest.mark.skipif(
    not io.asf_supports_results(),
    reason="ASF has no OnlineResult.save/load yet (F1 not merged)",
)


def _run_example_and_check(tmp_path, capsys) -> None:
    started = time.perf_counter()
    root = tmp_path / "results"
    config = load_config(EXAMPLE_CONFIG)
    for run_id in config.run_ids:
        code = run.main(
            [str(EXAMPLE_CONFIG), "--run-id", run_id, "--results-dir", str(root)]
        )
        assert code == 0, run_id

    capsys.readouterr()
    args = [str(EXAMPLE_CONFIG), "--missing", "--count", "--results-dir", str(root)]
    assert list_cli.main(args) == 0
    assert capsys.readouterr().out.strip() == "0"

    results = load_experiment(io.experiment_dir(root, config.name))
    runs, steps = results.runs, results.steps
    assert sorted(runs["run_id"]) == sorted(config.run_ids)
    assert len(runs) == 15
    assert runs.groupby(["method", "protocol_type"]).size().to_dict() == {
        ("ds-ts", "cold"): 3,
        ("ds-ts", "warm"): 3,
        ("rook-pairwise", "cold"): 3,
        ("rook-pairwise", "warm"): 3,
        ("static-pairwise", "warm"): 3,
    }
    by_protocol = runs.groupby("protocol_type")["n_evaluated"].unique()
    assert by_protocol["warm"].tolist() == [300]
    assert by_protocol["cold"].tolist() == [400]
    assert len(steps) == 9 * 300 + 6 * 400
    assert set(STEP_COLUMNS) <= set(steps.columns)

    runs_per_instance = runs.groupby("method")["mean_runs_per_instance"].mean()
    assert runs_per_instance["rook-pairwise"] == 2.0  # always one extra run
    assert runs_per_instance["ds-ts"] == 1.0
    assert runs_per_instance["static-pairwise"] == 1.0
    assert runs["closed_gap"].notna().all()
    assert (runs["closed_gap"] <= 1.0).all()
    assert (runs["total_compute_cost"] > 0).all()
    assert (runs["result_mode"] == "best_of_scheduled").all()
    assert runs["repo_commit"].str.len().eq(40).all()
    assert runs["param.gamma"].notna().sum() == 6

    assert time.perf_counter() - started < 60, "the end-to-end test must stay fast"


@needs_f1
def test_example_config_end_to_end(tmp_path, capsys):
    _run_example_and_check(tmp_path, capsys)


def test_example_config_with_stand_in(stand_in_f1, tmp_path, capsys):
    _run_example_and_check(tmp_path, capsys)


@needs_f1
def test_real_asf_reads_the_agreed_fixtures():
    """The hand-made schema-v1 fixtures must load with the real OnlineResult."""
    results = load_experiment(FIXTURE_RESULTS / "fixture_exp")
    assert len(results.runs) == 10
    assert len(results.steps) == 4 * 40 + 6 * 30
